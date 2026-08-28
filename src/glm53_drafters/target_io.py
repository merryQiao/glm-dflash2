from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from glm53_stage_a.provenance import (
    local_model_fingerprint,
    model_revision,
    tokenizer_fingerprint,
)

from .contracts import TARGET_CONTRACT
from .hidden_capture import validate_ascend_a2_evidence


TARGET_IO_SCHEMA = "glm53-target-io-v3"
PRODUCTION_MASK_TOKEN = "[MASK]"
PRODUCTION_MASK_TOKEN_ID = 154821
_EMBED_KEYS = (
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "transformer.embedding.word_embeddings.weight",
    "embed_tokens.weight",
)
_HEAD_KEYS = (
    "lm_head.weight",
    "model.language_model.lm_head.weight",
    "model.lm_head.weight",
    "output_layer.weight",
)


@dataclass(frozen=True)
class TargetCheckpointContract:
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    rms_norm_eps: float
    dtype: str

    def __post_init__(self) -> None:
        if self.model_type != "glm5_next":
            raise ValueError("target checkpoint contract must be GLM5Next")
        if min(self.num_hidden_layers, self.hidden_size, self.vocab_size) < 1:
            raise ValueError("target checkpoint dimensions must be positive")
        if self.rms_norm_eps != 1e-5:
            raise ValueError("target checkpoint RMS epsilon must be 1e-5")
        if self.dtype != "bfloat16":
            raise ValueError("target checkpoint dtype must be BF16")


PRODUCTION_TARGET_CHECKPOINT_CONTRACT = TargetCheckpointContract(
    model_type="glm5_next",
    num_hidden_layers=TARGET_CONTRACT.num_layers,
    hidden_size=TARGET_CONTRACT.hidden_size,
    vocab_size=TARGET_CONTRACT.vocab_size,
    rms_norm_eps=TARGET_CONTRACT.rms_norm_eps,
    dtype="bfloat16",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = (
        value.detach()
        .cpu()
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config.json is not an object")
    return value


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, Mapping) else config


def validate_target_checkpoint_config(
    config: Mapping[str, Any],
    *,
    contract: TargetCheckpointContract = PRODUCTION_TARGET_CHECKPOINT_CONTRACT,
) -> dict[str, Any]:
    """Resolve and validate the immutable unquantized GLM5Next text contract."""

    text = _text_config(config)
    root_model_type = config.get("model_type")
    text_model_type = text.get("model_type")
    if root_model_type != contract.model_type:
        raise ValueError(
            f"target root architecture must be {contract.model_type}, "
            f"got {root_model_type!r}"
        )
    expected_text_model_type = f"{contract.model_type}_text"
    if text_model_type != expected_text_model_type:
        raise ValueError(
            f"target text architecture must be {expected_text_model_type}, "
            f"got {text_model_type!r}"
        )
    if "dtype" not in text:
        raise ValueError("target text config must explicitly declare dtype")
    dtype_value = text.get("dtype")
    legacy_dtype = text.get("torch_dtype")
    if legacy_dtype is not None and str(legacy_dtype).lower() != str(dtype_value).lower():
        raise ValueError("target text dtype and torch_dtype conflict")
    resolved = {
        "model_type": root_model_type,
        "text_model_type": text_model_type,
        "num_hidden_layers": text.get(
            "num_hidden_layers", config.get("num_hidden_layers")
        ),
        "hidden_size": text.get("hidden_size", config.get("hidden_size")),
        "vocab_size": text.get("vocab_size", config.get("vocab_size")),
        "rms_norm_eps": text.get("rms_norm_eps", config.get("rms_norm_eps")),
        "dtype": dtype_value,
        "tie_word_embeddings": text.get(
            "tie_word_embeddings", config.get("tie_word_embeddings", False)
        ),
    }
    if "tie_word_embeddings" not in text:
        raise ValueError("target text config must explicitly declare tie_word_embeddings")
    integer_fields = (
        ("num_hidden_layers", contract.num_hidden_layers, "decoder layers"),
        ("hidden_size", contract.hidden_size, "hidden size"),
        ("vocab_size", contract.vocab_size, "vocab size"),
    )
    for field, expected, label in integer_fields:
        value = resolved[field]
        if not isinstance(value, int) or isinstance(value, bool) or value != expected:
            raise ValueError(f"target text {label} must be {expected}, got {value!r}")
    if resolved["rms_norm_eps"] != contract.rms_norm_eps:
        raise ValueError(
            f"target text RMS epsilon must be {contract.rms_norm_eps}, "
            f"got {resolved['rms_norm_eps']!r}"
        )
    dtype = str(resolved["dtype"] or "").lower()
    if dtype != contract.dtype:
        raise ValueError(f"target text checkpoint must advertise BF16, got {dtype!r}")
    quantization_keys = (
        "quantization_config",
        "quantization",
        "quant_method",
        "load_in_4bit",
        "load_in_8bit",
    )
    if resolved["tie_word_embeddings"] is not False:
        raise ValueError("GLM-5.3 target embedding and LM head must be explicitly untied")
    for location, values in (("root", config), ("text", text)):
        for key in quantization_keys:
            if key in values:
                raise ValueError(
                    "quantization metadata is forbidden for the production target: "
                    f"{location}.{key}"
                )
    return {
        **asdict(contract),
        "text_model_type": expected_text_model_type,
        "tie_word_embeddings": False,
    }


def resolve_mask_token_identity(
    model_dir: str | Path,
    *,
    expected_token: str = PRODUCTION_MASK_TOKEN,
    expected_token_id: int | None = None,
) -> dict[str, Any]:
    """Resolve the exact special diffusion mask from tokenizer.json."""

    tokenizer_path = Path(model_dir) / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"tokenizer.json is required to resolve {expected_token!r}: {tokenizer_path}"
        )
    value = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    added_tokens = value.get("added_tokens") if isinstance(value, Mapping) else None
    if not isinstance(added_tokens, list):
        raise ValueError("tokenizer.json has no added_tokens list")
    matches = [
        item
        for item in added_tokens
        if isinstance(item, Mapping) and item.get("content") == expected_token
    ]
    if len(matches) != 1:
        raise ValueError(
            f"target tokenizer must contain exactly one {expected_token!r} token"
        )
    item = matches[0]
    token_id = item.get("id")
    if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
        raise ValueError(f"{expected_token!r} token ID is invalid: {token_id!r}")
    if item.get("special") is not True:
        raise ValueError(f"{expected_token!r} must be marked special")
    if expected_token_id is not None and token_id != expected_token_id:
        raise ValueError(
            f"{expected_token!r} token ID must be {expected_token_id}, got {token_id}"
        )
    return {"token": expected_token, "token_id": token_id, "special": True}


def target_io_artifact_identity(manifest: Mapping[str, Any]) -> str:
    """Hash the path-independent immutable target-I/O contract."""

    fields = {
        key: manifest.get(key)
        for key in (
            "schema",
            "status",
            "source_model_fingerprint",
            "model_revision",
            "tokenizer_fingerprint",
            "hidden_size",
            "vocab_size",
            "dtype",
            "untied",
            "lm_head_bias",
            "logit_transform",
            "target_checkpoint_contract",
            "mask_token",
            "tensors",
            "weights_sha256",
        )
    }
    raw = json.dumps(
        fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _local_model_identity(
    model_dir: str | Path, contract: TargetCheckpointContract
) -> dict[str, Any]:
    model_dir = Path(model_dir).expanduser().resolve()
    config = _config(model_dir)
    resolved = validate_target_checkpoint_config(config, contract=contract)
    fingerprint = local_model_fingerprint(model_dir)
    return {
        "model_fingerprint": fingerprint,
        "model_revision": model_revision(config, fingerprint),
        "tokenizer_fingerprint": tokenizer_fingerprint(model_dir),
        "hidden_size": resolved["hidden_size"],
        "vocab_size": resolved["vocab_size"],
    }


def local_model_identity(model_dir: str | Path) -> dict[str, Any]:
    return _local_model_identity(model_dir, PRODUCTION_TARGET_CHECKPOINT_CONTRACT)


def _weight_map(model_dir: Path) -> dict[str, str]:
    indexes = sorted(model_dir.glob("*.safetensors.index.json"))
    if indexes:
        value = json.loads(indexes[0].read_text(encoding="utf-8"))
        mapping = value.get("weight_map")
        if not isinstance(mapping, dict):
            raise ValueError(f"invalid weight_map in {indexes[0]}")
        return {str(key): str(filename) for key, filename in mapping.items()}
    mapping: dict[str, str] = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in mapping:
                    raise ValueError(f"duplicate tensor key: {key}")
                mapping[key] = path.name
    if not mapping:
        raise FileNotFoundError(f"no safetensors weights under {model_dir}")
    return mapping


def _resolve_key(
    mapping: Mapping[str, str], candidates: tuple[str, ...], suffix: str, label: str
) -> str:
    for key in candidates:
        if key in mapping:
            return key
    matches = [key for key in mapping if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"could not resolve a unique {label}: {matches}")
    return matches[0]


def _load_tensor(model_dir: Path, mapping: Mapping[str, str], key: str) -> torch.Tensor:
    path = model_dir / mapping[key]
    if not path.is_file():
        raise FileNotFoundError(f"weight shard for {key!r} is missing: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).detach().cpu().contiguous()


def _validate_identity_logits(config: Mapping[str, Any], mapping: Mapping[str, str]) -> None:
    bias = [
        key
        for key in mapping
        if key.endswith("lm_head.bias") or key.endswith("output_layer.bias")
    ]
    if bias:
        raise ValueError(f"LM-head bias is unsupported: {bias}")
    locations = [("config", config)]
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping):
        locations.append(("text_config", text_config))
    for location, values in locations:
        for key in ("logit_scale", "output_logits_scale", "logits_scaling"):
            value = values.get(key)
            if value is not None and float(value) != 1.0:
                raise ValueError(f"non-identity {location}.{key}={value}")
        for key in (
            "final_logit_softcapping",
            "logits_soft_cap",
            "logit_softcap",
        ):
            value = values.get(key)
            if value is not None and float(value) != 0.0:
                raise ValueError(f"non-identity softcap {location}.{key}={value}")


def _extract_target_io(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    contract: TargetCheckpointContract,
    cache_manifest: Mapping[str, Any] | None = None,
    expected_layer_ids: tuple[int, ...] = TARGET_CONTRACT.logical_layer_ids,
) -> dict[str, Any]:
    model_dir = Path(model_dir).expanduser().resolve()
    output_dir = Path(output_dir)
    config = _config(model_dir)
    resolved = validate_target_checkpoint_config(config, contract=contract)
    hidden_size = int(resolved["hidden_size"])
    vocab_size = int(resolved["vocab_size"])
    if resolved["tie_word_embeddings"]:
        raise ValueError("GLM-5.3 target embedding and LM head must be untied")
    mapping = _weight_map(model_dir)
    _validate_identity_logits(config, mapping)
    embed_key = _resolve_key(
        mapping, _EMBED_KEYS, ".embed_tokens.weight", "embed_tokens"
    )
    head_key = _resolve_key(mapping, _HEAD_KEYS, ".lm_head.weight", "lm_head")
    if embed_key == head_key:
        raise ValueError("GLM-5.3 target embedding and LM head must be untied")
    embed = _load_tensor(model_dir, mapping, embed_key)
    head = _load_tensor(model_dir, mapping, head_key)
    expected_shape = (vocab_size, hidden_size)
    if tuple(embed.shape) != expected_shape or tuple(head.shape) != expected_shape:
        raise ValueError(
            f"target I/O shape must be {expected_shape}, got "
            f"embed={tuple(embed.shape)}, lm_head={tuple(head.shape)}"
        )
    if embed.dtype != torch.bfloat16 or head.dtype != torch.bfloat16:
        raise ValueError(
            "target embedding and LM head must be dense BF16 tensors, got "
            f"{embed.dtype} and {head.dtype}"
        )
    identity = _local_model_identity(model_dir, contract)
    mask_token = resolve_mask_token_identity(
        model_dir,
        expected_token_id=(
            PRODUCTION_MASK_TOKEN_ID
            if contract == PRODUCTION_TARGET_CHECKPOINT_CONTRACT
            else None
        ),
    )
    manifest: dict[str, Any] = {
        "schema": TARGET_IO_SCHEMA,
        "status": "frozen",
        "source_model_dir": str(model_dir),
        "source_model_fingerprint": identity["model_fingerprint"],
        "model_revision": identity["model_revision"],
        "tokenizer_fingerprint": identity["tokenizer_fingerprint"],
        "mask_token": mask_token,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "dtype": "bfloat16",
        "untied": True,
        "lm_head_bias": False,
        "logit_transform": "identity",
        "target_checkpoint_contract": asdict(contract),
        "source_keys": {"embed_tokens": embed_key, "lm_head": head_key},
        "source_dtypes": {
            "embed_tokens": str(embed.dtype),
            "lm_head": str(head.dtype),
        },
        "config_sha256": _sha256(model_dir / "config.json"),
        "tensors": {
            "embed_tokens": {
                "shape": list(embed.shape),
                "dtype": "bfloat16",
                "sha256": _tensor_sha256(embed),
            },
            "lm_head": {
                "shape": list(head.shape),
                "dtype": "bfloat16",
                "sha256": _tensor_sha256(head),
            },
        },
        "hidden_cache_identity": (
            str(cache_manifest.get("cache_identity"))
            if cache_manifest is not None
            else None
        ),
    }
    if cache_manifest is not None:
        _validate_cache_io_compatibility(
            cache_manifest,
            manifest,
            contract=contract,
            expected_layer_ids=expected_layer_ids,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = output_dir / "model.safetensors"
    temporary = output_dir / "model.safetensors.tmp"
    save_file(
        {"embed_tokens.weight": embed, "lm_head.weight": head}, temporary
    )
    os.replace(temporary, weights)
    manifest["weights_sha256"] = _sha256(weights)
    manifest["artifact_identity"] = target_io_artifact_identity(manifest)
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def extract_target_io(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    cache_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Production entrypoint with no contract override or downgrade path."""

    return _extract_target_io(
        model_dir,
        output_dir,
        contract=PRODUCTION_TARGET_CHECKPOINT_CONTRACT,
        cache_manifest=cache_manifest,
        expected_layer_ids=TARGET_CONTRACT.logical_layer_ids,
    )


class FrozenLinear(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight)


@dataclass(frozen=True)
class FrozenTargetIO:
    embed_tokens: nn.Embedding
    lm_head: FrozenLinear
    manifest: dict[str, Any]


def _load_frozen_target_io(
    output_dir: str | Path,
    *,
    contract: TargetCheckpointContract,
    dtype: torch.dtype = torch.bfloat16,
) -> FrozenTargetIO:
    if dtype != torch.bfloat16:
        raise ValueError("frozen GLM-5.3 target I/O must be loaded as BF16")
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TARGET_IO_SCHEMA or manifest.get("status") != "frozen":
        raise ValueError("target-I/O manifest is not frozen schema v3 with mask identity")
    if manifest.get("target_checkpoint_contract") != asdict(contract):
        raise ValueError(
            "target-I/O checkpoint contract is incompatible; expected "
            f"{contract.num_hidden_layers} layers, hidden {contract.hidden_size}, "
            f"vocab {contract.vocab_size}"
        )
    if (
        manifest.get("hidden_size") != contract.hidden_size
        or manifest.get("vocab_size") != contract.vocab_size
        or manifest.get("dtype") != contract.dtype
        or manifest.get("untied") is not True
        or manifest.get("lm_head_bias") is not False
        or manifest.get("logit_transform") != "identity"
    ):
        raise ValueError("target-I/O manifest differs from exact checkpoint contract")
    mask_token = manifest.get("mask_token")
    if not isinstance(mask_token, Mapping) or mask_token.get("token") != PRODUCTION_MASK_TOKEN:
        raise ValueError("target-I/O manifest has no exact [MASK] identity")
    token_id = mask_token.get("token_id")
    if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
        raise ValueError("target-I/O manifest has an invalid [MASK] token ID")
    if contract == PRODUCTION_TARGET_CHECKPOINT_CONTRACT and token_id != PRODUCTION_MASK_TOKEN_ID:
        raise ValueError(
            f"production target-I/O [MASK] token ID must be {PRODUCTION_MASK_TOKEN_ID}"
        )
    weights = output_dir / "model.safetensors"
    if _sha256(weights) != manifest.get("weights_sha256"):
        raise ValueError("target-I/O weights file checksum differs")
    tensors = load_file(weights, device="cpu")
    embed = tensors.get("embed_tokens.weight")
    head = tensors.get("lm_head.weight")
    if embed is None or head is None:
        raise ValueError("target-I/O weights are incomplete")
    for name, value in (("embed_tokens", embed), ("lm_head", head)):
        if _tensor_sha256(value) != manifest["tensors"][name]["sha256"]:
            raise ValueError(f"{name} tensor checksum differs")
        if value.dtype != torch.bfloat16:
            raise ValueError(f"{name} tensor is not BF16")
        if tuple(value.shape) != (contract.vocab_size, contract.hidden_size):
            raise ValueError(f"{name} tensor shape differs from checkpoint contract")
    if manifest.get("artifact_identity") != target_io_artifact_identity(manifest):
        raise ValueError("target-I/O artifact identity differs")
    embedding = nn.Embedding.from_pretrained(embed, freeze=True)
    return FrozenTargetIO(
        embed_tokens=embedding,
        lm_head=FrozenLinear(head),
        manifest=manifest,
    )


def load_frozen_target_io(
    output_dir: str | Path, *, dtype: torch.dtype = torch.bfloat16
) -> FrozenTargetIO:
    return _load_frozen_target_io(
        output_dir,
        contract=PRODUCTION_TARGET_CHECKPOINT_CONTRACT,
        dtype=dtype,
    )


def _validate_cache_io_compatibility(
    cache_manifest: Mapping[str, Any],
    io_manifest: Mapping[str, Any],
    *,
    contract: TargetCheckpointContract,
    expected_layer_ids: tuple[int, ...],
) -> None:
    if io_manifest.get("schema") != TARGET_IO_SCHEMA:
        raise ValueError("target-I/O schema is incompatible")
    if (
        io_manifest.get("status") != "frozen"
        or io_manifest.get("dtype") != contract.dtype
        or io_manifest.get("untied") is not True
        or io_manifest.get("lm_head_bias") is not False
        or io_manifest.get("logit_transform") != "identity"
    ):
        raise ValueError("target-I/O artifact differs from the exact frozen contract")
    if cache_manifest.get("status") != "frozen" or cache_manifest.get(
        "production_eligible"
    ) is not True:
        raise ValueError("hidden cache is not production frozen")
    cache_identity = str(cache_manifest.get("cache_identity") or "")
    if not cache_identity:
        raise ValueError("hidden cache has no immutable cache identity")
    bound_identity = io_manifest.get("hidden_cache_identity")
    if bound_identity not in (None, cache_identity):
        raise ValueError("target I/O is bound to a different hidden cache identity")
    spec = cache_manifest.get("spec")
    provenance = cache_manifest.get("provenance")
    if not isinstance(spec, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("hidden cache manifest is incomplete")
    if io_manifest.get("target_checkpoint_contract") != asdict(contract):
        raise ValueError(
            "target-I/O checkpoint contract is incompatible; expected "
            f"{contract.num_hidden_layers}/{contract.hidden_size}/{contract.vocab_size}"
        )
    if contract == PRODUCTION_TARGET_CHECKPOINT_CONTRACT:
        validate_ascend_a2_evidence(provenance.get("ascend_a2_runtime"))
    expected = list(expected_layer_ids)
    if spec.get("layer_ids") != expected or provenance.get(
        "logical_layer_ids"
    ) != expected:
        raise ValueError("hidden cache logical layer order is incompatible")
    if provenance.get("physical_layer_ids") != [layer + 1 for layer in expected]:
        raise ValueError("hidden cache physical layer mapping is incompatible")
    if int(spec.get("schema_version", -1)) != 2:
        raise ValueError("hidden cache must use schema v2")
    if spec.get("dtype") != "bfloat16" or provenance.get(
        "target_hidden_dtype"
    ) != "bfloat16":
        raise ValueError("hidden cache dtype is incompatible")
    if spec.get("mask_semantics") != "dflash_target_token":
        raise ValueError("hidden cache mask semantics are incompatible")
    if spec.get("final_hidden_semantics") != "post_final_norm_lm_head_input":
        raise ValueError("hidden cache final hidden semantics are incompatible")
    if int(spec.get("hidden_size", -1)) != int(io_manifest.get("hidden_size", -2)):
        raise ValueError("hidden cache width differs from target I/O")
    if int(spec.get("hidden_size", -1)) != contract.hidden_size:
        raise ValueError("hidden cache width differs from checkpoint contract")
    if int(spec.get("target_num_hidden_layers", -1)) != contract.num_hidden_layers:
        raise ValueError("hidden cache target depth differs from checkpoint contract")
    if int(spec.get("vocab_size", -1)) != contract.vocab_size:
        raise ValueError("hidden cache vocabulary differs from checkpoint contract")
    checks = (
        ("model_fingerprint", "source_model_fingerprint", "model fingerprint"),
        ("model_revision", "model_revision", "model revision"),
        ("tokenizer_fingerprint", "tokenizer_fingerprint", "tokenizer fingerprint"),
        ("vocab_size", "vocab_size", "vocab size"),
    )
    for cache_key, io_key, label in checks:
        if provenance.get(cache_key) != io_manifest.get(io_key):
            raise ValueError(f"hidden cache {label} differs from target I/O")


def validate_cache_io_compatibility(
    cache_manifest: Mapping[str, Any],
    io_manifest: Mapping[str, Any],
) -> None:
    _validate_cache_io_compatibility(
        cache_manifest,
        io_manifest,
        contract=PRODUCTION_TARGET_CHECKPOINT_CONTRACT,
        expected_layer_ids=TARGET_CONTRACT.logical_layer_ids,
    )
