from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from .contracts import validate_w8a8_target_config


SCHEMA = "formal-glm53-w8a8-target-io-v2"
_EMBED_KEYS = ("model.embed_tokens.weight", "model.model.embed_tokens.weight")
_HEAD_KEYS = ("lm_head.weight", "model.lm_head.weight")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    found = False
    for name in (
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
    ):
        path = root / name
        if path.is_file():
            found = True
            digest.update(name.encode() + b"\0" + path.read_bytes())
    if not found:
        raise FileNotFoundError(f"no tokenizer artifacts found under {root}")
    return digest.hexdigest()


def _checkpoint_fingerprint(root: Path) -> str:
    """Bind metadata and the content of every weight shard."""

    digest = hashlib.sha256()
    for path in sorted(root.glob("*.json")):
        if (
            path.name == "config.json"
            or path.name == "quant_model_description.json"
            or path.name.endswith(".index.json")
        ):
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
    shards = sorted(root.rglob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors found under {root}")
    for path in shards:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode() + b"\0" + _sha256(path).encode() + b"\0")
    return digest.hexdigest()


def _weight_map(root: Path) -> dict[str, str]:
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if indexes:
        value = json.loads(indexes[0].read_text(encoding="utf-8"))
        mapping = value.get("weight_map")
        if not isinstance(mapping, Mapping):
            raise ValueError("invalid safetensors index")
        return {str(key): str(filename) for key, filename in mapping.items()}
    mapping: dict[str, str] = {}
    for path in sorted(root.rglob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in mapping:
                    raise ValueError(f"duplicate tensor {key}")
                mapping[key] = path.relative_to(root).as_posix()
    return mapping


def _safe_weight_path(root: Path, filename: str) -> Path:
    path = (root / str(filename)).resolve()
    if path == root.resolve() or root.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(f"weight shard is not a file under {root}: {filename}")
    return path


def _resolve(
    mapping: Mapping[str, str], candidates: tuple[str, ...], explicit: str | None
) -> str:
    if explicit:
        if explicit not in mapping:
            raise KeyError(explicit)
        return explicit
    for key in candidates:
        if key in mapping:
            return key
    raise KeyError(f"none of {candidates} exists")


def _load(root: Path, mapping: Mapping[str, str], key: str) -> torch.Tensor:
    with safe_open(_safe_weight_path(root, mapping[key]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).detach().cpu().contiguous()


def extract_w8a8_target_io(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_hidden_size: int = 6144,
    expected_vocab_size: int = 154880,
    embed_key: str | None = None,
    lm_head_key: str | None = None,
) -> dict[str, Any]:
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # ModelSlim keeps the authoritative per-tensor W8A8 ABI in this sidecar;
    # it is not normally copied into Hugging Face ``config.json``.  Accept the
    # compact config marker used by smoke fixtures as well, but production
    # ModelSlim exports are validated from the sidecar itself.
    description_path = model_dir / "quant_model_description.json"
    quant_description: Mapping[str, Any] | None = None
    if description_path.is_file():
        quant_description = json.loads(description_path.read_text(encoding="utf-8"))
        if not isinstance(quant_description, Mapping):
            raise ValueError("quant_model_description.json must contain an object")
    validate_w8a8_target_config(
        config,
        expected_hidden_size=expected_hidden_size,
        expected_vocab_size=expected_vocab_size,
        quant_description=quant_description,
    )
    mapping = _weight_map(model_dir)
    resolved_embed = _resolve(mapping, _EMBED_KEYS, embed_key)
    resolved_head = _resolve(mapping, _HEAD_KEYS, lm_head_key)
    embed = _load(model_dir, mapping, resolved_embed)
    head = _load(model_dir, mapping, resolved_head)
    expected_shape = (int(expected_vocab_size), int(expected_hidden_size))
    if tuple(embed.shape) != expected_shape or tuple(head.shape) != expected_shape:
        raise ValueError(
            f"target I/O shape must be {expected_shape}; got "
            f"{tuple(embed.shape)}/{tuple(head.shape)}"
        )
    if embed.dtype != torch.bfloat16 or head.dtype != torch.bfloat16:
        raise ValueError("target embedding and lm_head must both be dense BF16 tensors")
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = output_dir / "model.safetensors"
    save_file({"embed_tokens.weight": embed, "lm_head.weight": head}, weight_path)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "target_quantization": "W8A8",
        "runtime_backend": "vllm-ascend",
        "storage_dtype": "bfloat16",
        "source_model_dir": str(model_dir),
        "source_model_fingerprint": _checkpoint_fingerprint(model_dir),
        "tokenizer_fingerprint": _tokenizer_fingerprint(model_dir),
        "config_sha256": _sha256(config_path),
        "hidden_size": int(expected_hidden_size),
        "vocab_size": int(expected_vocab_size),
        "source_keys": {"embed_tokens": resolved_embed, "lm_head": resolved_head},
        "source_dtypes": {"embed_tokens": str(embed.dtype), "lm_head": str(head.dtype)},
        "weights_file": weight_path.name,
        "weights_sha256": _sha256(weight_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


@dataclass(frozen=True)
class FrozenTargetIO:
    embed_tokens: nn.Embedding
    lm_head: nn.Linear
    manifest: Mapping[str, Any]


def load_frozen_target_io(
    output_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> FrozenTargetIO:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("target_quantization") != "W8A8":
        raise ValueError("unsupported or non-W8A8 target I/O artifact")
    if manifest.get("storage_dtype") != "bfloat16":
        raise ValueError("target I/O storage must be BF16")
    required_identity = (
        "source_model_dir",
        "source_model_fingerprint",
        "tokenizer_fingerprint",
        "config_sha256",
        "weights_file",
        "weights_sha256",
    )
    if any(not manifest.get(key) for key in required_identity):
        raise ValueError("target I/O manifest is missing identity provenance")
    weight_path = _safe_weight_path(output_dir, str(manifest["weights_file"]))
    if _sha256(weight_path) != manifest.get("weights_sha256"):
        raise ValueError("target I/O checksum mismatch")
    tensors = load_file(weight_path, device="cpu")
    embed_weight = tensors["embed_tokens.weight"]
    head_weight = tensors["lm_head.weight"]
    if embed_weight.dtype != torch.bfloat16 or head_weight.dtype != torch.bfloat16:
        raise ValueError("target I/O tensors are not BF16")
    expected = (int(manifest["vocab_size"]), int(manifest["hidden_size"]))
    if tuple(embed_weight.shape) != expected or tuple(head_weight.shape) != expected:
        raise ValueError("target I/O shape mismatch")
    embed = nn.Embedding.from_pretrained(
        embed_weight.to(device=device, dtype=dtype), freeze=True
    )
    head = nn.Linear(expected[1], expected[0], bias=False, device="meta")
    head.weight = nn.Parameter(
        head_weight.to(device=device, dtype=dtype), requires_grad=False
    )
    embed.eval()
    head.eval()
    return FrozenTargetIO(embed, head, manifest)
