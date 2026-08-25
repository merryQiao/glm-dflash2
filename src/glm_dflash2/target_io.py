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

from .provenance import local_model_fingerprint


_EMBED_KEYS = (
    "model.embed_tokens.weight",
    "model.model.embed_tokens.weight",
    "transformer.embedding.word_embeddings.weight",
    "embed_tokens.weight",
)
_HEAD_KEYS = (
    "lm_head.weight",
    "model.lm_head.weight",
    "output_layer.weight",
)
_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_fingerprint(model_dir: Path) -> str:
    names = ("tokenizer_config.json", "tokenizer.json", "tokenizer.model", "special_tokens_map.json")
    digest = hashlib.sha256()
    found = False
    for name in names:
        path = model_dir / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    if not found:
        raise FileNotFoundError(f"no tokenizer artifacts found under {model_dir}")
    return digest.hexdigest()


def _weight_map(model_dir: Path) -> dict[str, str]:
    indexes = sorted(model_dir.glob("*.safetensors.index.json"))
    if indexes:
        value = json.loads(indexes[0].read_text(encoding="utf-8"))
        mapping = value.get("weight_map")
        if not isinstance(mapping, dict):
            raise ValueError(f"invalid safetensors weight_map in {indexes[0]}")
        return {str(key): str(filename) for key, filename in mapping.items()}

    mapping: dict[str, str] = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in mapping:
                    raise ValueError(f"duplicate tensor key {key!r} in safetensors files")
                mapping[key] = path.name
    if not mapping:
        raise FileNotFoundError(f"no safetensors weights found under {model_dir}")
    return mapping


def _resolve_key(mapping: Mapping[str, str], candidates: tuple[str, ...], explicit: str | None, kind: str) -> str:
    if explicit is not None:
        if explicit not in mapping:
            raise KeyError(f"requested {kind} key {explicit!r} is absent")
        return explicit
    for key in candidates:
        if key in mapping:
            return key
    suffix = ".embed_tokens.weight" if kind == "embed_tokens" else ".lm_head.weight"
    matches = [key for key in mapping if key.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"could not resolve a unique {kind} tensor; candidates={candidates}, matches={matches}")


def _load_one(model_dir: Path, mapping: Mapping[str, str], key: str) -> torch.Tensor:
    path = model_dir / mapping[key]
    if not path.is_file():
        raise FileNotFoundError(f"weight shard for {key!r} does not exist: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        value = handle.get_tensor(key)
    return value.detach().cpu().contiguous()


def extract_target_io(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    embed_key: str | None = None,
    lm_head_key: str | None = None,
) -> dict[str, Any]:
    """Copy only dense token embedding and LM-head tensors from a GLM checkpoint."""

    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden_size = int(config["hidden_size"])
    vocab_size = int(config["vocab_size"])
    mapping = _weight_map(model_dir)
    resolved_embed = _resolve_key(mapping, _EMBED_KEYS, embed_key, "embed_tokens")
    try:
        resolved_head = _resolve_key(mapping, _HEAD_KEYS, lm_head_key, "lm_head")
    except KeyError:
        if not bool(config.get("tie_word_embeddings", False)):
            raise
        resolved_head = resolved_embed

    embed = _load_one(model_dir, mapping, resolved_embed)
    head = embed if resolved_head == resolved_embed else _load_one(model_dir, mapping, resolved_head)
    expected = (vocab_size, hidden_size)
    if tuple(embed.shape) != expected or tuple(head.shape) != expected:
        raise ValueError(
            f"target I/O shape must match config {expected}; embed={tuple(embed.shape)}, lm_head={tuple(head.shape)}"
        )
    if embed.dtype not in _FLOAT_DTYPES or head.dtype not in _FLOAT_DTYPES:
        raise ValueError(
            f"target I/O must be dense floating-point tensors, got embed={embed.dtype}, lm_head={head.dtype}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = output_dir / "model.safetensors"
    # Clone the head when tied: safetensors intentionally rejects aliasing.
    save_file(
        {
            "embed_tokens.weight": embed.contiguous(),
            "lm_head.weight": head.clone().contiguous() if head.data_ptr() == embed.data_ptr() else head.contiguous(),
        },
        weight_path,
    )
    manifest: dict[str, Any] = {
        "schema": "glm-dflash2-target-io-v1",
        "source_model_dir": str(model_dir),
        "source_model_fingerprint": local_model_fingerprint(model_dir),
        "config_sha256": _sha256(config_path),
        "tokenizer_fingerprint": _tokenizer_fingerprint(model_dir),
        "model_type": str(config.get("model_type", "")),
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "tie_word_embeddings": bool(config.get("tie_word_embeddings", False)),
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
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing target I/O manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "glm-dflash2-target-io-v1":
        raise ValueError("unsupported target I/O manifest schema")
    weight_path = output_dir / str(manifest["weights_file"])
    if _sha256(weight_path) != manifest.get("weights_sha256"):
        raise ValueError("target I/O weights checksum mismatch")
    tensors = load_file(weight_path, device="cpu")
    embed_weight = tensors["embed_tokens.weight"].to(device=device, dtype=dtype)
    head_weight = tensors["lm_head.weight"].to(device=device, dtype=dtype)
    embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
    # Construct on meta and adopt the loaded tensor directly.  A normal Linear
    # constructor followed by copy would transiently allocate a second full
    # 154880x6144 head on every NPU rank.
    head = nn.Linear(
        int(manifest["hidden_size"]),
        int(manifest["vocab_size"]),
        bias=False,
        device="meta",
    )
    head.weight = nn.Parameter(head_weight, requires_grad=False)
    embed.eval()
    head.eval()
    return FrozenTargetIO(embed_tokens=embed, lm_head=head, manifest=manifest)


def validate_cache_io_compatibility(
    cache_manifest: Mapping[str, Any],
    io_manifest: Mapping[str, Any],
    *,
    expected_layer_ids: tuple[int, ...] = (1, 20, 38, 56, 75),
) -> None:
    spec = cache_manifest.get("spec") or {}
    provenance = cache_manifest.get("provenance") or {}
    if provenance.get("model_fingerprint") != io_manifest.get("source_model_fingerprint"):
        raise ValueError("cache and target I/O model fingerprint differ")
    logical = tuple(int(value) for value in spec.get("layer_ids", ()))
    recorded_logical = tuple(int(value) for value in provenance.get("logical_layer_ids", ()))
    if logical != expected_layer_ids or recorded_logical != expected_layer_ids:
        raise ValueError(
            f"cache logical layer order must be {expected_layer_ids}, got spec={logical}, provenance={recorded_logical}"
        )
    if int(spec.get("hidden_size", -1)) != int(io_manifest.get("hidden_size", -2)):
        raise ValueError("cache hidden size differs from target I/O hidden size")
    if spec.get("dtype") != "bfloat16" or spec.get("mask_semantics") != "dflash_target_token":
        raise ValueError("cache dtype or mask semantics are incompatible with DFlash2 training")
