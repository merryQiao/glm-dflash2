from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from omni_sd.data_io import atomic_write_json

from .contracts import DEFAULT_MASK_TOKEN_ID, TARGET_CONTRACT


EMBED_KEY = "thinker.model.embed_tokens.weight"
HEAD_KEY = "thinker.lm_head.weight"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tokenizer_identity(model_path: Path, mask_token_id: int) -> str:
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError("official checkpoint lacks tokenizer.json")
    tokenizer = json.loads(tokenizer_path.read_text())
    used_ids: set[int] = set()
    vocabulary = tokenizer.get("model", {}).get("vocab", {})
    if isinstance(vocabulary, dict):
        used_ids.update(int(value) for value in vocabulary.values())
    for item in tokenizer.get("added_tokens", []):
        if isinstance(item, dict) and "id" in item:
            used_ids.add(int(item["id"]))
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        decoder = tokenizer_config.get("added_tokens_decoder", {})
        if isinstance(decoder, dict):
            used_ids.update(int(value) for value in decoder)
    if mask_token_id in used_ids:
        raise ValueError("reserved mask row is assigned by the official tokenizer")
    relevant = sorted(
        path for path in model_path.iterdir()
        if path.is_file() and path.name in {
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "processor_config.json", "preprocessor_config.json",
        }
    )
    return _canonical_hash({path.name: _sha256(path) for path in relevant})


def _thinker_text_config(root: dict[str, Any]) -> dict[str, Any]:
    thinker = root.get("thinker_config")
    if not isinstance(thinker, dict):
        raise ValueError("official checkpoint lacks thinker_config")
    text = thinker.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("official checkpoint lacks thinker_config.text_config")
    return text


def validate_official_model_config(root: dict[str, Any]) -> dict[str, Any]:
    if root.get("quantization_config") is not None:
        raise ValueError("Stage C target I/O must come from the BF16, non-quantized checkpoint")
    text = _thinker_text_config(root)
    expected = {
        "num_hidden_layers": TARGET_CONTRACT.num_layers,
        "hidden_size": TARGET_CONTRACT.hidden_size,
        "vocab_size": TARGET_CONTRACT.vocab_size,
        "num_attention_heads": TARGET_CONTRACT.num_attention_heads,
        "num_key_value_heads": TARGET_CONTRACT.num_key_value_heads,
        "head_dim": TARGET_CONTRACT.head_dim,
    }
    for key, value in expected.items():
        if int(text.get(key, -1)) != value:
            raise ValueError(f"official Thinker {key} mismatch: {text.get(key)!r} != {value}")
    if bool(text.get("tie_word_embeddings", root.get("tie_word_embeddings", False))):
        raise ValueError("Stage C requires the official untied Thinker LM head")
    if text.get("lm_head_bias", False) or text.get("logit_scale") not in (None, 1, 1.0):
        raise ValueError("biased or scaled Thinker LM heads are unsupported")
    if text.get("final_logit_softcapping") not in (None, 0, 0.0):
        raise ValueError("soft-capped Thinker logits are unsupported")
    return text


def _weight_map(model_path: Path) -> dict[str, Path]:
    index_files = sorted(model_path.glob("*.safetensors.index.json"))
    if index_files:
        index = json.loads(index_files[0].read_text())
        return {key: model_path / value for key, value in index["weight_map"].items()}
    shards = sorted(model_path.glob("*.safetensors"))
    result: dict[str, Path] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            result.update({key: shard for key in handle.keys()})
    return result


def _load_exact_weight(mapping: dict[str, Path], key: str) -> torch.Tensor:
    path = mapping.get(key)
    if path is None:
        raise KeyError(f"missing official weight {key}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(key)
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{key} must be BF16, got {tensor.dtype}")
    expected = (TARGET_CONTRACT.vocab_size, TARGET_CONTRACT.hidden_size)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{key} shape {tuple(tensor.shape)} != {expected}")
    return tensor.contiguous()


def _assert_mask_absent_from_cache(cache_root: Path, mask_token_id: int) -> None:
    manifest = json.loads((cache_root / "manifest.json").read_text())
    for item in manifest["files"]:
        path = cache_root / item["data"]["path"]
        with safe_open(path, framework="pt", device="cpu") as handle:
            if bool(handle.get_tensor("input_ids").eq(mask_token_id).any()):
                raise ValueError("reserved mask row occurs in Stage A/B trajectories")


def extract_target_io(model_path: str | Path, cache_root: str | Path,
                      output_dir: str | Path,
                      mask_token_id: int = DEFAULT_MASK_TOKEN_ID) -> Path:
    model_path, cache_root, output_dir = Path(model_path), Path(cache_root), Path(output_dir)
    config_path = model_path / "config.json"
    root_config = json.loads(config_path.read_text())
    validate_official_model_config(root_config)
    cache_manifest = json.loads((cache_root / "manifest.json").read_text())
    if cache_manifest.get("schema") != "omni-thinker-hidden-cache-v3":
        raise ValueError("target I/O can only bind a Stage B v3 cache")
    if not 0 <= mask_token_id < TARGET_CONTRACT.vocab_size:
        raise ValueError("mask token row is outside the padded vocabulary")
    tokenizer_fingerprint = _tokenizer_identity(model_path, mask_token_id)
    _assert_mask_absent_from_cache(cache_root, mask_token_id)

    mapping = _weight_map(model_path)
    embedding, head = _load_exact_weight(mapping, EMBED_KEY), _load_exact_weight(mapping, HEAD_KEY)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "target_io.safetensors"
    temporary = tensor_path.with_name(f".{tensor_path.name}.{os.getpid()}.tmp")
    save_file({"embedding.weight": embedding, "lm_head.weight": head}, str(temporary))
    os.replace(temporary, tensor_path)
    manifest = {
        "schema": "omni-thinker-target-io-v1",
        "status": "PASS",
        "dtype": "bfloat16",
        "embedding_shape": list(embedding.shape),
        "lm_head_shape": list(head.shape),
        "lm_head_bias": False,
        "mask_token_id": int(mask_token_id),
        "model_config_sha256": _sha256(config_path),
        "model_contract_sha256": _canonical_hash(root_config),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "cache_fingerprint": cache_manifest["cache_fingerprint"],
        "trajectory_generation_fingerprint": cache_manifest["trajectory_generation_fingerprint"],
        "tensor_sha256": _sha256(tensor_path),
        "source_keys": {"embedding": EMBED_KEY, "lm_head": HEAD_KEY},
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return output_dir


@dataclass(frozen=True)
class FrozenTargetIO:
    embedding: nn.Embedding
    lm_head: nn.Linear
    manifest: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path, cache_fingerprint: str | None = None) -> "FrozenTargetIO":
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        path = root / "target_io.safetensors"
        if manifest.get("schema") != "omni-thinker-target-io-v1" or manifest.get("status") != "PASS":
            raise ValueError("invalid Thinker target-I/O artifact")
        if _sha256(path) != manifest.get("tensor_sha256"):
            raise ValueError("target-I/O checksum mismatch")
        if cache_fingerprint and manifest.get("cache_fingerprint") != cache_fingerprint:
            raise ValueError("target I/O and hidden cache belong to different runs")
        tensors = load_file(path, device="cpu")
        embedding = nn.Embedding.from_pretrained(tensors["embedding.weight"], freeze=True)
        head = nn.Linear(TARGET_CONTRACT.hidden_size, TARGET_CONTRACT.vocab_size, bias=False)
        head.weight = nn.Parameter(tensors["lm_head.weight"], requires_grad=False)
        embedding.eval(), head.eval()
        return cls(embedding, head, manifest)
