from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from ..target_io import FrozenTargetIO


EXPORT_SCHEMA = "glm-drafter-speculator-export-v2"
CANDIDATE_STATUS = "candidate-not-deployable"
ATTESTATION_FILENAME = "deploy_attestation.json"
EXPECTED_TARGET_LAYERS = (1, 20, 38, 56, 75)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def transformer_layer_config(config: Qwen3Config) -> dict[str, Any]:
    layer_types = list(getattr(config, "layer_types", []) or [])
    if not layer_types:
        layer_types = ["full_attention"] * int(config.num_hidden_layers)
    return {
        "attention_bias": bool(getattr(config, "attention_bias", False)),
        "attention_dropout": float(getattr(config, "attention_dropout", 0.0)),
        "head_dim": int(config.head_dim),
        "hidden_act": str(config.hidden_act),
        "hidden_size": int(config.hidden_size),
        "initializer_range": float(config.initializer_range),
        "intermediate_size": int(config.intermediate_size),
        "layer_types": layer_types,
        "max_position_embeddings": int(config.max_position_embeddings),
        "model_type": "qwen3",
        "num_attention_heads": int(config.num_attention_heads),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_key_value_heads": int(config.num_key_value_heads),
        "rms_norm_eps": float(config.rms_norm_eps),
        "rope_parameters": {
            "rope_theta": float(config.rope_theta),
            "rope_type": "default",
        },
        "sliding_window": getattr(config, "sliding_window", None),
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_sliding_window": bool(getattr(config, "use_sliding_window", False)),
        "vocab_size": int(config.vocab_size),
    }


def validate_export_inputs(
    config: Qwen3Config, target_io: FrozenTargetIO, *, allowed_block_sizes: tuple[int, ...]
) -> tuple[int, list[int]]:
    manifest = target_io.manifest
    if manifest.get("schema") != "glm-drafter-target-io-v2":
        raise ValueError("candidate export requires target I/O manifest schema v2")
    if int(manifest.get("vocab_size", -1)) != int(config.vocab_size):
        raise ValueError("target I/O vocabulary differs from draft configuration")
    if int(manifest.get("hidden_size", -1)) != int(config.hidden_size):
        raise ValueError("target I/O hidden size differs from draft configuration")
    expected_shape = (int(config.vocab_size), int(config.hidden_size))
    if tuple(target_io.embed_tokens.weight.shape) != expected_shape:
        raise ValueError("target embed_tokens shape differs from draft configuration")
    if tuple(target_io.lm_head.weight.shape) != expected_shape:
        raise ValueError("target lm_head shape differs from draft configuration")
    dflash = dict(config.dflash_config)
    block_size = int(dflash["block_size"])
    if block_size not in allowed_block_sizes:
        allowed = "/".join(f"B{value}" for value in allowed_block_sizes)
        raise ValueError(f"unsupported block size {block_size}; expected {allowed}")
    layer_ids = [int(value) for value in dflash["target_layer_ids"]]
    if tuple(layer_ids) != EXPECTED_TARGET_LAYERS:
        raise ValueError(
            f"target layer order must be {EXPECTED_TARGET_LAYERS}, got {tuple(layer_ids)}"
        )
    if int(config.num_hidden_layers) != 5:
        raise ValueError("runtime export requires the five-layer draft backbone")
    if any(value != "full_attention" for value in transformer_layer_config(config)["layer_types"]):
        raise ValueError("runtime export requires full attention in every draft layer")
    if bool(getattr(config, "use_sliding_window", False)) or getattr(config, "sliding_window", None) is not None:
        raise ValueError("runtime export does not support sliding-window attention")
    return block_size, layer_ids


def base_export_config(
    *, method: str, config: Qwen3Config, target_io: FrozenTargetIO
) -> dict[str, Any]:
    block_size = int(config.dflash_config["block_size"])
    layer_ids = [int(value) for value in config.dflash_config["target_layer_ids"]]
    return {
        "aux_hidden_state_layer_ids": layer_ids,
        "block_size": block_size,
        "draft_vocab_size": int(config.vocab_size),
        "dtype": "bfloat16",
        "mask_token_id": int(config.dflash_config["mask_token_id"]),
        "max_anchors": 64,
        "num_speculative_tokens": block_size - 1,
        "sample_from_anchor": False,
        "speculators_config": {
            "algorithm": method,
            "default_proposal_method": "greedy",
            "proposal_methods": [{
                "accept_tolerance": 0.0,
                "proposal_type": "greedy",
                "speculative_tokens": block_size - 1,
                "verifier_accept_k": 1,
            }],
            "verifier": {
                "architectures": [],
                "name_or_path": str(target_io.manifest.get("source_model_dir", "")),
            },
        },
        "speculators_model_type": method,
        "speculators_version": "0.5.0",
        "target_hidden_size": None,
        "tie_word_embeddings": False,
        "transformer_layer_config": transformer_layer_config(config),
    }


def cpu_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state {key!r} is not a tensor")
        if key in result:
            raise ValueError(f"duplicate state key {key!r}")
        result[str(key)] = value.detach().cpu().contiguous()
    return result


def _validate_metadata(method: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "aux_hidden_state_layer_ids",
        "block_size",
        "num_speculative_tokens",
        "sample_from_anchor",
        "method_parameters",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"candidate metadata is missing {missing}")
    result = json.loads(json.dumps(dict(metadata)))
    if tuple(int(value) for value in result["aux_hidden_state_layer_ids"]) != EXPECTED_TARGET_LAYERS:
        raise ValueError("candidate auxiliary hidden layer order is invalid")
    block_size = int(result["block_size"])
    if int(result["num_speculative_tokens"]) != block_size - 1:
        raise ValueError("candidate proposal count must equal block_size - 1")
    if bool(result["sample_from_anchor"]):
        raise ValueError("GLM draft candidates require sample_from_anchor=false")
    result["method"] = method
    return result


def _atomic_replace_directory(source: Path, destination: Path) -> None:
    backup: Path | None = None
    try:
        if destination.exists():
            backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
            os.replace(destination, backup)
        os.replace(source, destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup)


def write_candidate_export(
    output_dir: str | Path,
    *,
    method: str,
    config: Mapping[str, Any],
    weights: Mapping[str, torch.Tensor],
    target_io: FrozenTargetIO,
    method_metadata: Mapping[str, Any],
    config_source: str | None = None,
    runtime_adapter: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    metadata = _validate_metadata(method, method_metadata)
    state = cpu_state(weights)
    for key, value in (
        ("embed_tokens.weight", target_io.embed_tokens.weight),
        ("lm_head.weight", target_io.lm_head.weight),
    ):
        if key in state:
            raise ValueError(f"draft state unexpectedly contains frozen {key}")
        state[key] = value.detach().cpu().contiguous()
    expected_shape = (
        int(target_io.manifest.get("vocab_size", -1)),
        int(target_io.manifest.get("hidden_size", -1)),
    )
    if tuple(state["embed_tokens.weight"].shape) != expected_shape:
        raise ValueError("target embed_tokens shape is inconsistent with its manifest")
    if tuple(state["lm_head.weight"].shape) != expected_shape:
        raise ValueError("target lm_head shape is inconsistent with its manifest")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        config_path = temporary / "config.json"
        config_path.write_text(
            json.dumps(dict(config), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        weights_path = temporary / "model.safetensors"
        save_file(state, weights_path)
        files = {
            "config.json": sha256_file(config_path),
            "model.safetensors": sha256_file(weights_path),
        }
        if config_source is not None:
            source_path = temporary / "config.py"
            source_path.write_text(config_source, encoding="utf-8")
            files["config.py"] = sha256_file(source_path)
        manifest: dict[str, Any] = {
            "schema": EXPORT_SCHEMA,
            "status": CANDIDATE_STATUS,
            "method": method,
            "runtime_compatibility": "candidate-requires-runtime-attestation",
            "runtime_adapter": runtime_adapter or f"method:{method}",
            "files": files,
            "config_sha256": files["config.json"],
            "weights_sha256": files["model.safetensors"],
            "checkpoint_sha256": files["model.safetensors"],
            "target_io_schema": target_io.manifest.get("schema"),
            "target_io_sha256": target_io.manifest.get("weights_sha256", "unknown"),
            "target_model_fingerprint": target_io.manifest.get("source_model_fingerprint"),
            "target_model_revision": target_io.manifest.get("model_revision"),
            "tokenizer_fingerprint": target_io.manifest.get("tokenizer_fingerprint"),
            **metadata,
        }
        manifest_path = temporary / "export_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _atomic_replace_directory(temporary, output_dir)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


@dataclass(frozen=True)
class CandidateExport:
    method: str
    config: Mapping[str, Any]
    weights: dict[str, torch.Tensor]
    manifest: Mapping[str, Any]
    path: Path


def load_candidate_export(output_dir: str | Path) -> CandidateExport:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "export_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing candidate manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPORT_SCHEMA:
        raise ValueError("unsupported candidate export schema")
    if manifest.get("status") not in {CANDIDATE_STATUS, "runtime-attested"}:
        raise ValueError("unsupported candidate export status")
    files = manifest.get("files") or {}
    for filename, expected in files.items():
        path = output_dir / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"candidate {filename} checksum mismatch")
    if manifest.get("status") == CANDIDATE_STATUS and (output_dir / ATTESTATION_FILENAME).exists():
        raise ValueError("unattested candidate unexpectedly contains a deploy attestation")
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    weights = load_file(output_dir / "model.safetensors")
    return CandidateExport(str(manifest["method"]), config, weights, manifest, output_dir)
