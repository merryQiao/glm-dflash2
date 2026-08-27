from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .dflash2_model import Qwen3DFlash2DraftModel, build_dflash2_config
from .draft_backbone import DFlashDraftModel
from .dspark_model import DSparkDraftModel
from .target_io import FrozenTargetIO


_EXPORT_SCHEMA = "glm-drafter-speculator-export-v1"
_METHODS = ("dflash", "dflash2", "dspark")
_ARCHITECTURES = {
    "dflash": "DFlashDraftModel",
    "dflash2": "DFlash2DraftModel",
    "dspark": "DSparkDraftModel",
}
_CONFIG_CLASSES = {
    "dflash": "DFlashSpeculatorConfig",
    "dflash2": "DFlash2SpeculatorConfig",
    "dspark": "DSparkSpeculatorConfig",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _transformer_config(config: Qwen3Config) -> dict[str, Any]:
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


def build_export_config(
    *, method: str, config: Qwen3Config, target_io_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if method not in _METHODS:
        raise ValueError(f"unknown drafter method {method!r}")
    dflash = dict(config.dflash_config)
    block_size = int(dflash["block_size"])
    if block_size < 2:
        raise ValueError("physical block size must contain one anchor and proposals")
    layer_ids = [int(value) for value in dflash["target_layer_ids"]]
    if target_io_manifest.get("schema") != "glm-drafter-target-io-v2":
        raise ValueError("standard export requires target I/O manifest schema v2")
    if int(target_io_manifest.get("vocab_size", -1)) != int(config.vocab_size):
        raise ValueError("target I/O vocabulary differs from draft configuration")
    if int(target_io_manifest.get("hidden_size", -1)) != int(config.hidden_size):
        raise ValueError("target I/O hidden size differs from draft configuration")

    result: dict[str, Any] = {
        "architectures": [_ARCHITECTURES[method]],
        "auto_map": {"": f"config.{_CONFIG_CLASSES[method]}"},
        "aux_hidden_state_layer_ids": layer_ids,
        "block_size": block_size,
        "draft_vocab_size": int(config.vocab_size),
        "dtype": "bfloat16",
        "mask_token_id": int(dflash["mask_token_id"]),
        "max_anchors": 64,
        # Training and inference both treat position zero as a known anchor.
        "sample_from_anchor": False,
        "speculators_config": {
            "algorithm": method,
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "accept_tolerance": 0.0,
                    "proposal_type": "greedy",
                    "speculative_tokens": block_size - 1,
                    "verifier_accept_k": 1,
                }
            ],
            "verifier": {
                "architectures": [],
                "name_or_path": str(target_io_manifest.get("source_model_dir", "")),
            },
        },
        "speculators_model_type": method,
        "speculators_version": "0.5.0",
        "target_hidden_size": None,
        "tie_word_embeddings": False,
        "transformer_layer_config": _transformer_config(config),
    }
    if method == "dspark":
        dspark = dict(getattr(config, "dspark_config", {}) or {})
        result.update(
            {
                "markov_rank": int(dspark.get("markov_rank", 256)),
                "markov_head_type": str(dspark.get("markov_head_type", "vanilla")),
                "enable_confidence_head": True,
                "confidence_head_with_markov": True,
            }
        )
    elif method == "dflash2":
        result["dflash2_config"] = {
            "conv_kernel_size": int(dflash["conv_kernel_size"]),
            "conv_group_size": int(dflash["conv_group_size"]),
            "selector_rank": int(dflash["selector_rank"]),
            "selector_top_k": int(dflash["selector_top_k"]),
        }
    return result


def _export_state(
    *,
    method: str,
    state_dict: Mapping[str, torch.Tensor],
    target_io: FrozenTargetIO,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        export_key = key
        if method == "dspark" and key.startswith("confidence_head."):
            export_key = "confidence_head.proj." + key.removeprefix("confidence_head.")
        state[export_key] = value.detach().cpu().contiguous()
    for key, value in (
        ("embed_tokens.weight", target_io.embed_tokens.weight),
        ("lm_head.weight", target_io.lm_head.weight),
    ):
        if key in state:
            raise ValueError(f"draft state unexpectedly contains frozen {key}")
        state[key] = value.detach().cpu().contiguous()
    return state


def _config_source(method: str) -> str:
    # vLLM/speculators resolves the empty auto_map entry through this module.
    # DFlash2 deliberately registers a distinct method and is not claimed to be
    # supported by stock vLLM-Ascend; its runtime adapter must provide the model.
    class_name = _CONFIG_CLASSES[method]
    parent = "DFlashSpeculatorConfig"
    model_type = method
    if method == "dflash":
        return (
            "from speculators.models.dflash.config import DFlashSpeculatorConfig\n"
            "__all__ = ['DFlashSpeculatorConfig']\n"
        )
    fields = ""
    if method == "dspark":
        fields = (
            "    markov_rank: int = 256\n"
            "    markov_head_type: str = 'vanilla'\n"
            "    enable_confidence_head: bool = True\n"
            "    confidence_head_with_markov: bool = True\n"
            "    sample_from_anchor: bool = False\n"
        )
    return (
        "from typing import Literal\n"
        "from speculators import SpeculatorModelConfig\n"
        "from speculators.models.dflash.config import DFlashSpeculatorConfig\n\n"
        f"@SpeculatorModelConfig.register('{model_type}')\n"
        f"class {class_name}({parent}):\n"
        f"    speculators_model_type: Literal['{model_type}'] = '{model_type}'\n"
        f"{fields or '    pass\n'}"
        f"__all__ = ['{class_name}']\n"
    )


def export_speculator(
    output_dir: str | Path,
    *,
    method: str,
    config: Qwen3Config,
    model: torch.nn.Module,
    target_io: FrozenTargetIO,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_config = build_export_config(
        method=method, config=config, target_io_manifest=target_io.manifest
    )
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(exported_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.py").write_text(_config_source(method), encoding="utf-8")
    weights_path = output_dir / "model.safetensors"
    weights = _export_state(
        method=method,
        state_dict=model.state_dict() if state_dict is None else state_dict,
        target_io=target_io,
    )
    save_file(weights, weights_path)
    # The public speculators key layout is useful for export/round-trip, but it
    # does not prove that the selected Ascend runtime implements this GLM-5.2
    # draft architecture.  Fail closed until the deployment fork has passed
    # an offline-vs-runtime logits/acceptance parity gate for the method.
    runtime = "custom-glm52-vllm-ascend-adapter-required"
    manifest = {
        "schema": _EXPORT_SCHEMA,
        "method": method,
        "runtime_compatibility": runtime,
        "config_sha256": _sha256(config_path),
        "weights_sha256": _sha256(weights_path),
        "target_io_schema": target_io.manifest.get("schema"),
        "target_model_fingerprint": target_io.manifest.get("source_model_fingerprint"),
        "target_model_revision": target_io.manifest.get("model_revision"),
        "tokenizer_fingerprint": target_io.manifest.get("tokenizer_fingerprint"),
        "sample_from_anchor": False,
        "num_speculative_tokens": int(exported_config["block_size"]) - 1,
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


@dataclass(frozen=True)
class LoadedSpeculator:
    method: str
    model: torch.nn.Module
    config: Qwen3Config
    embed_tokens_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    manifest: Mapping[str, Any]


def _internal_model(method: str, exported: Mapping[str, Any]) -> tuple[Qwen3Config, torch.nn.Module]:
    transformer = dict(exported["transformer_layer_config"])
    rope = dict(transformer.get("rope_parameters") or {})
    dflash2 = dict(exported.get("dflash2_config") or {})
    config = build_dflash2_config(
        vocab_size=int(transformer["vocab_size"]),
        hidden_size=int(transformer["hidden_size"]),
        intermediate_size=int(transformer["intermediate_size"]),
        num_hidden_layers=int(transformer["num_hidden_layers"]),
        num_attention_heads=int(transformer["num_attention_heads"]),
        num_key_value_heads=int(transformer["num_key_value_heads"]),
        head_dim=int(transformer["head_dim"]),
        target_layer_ids=list(exported["aux_hidden_state_layer_ids"]),
        num_target_layers=max(int(value) for value in exported["aux_hidden_state_layer_ids"]) + 1,
        block_size=int(exported["block_size"]),
        mask_token_id=int(exported["mask_token_id"]),
        conv_group_size=int(dflash2.get("conv_group_size", 16)),
        selector_rank=int(dflash2.get("selector_rank", 256)),
        selector_top_k=int(dflash2.get("selector_top_k", 16)),
        sliding_window=transformer.get("sliding_window"),
        conv_kernel_size=int(dflash2.get("conv_kernel_size", 2)),
        rms_norm_eps=float(transformer["rms_norm_eps"]),
        rope_theta=float(rope.get("rope_theta", 8_000_000.0)),
        max_position_embeddings=int(transformer["max_position_embeddings"]),
    )
    config.drafter_method = method
    config.position_contract = "absolute_anchor_plus_local"
    config.target_layer_ids = list(config.dflash_config["target_layer_ids"])
    config.physical_block_size = int(config.dflash_config["block_size"])
    config.num_speculative_tokens = config.physical_block_size - 1
    if method == "dflash":
        model = DFlashDraftModel(config)
    elif method == "dflash2":
        model = Qwen3DFlash2DraftModel(config)
    elif method == "dspark":
        model = DSparkDraftModel(config, markov_rank=int(exported["markov_rank"]))
    else:
        raise ValueError(f"unknown exported method {method!r}")
    return config, model


def load_exported_speculator(output_dir: str | Path) -> LoadedSpeculator:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "export_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != _EXPORT_SCHEMA:
        raise ValueError("unsupported speculator export schema")
    if _sha256(output_dir / "config.json") != manifest.get("config_sha256"):
        raise ValueError("export config checksum mismatch")
    if _sha256(output_dir / "model.safetensors") != manifest.get("weights_sha256"):
        raise ValueError("export weights checksum mismatch")
    exported = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    method = str(manifest["method"])
    config, model = _internal_model(method, exported)
    weights = load_file(output_dir / "model.safetensors")
    embed = weights.pop("embed_tokens.weight")
    head = weights.pop("lm_head.weight")
    if method == "dspark":
        for suffix in ("weight", "bias"):
            public_key = f"confidence_head.proj.{suffix}"
            if public_key in weights:
                weights[f"confidence_head.{suffix}"] = weights.pop(public_key)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"exported drafter state is not exact: missing={missing}, unexpected={unexpected}"
        )
    return LoadedSpeculator(method, model, config, embed, head, manifest)
