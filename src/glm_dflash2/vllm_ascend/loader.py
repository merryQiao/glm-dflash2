from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from ..dflash2_model import Qwen3DFlash2DraftModel, build_dflash2_config
from ..draft_backbone import DFlashDraftModel
from ..dspark_model import DSparkDraftModel
from .export_common import CandidateExport, load_candidate_export, sha256_file


LEGACY_EXPORT_SCHEMA = "glm-drafter-speculator-export-v1"


@dataclass(frozen=True)
class LoadedSpeculator:
    method: str
    model: torch.nn.Module
    config: Qwen3Config
    embed_tokens_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    manifest: Mapping[str, Any]


def _training_model(method: str, exported: Mapping[str, Any]) -> tuple[Qwen3Config, torch.nn.Module]:
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


def _restore(
    *, method: str, exported: Mapping[str, Any], tensors: Mapping[str, torch.Tensor], manifest: Mapping[str, Any]
) -> LoadedSpeculator:
    weights = dict(tensors)
    try:
        embed = weights.pop("embed_tokens.weight")
        head = weights.pop("lm_head.weight")
    except KeyError as exc:
        raise ValueError(f"export is missing frozen target tensor {exc.args[0]}") from exc
    if method == "dspark":
        for suffix in ("weight", "bias"):
            public_key = f"confidence_head.proj.{suffix}"
            if public_key in weights:
                weights[f"confidence_head.{suffix}"] = weights.pop(public_key)
    config, model = _training_model(method, exported)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"exported drafter state is not exact: missing={missing}, unexpected={unexpected}"
        )
    return LoadedSpeculator(method, model, config, embed, head, manifest)


def load_v2_export(path: str | Path) -> LoadedSpeculator:
    candidate: CandidateExport = load_candidate_export(path)
    return _restore(
        method=candidate.method,
        exported=candidate.config,
        tensors=candidate.weights,
        manifest=candidate.manifest,
    )


def load_legacy_v1_export(path: str | Path) -> LoadedSpeculator:
    path = Path(path)
    manifest = json.loads((path / "export_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != LEGACY_EXPORT_SCHEMA:
        raise ValueError("unsupported legacy speculator export schema")
    if sha256_file(path / "config.json") != manifest.get("config_sha256"):
        raise ValueError("legacy export config checksum mismatch")
    if sha256_file(path / "model.safetensors") != manifest.get("weights_sha256"):
        raise ValueError("legacy export weights checksum mismatch")
    exported = json.loads((path / "config.json").read_text(encoding="utf-8"))
    untrusted_manifest = {**manifest, "trust_status": "legacy-untrusted"}
    return _restore(
        method=str(manifest["method"]),
        exported=exported,
        tensors=load_file(path / "model.safetensors"),
        manifest=untrusted_manifest,
    )
