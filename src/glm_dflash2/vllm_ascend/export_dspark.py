from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from ..target_io import FrozenTargetIO
from .export_common import (
    base_export_config,
    cpu_state,
    validate_export_inputs,
    write_candidate_export,
)


def _config_source() -> str:
    return (
        "from typing import Literal\n"
        "from speculators import SpeculatorModelConfig\n"
        "from speculators.models.dflash.config import DFlashSpeculatorConfig\n\n"
        "@SpeculatorModelConfig.register('dspark')\n"
        "class DSparkSpeculatorConfig(DFlashSpeculatorConfig):\n"
        "    speculators_model_type: Literal['dspark'] = 'dspark'\n"
        "    markov_rank: int = 256\n"
        "    markov_head_type: str = 'vanilla'\n"
        "    enable_confidence_head: bool = True\n"
        "    confidence_head_with_markov: bool = True\n"
        "    sample_from_anchor: bool = False\n\n"
        "__all__ = ['DSparkSpeculatorConfig']\n"
    )


def build_dspark_export_config(config: Qwen3Config, target_io: FrozenTargetIO) -> dict[str, Any]:
    validate_export_inputs(config, target_io, allowed_block_sizes=(8,))
    dspark = dict(getattr(config, "dspark_config", {}) or {})
    if str(dspark.get("markov_head_type", "vanilla")) != "vanilla":
        raise ValueError("DSpark runtime export requires vanilla Markov head")
    result = base_export_config(method="dspark", config=config, target_io=target_io)
    result.update({
        "architectures": ["DSparkDraftModel"],
        "auto_map": {"": "config.DSparkSpeculatorConfig"},
        "config_class": "DSparkSpeculatorConfig",
        "markov_rank": int(dspark.get("markov_rank", 256)),
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    })
    return result


def export_dspark(
    output_dir: str | Path,
    *,
    config: Qwen3Config,
    state_dict: Mapping[str, torch.Tensor],
    target_io: FrozenTargetIO,
) -> dict[str, Any]:
    exported = build_dspark_export_config(config, target_io)
    state = cpu_state(state_dict)
    if any(key.startswith("candidate_selector.") for key in state):
        raise ValueError("DSpark export contains DFlash2 selector tensors")
    for suffix in ("weight", "bias"):
        source = f"confidence_head.{suffix}"
        destination = f"confidence_head.proj.{suffix}"
        if source not in state:
            raise ValueError(f"DSpark export is missing {source}")
        state[destination] = state.pop(source)
    required = (
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"DSpark export is missing runtime tensors: {missing}")
    block_size = int(exported["block_size"])
    return write_candidate_export(
        output_dir,
        method="dspark",
        config=exported,
        weights=state,
        target_io=target_io,
        config_source=_config_source(),
        method_metadata={
            "aux_hidden_state_layer_ids": exported["aux_hidden_state_layer_ids"],
            "block_size": block_size,
            "num_speculative_tokens": block_size - 1,
            "sample_from_anchor": False,
            "method_parameters": {
                "markov_rank": int(exported["markov_rank"]),
                "markov_head_type": "vanilla",
                "confidence_head": True,
            },
        },
    )
