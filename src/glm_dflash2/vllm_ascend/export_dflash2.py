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
        "@SpeculatorModelConfig.register('dflash2')\n"
        "class DFlash2SpeculatorConfig(DFlashSpeculatorConfig):\n"
        "    speculators_model_type: Literal['dflash2'] = 'dflash2'\n\n"
        "__all__ = ['DFlash2SpeculatorConfig']\n"
    )


def build_dflash2_export_config(config: Qwen3Config, target_io: FrozenTargetIO) -> dict[str, Any]:
    validate_export_inputs(config, target_io, allowed_block_sizes=(8, 16))
    dflash = dict(config.dflash_config)
    result = base_export_config(method="dflash2", config=config, target_io=target_io)
    result.update({
        "architectures": ["DFlash2DraftModel"],
        "auto_map": {"": "config.DFlash2SpeculatorConfig"},
        "config_class": "DFlash2SpeculatorConfig",
        "dflash2_config": {
            "conv_kernel_size": int(dflash["conv_kernel_size"]),
            "conv_group_size": int(dflash["conv_group_size"]),
            "selector_rank": int(dflash["selector_rank"]),
            "selector_top_k": int(dflash["selector_top_k"]),
        },
    })
    return result


def export_dflash2(
    output_dir: str | Path,
    *,
    config: Qwen3Config,
    state_dict: Mapping[str, torch.Tensor],
    target_io: FrozenTargetIO,
) -> dict[str, Any]:
    exported = build_dflash2_export_config(config, target_io)
    state = cpu_state(state_dict)
    required = (
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
        "candidate_selector.hidden_projection.weight",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"DFlash2 export is missing selector tensors: {missing}")
    if not any("attention_conv" in key for key in state) or not any("mlp_conv" in key for key in state):
        raise ValueError("DFlash2 export is missing dynamic-convolution tensors")
    if any(key.startswith(("markov_head.", "confidence_head.")) for key in state):
        raise ValueError("DFlash2 export contains DSpark tensors")
    block_size = int(exported["block_size"])
    return write_candidate_export(
        output_dir,
        method="dflash2",
        config=exported,
        weights=state,
        target_io=target_io,
        config_source=_config_source(),
        runtime_adapter="custom_class:dflash2",
        method_metadata={
            "aux_hidden_state_layer_ids": exported["aux_hidden_state_layer_ids"],
            "block_size": block_size,
            "num_speculative_tokens": block_size - 1,
            "sample_from_anchor": False,
            "method_parameters": dict(exported["dflash2_config"]),
        },
    )
