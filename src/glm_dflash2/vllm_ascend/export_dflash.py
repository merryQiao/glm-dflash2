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
        "from speculators.models.dflash.config import DFlashSpeculatorConfig\n"
        "__all__ = ['DFlashSpeculatorConfig']\n"
    )


def build_dflash_export_config(config: Qwen3Config, target_io: FrozenTargetIO) -> dict[str, Any]:
    validate_export_inputs(config, target_io, allowed_block_sizes=(8, 16))
    result = base_export_config(method="dflash", config=config, target_io=target_io)
    result.update({
        "architectures": ["DFlashDraftModel"],
        "auto_map": {"": "config.DFlashSpeculatorConfig"},
        "config_class": "DFlashSpeculatorConfig",
    })
    return result


def export_dflash(
    output_dir: str | Path,
    *,
    config: Qwen3Config,
    state_dict: Mapping[str, torch.Tensor],
    target_io: FrozenTargetIO,
) -> dict[str, Any]:
    exported = build_dflash_export_config(config, target_io)
    state = cpu_state(state_dict)
    forbidden = ("candidate_selector.", "markov_head.", "confidence_head.")
    unexpected = [key for key in state if key.startswith(forbidden)]
    if unexpected:
        raise ValueError(f"DFlash export contains method-specific tensors: {unexpected[:3]}")
    block_size = int(exported["block_size"])
    return write_candidate_export(
        output_dir,
        method="dflash",
        config=exported,
        weights=state,
        target_io=target_io,
        config_source=_config_source(),
        method_metadata={
            "aux_hidden_state_layer_ids": exported["aux_hidden_state_layer_ids"],
            "block_size": block_size,
            "num_speculative_tokens": block_size - 1,
            "sample_from_anchor": False,
            "method_parameters": {},
        },
    )
