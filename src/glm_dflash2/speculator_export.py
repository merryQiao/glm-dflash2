"""Compatibility facade for method-specific vLLM-Ascend exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .target_io import FrozenTargetIO
from .vllm_ascend.export_common import EXPORT_SCHEMA
from .vllm_ascend.export_dflash import export_dflash
from .vllm_ascend.export_dflash2 import export_dflash2
from .vllm_ascend.export_dspark import export_dspark
from .vllm_ascend.loader import (
    LEGACY_EXPORT_SCHEMA,
    LoadedSpeculator,
    load_legacy_v1_export,
    load_v2_export,
)


_EXPORTERS = {
    "dflash": export_dflash,
    "dflash2": export_dflash2,
    "dspark": export_dspark,
}


def export_speculator(
    output_dir: str | Path,
    *,
    method: str,
    config: Qwen3Config,
    model: torch.nn.Module,
    target_io: FrozenTargetIO,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> dict:
    try:
        exporter = _EXPORTERS[method]
    except KeyError as exc:
        raise ValueError(f"unknown drafter method {method!r}") from exc
    return exporter(
        output_dir,
        config=config,
        state_dict=model.state_dict() if state_dict is None else state_dict,
        target_io=target_io,
    )


def load_exported_speculator(output_dir: str | Path) -> LoadedSpeculator:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "export_manifest.json").read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema == EXPORT_SCHEMA:
        return load_v2_export(output_dir)
    if schema == LEGACY_EXPORT_SCHEMA:
        return load_legacy_v1_export(output_dir)
    raise ValueError(f"unsupported speculator export schema {schema!r}")


__all__ = ["LoadedSpeculator", "export_speculator", "load_exported_speculator"]
