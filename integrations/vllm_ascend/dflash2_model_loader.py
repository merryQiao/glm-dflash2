from __future__ import annotations

from pathlib import Path

from glm_dflash2.vllm_ascend.export_common import load_candidate_export


def load_dflash2_candidate(path: str | Path):
    candidate = load_candidate_export(path)
    if candidate.method != "dflash2":
        raise ValueError(f"expected a DFlash2 candidate, got {candidate.method!r}")
    if candidate.manifest.get("runtime_adapter") != "custom_class:dflash2":
        raise ValueError("DFlash2 candidate does not declare the custom-class adapter")
    return candidate
