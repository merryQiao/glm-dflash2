"""Ascend runtime identity helpers with no eager NPU imports."""

from __future__ import annotations

from importlib import metadata
import os
from typing import Mapping


PACKAGES = {
    "vllm": "vllm",
    "vllm_ascend": "vllm-ascend",
    "torch": "torch",
    "torch_npu": "torch-npu",
    "transformers": "transformers",
    "qwen_omni_utils": "qwen-omni-utils",
}


def installed_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for label, distribution in PACKAGES.items():
        try:
            result[label] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result[label] = "not-installed"
    return result


def parse_visible_devices(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must contain integer chip IDs") from error
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must contain unique non-negative chip IDs")
    return devices


def runtime_identity(
    *,
    env: Mapping[str, str] | None = None,
    versions: Mapping[str, str] | None = None,
    hardware: str,
) -> dict:
    values = os.environ if env is None else env
    visible = values.get("ASCEND_RT_VISIBLE_DEVICES", "")
    return {
        "backend": "vllm_ascend",
        "hardware": hardware.lower(),
        "visible_devices": parse_visible_devices(visible),
        "hccl_op_expansion_mode": values.get("HCCL_OP_EXPANSION_MODE"),
        "hccl_buffsize": values.get("HCCL_BUFFSIZE"),
        "versions": dict(installed_versions() if versions is None else versions),
    }
