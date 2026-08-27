from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


RUNTIME_IDENTITY_KEYS = (
    "vllm_version",
    "vllm_commit",
    "vllm_ascend_version",
    "vllm_ascend_commit",
    "speculators_version",
    "speculators_commit",
    "adapter_revision",
    "cann_version",
    "torch_npu_version",
    "driver_version",
    "firmware_version",
    "device_name",
    "attention_backend",
    "model_runner",
    "tp_size",
    "ep_size",
    "pp_size",
    "dp_size",
    "nnodes",
    "graph_mode",
    "chunked_prefill",
    "prefix_cache",
)
_INTEGER_KEYS = {"tp_size", "ep_size", "pp_size", "dp_size", "nnodes"}
_BOOLEAN_KEYS = {"chunked_prefill", "prefix_cache"}
_UNKNOWN = {"", "unknown", "not-installed", "none", "null", "pin_on_ascend_host"}


def normalize_runtime_identity(
    identity: Mapping[str, Any], *, production: bool
) -> dict[str, Any]:
    missing = [key for key in RUNTIME_IDENTITY_KEYS if key not in identity]
    if missing:
        raise ValueError(f"runtime identity is missing: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for key in RUNTIME_IDENTITY_KEYS:
        value = identity[key]
        if key in _INTEGER_KEYS:
            value = int(value)
            if value < 1:
                raise ValueError(f"runtime identity {key} must be positive")
        elif key in _BOOLEAN_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"runtime identity {key} must be boolean")
        else:
            value = str(value).strip()
            if production and value.lower() in _UNKNOWN:
                raise ValueError(f"production runtime identity has unknown {key}")
        result[key] = value
    if production:
        if result["pp_size"] != 1 or result["dp_size"] != 1:
            raise ValueError("attested runtime requires PP=1 and DP=1")
        if result["chunked_prefill"]:
            raise ValueError("attested runtime requires chunked prefill disabled")
        if result["prefix_cache"]:
            raise ValueError("attested runtime requires prefix cache disabled")
        if result["graph_mode"].lower() not in {"disabled", "none"}:
            raise ValueError("attested runtime requires execution graphs disabled")
    return result


def runtime_identities_match(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> tuple[bool, dict[str, dict[str, Any]]]:
    left = normalize_runtime_identity(expected, production=True)
    right = normalize_runtime_identity(actual, production=True)
    drift = {
        key: {"expected": left[key], "actual": right[key]}
        for key in RUNTIME_IDENTITY_KEYS
        if left[key] != right[key]
    }
    return not drift, drift


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not-installed"


def _git_revision_from_package(module_name: str) -> str:
    try:
        distribution = importlib.metadata.distribution(module_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
    root = Path(distribution.locate_file(""))
    for candidate in (root, *root.parents):
        if not (candidate / ".git").exists():
            continue
        try:
            return subprocess.check_output(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            break
    return "unknown"


def collect_runtime_identity(
    *,
    tp_size: int,
    ep_size: int,
    pp_size: int = 1,
    dp_size: int = 1,
    nnodes: int = 1,
    attention_backend: str = "ascend",
    model_runner: str = "v1",
    graph_mode: str = "disabled",
    chunked_prefill: bool = False,
    prefix_cache: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "vllm_version": _package_version("vllm"),
        "vllm_commit": os.environ.get("VLLM_COMMIT", _git_revision_from_package("vllm")),
        "vllm_ascend_version": _package_version("vllm-ascend", "vllm_ascend"),
        "vllm_ascend_commit": os.environ.get(
            "VLLM_ASCEND_COMMIT", _git_revision_from_package("vllm-ascend")
        ),
        "speculators_version": _package_version("speculators"),
        "speculators_commit": os.environ.get(
            "SPECULATORS_COMMIT", _git_revision_from_package("speculators")
        ),
        "adapter_revision": os.environ.get(
            "GLM_DFLASH2_ADAPTER_REVISION", "vllm-ascend-dflash2-adapter-v1"
        ),
        "cann_version": os.environ.get("ASCEND_TOOLKIT_VERSION", "unknown"),
        "torch_npu_version": _package_version("torch-npu", "torch_npu"),
        "driver_version": os.environ.get("ASCEND_DRIVER_VERSION", "unknown"),
        "firmware_version": os.environ.get("ASCEND_FIRMWARE_VERSION", "unknown"),
        "device_name": os.environ.get("ASCEND_DEVICE_NAME", "unknown"),
        "attention_backend": attention_backend,
        "model_runner": model_runner,
        "tp_size": int(tp_size),
        "ep_size": int(ep_size),
        "pp_size": int(pp_size),
        "dp_size": int(dp_size),
        "nnodes": int(nnodes),
        "graph_mode": graph_mode,
        "chunked_prefill": bool(chunked_prefill),
        "prefix_cache": bool(prefix_cache),
    }
    if overrides:
        value.update(dict(overrides))
    return normalize_runtime_identity(value, production=False)


def load_runtime_identity(path: str | Path, *, production: bool) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("runtime identity JSON must be an object")
    return normalize_runtime_identity(value, production=production)
