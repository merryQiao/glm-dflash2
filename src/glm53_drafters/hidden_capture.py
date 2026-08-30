from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.metadata
import math
import re
from typing import Any

import torch


FINAL_HIDDEN_SEMANTICS = "post_final_norm_lm_head_input"
CAPTURE_ATTESTATION_SCHEMA = "glm53-capture-parity-v1"
ASCEND_A2_ATTESTATION_SCHEMA = "glm53-ascend-910b-a2-runtime-v1"


def validate_ascend_a2_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("cache has no structured Ascend 910B A2 runtime evidence")
    if value.get("schema") != ASCEND_A2_ATTESTATION_SCHEMA or value.get("passed") is not True:
        raise ValueError("cache lacks a passing Ascend 910B A2 runtime attestation")
    device_name = str(value.get("device_name") or "").replace(" ", "")
    if re.fullmatch(r"Ascend910B[1-4]?", device_name, flags=re.IGNORECASE) is None:
        raise ValueError(f"runtime device is not Ascend 910B A2: {device_name!r}")
    if not isinstance(value.get("device_index"), int) or int(value["device_index"]) < 0:
        raise ValueError("Ascend 910B A2 runtime evidence has invalid device index")
    if not isinstance(value.get("device_count"), int) or int(value["device_count"]) < 1:
        raise ValueError("Ascend 910B A2 runtime evidence has invalid device count")
    required = (
        "runner_module",
        "runner_class",
        "attention_backend_module",
        "attention_backend_class",
        "sglang_version",
        "torch_npu_version",
        "cann_version",
    )
    missing = [key for key in required if not str(value.get(key) or "").strip()]
    if missing:
        raise ValueError(
            "Ascend 910B A2 runtime evidence is incomplete: " + ", ".join(missing)
        )
    invalid_versions = []
    for key in ("sglang_version", "torch_npu_version", "cann_version"):
        version = str(value[key]).strip()
        normalized = version.lower().replace("_", "-")
        if (
            not re.search(r"\d", version)
            or normalized in {"unknown", "not-installed", "none", "n/a"}
            or "not installed" in normalized
        ):
            invalid_versions.append(key)
    if invalid_versions:
        raise ValueError(
            "Ascend 910B A2 runtime evidence has invalid version fields: "
            + ", ".join(invalid_versions)
        )
    _canonical_cann_version(str(value["cann_version"]))
    return dict(value)


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"required runtime distribution is not installed: {distribution}"
        ) from exc


def _canonical_cann_version(value: str) -> str:
    version = str(value).strip()
    pattern = r"[0-9]+(?:\.[0-9]+)+(?:[._-](?:RC|T|alpha|beta)?[0-9A-Za-z]+)*"
    if re.fullmatch(pattern, version, flags=re.IGNORECASE) is None:
        raise ValueError(f"CANN runtime returned a non-canonical version: {version!r}")
    return version


def _live_cann_version() -> str:
    """Read CANN through torch-npu's C-extension-backed runtime API only."""

    torch_npu = importlib.import_module("torch_npu")
    npu = getattr(torch_npu, "npu", None)
    getter = getattr(npu, "get_cann_version", None)
    if not callable(getter):
        raise RuntimeError(
            "torch_npu.npu.get_cann_version runtime API is unavailable"
        )
    return _canonical_cann_version(str(getter("CANN")))


def _require_imported_class(module_name: str, class_name: str) -> type[Any]:
    module = importlib.import_module(module_name)
    expected = getattr(module, class_name, None)
    if not isinstance(expected, type):
        raise RuntimeError(
            f"runtime class is not exported by its imported module: "
            f"{module_name}.{class_name}"
        )
    return expected


def validate_live_ascend_a2_evidence(value: Any) -> dict[str, Any]:
    """Re-probe the current process before a production cache may freeze."""

    evidence = validate_ascend_a2_evidence(value)
    importlib.import_module("torch_npu")
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "get_device_name"):
        raise RuntimeError("torch_npu did not register torch.npu device APIs")
    index = int(evidence["device_index"])
    actual_name = str(npu.get_device_name(index))
    if actual_name.replace(" ", "").lower() != str(
        evidence["device_name"]
    ).replace(" ", "").lower():
        raise RuntimeError("live Ascend device identity differs from runtime evidence")
    if int(npu.device_count()) != int(evidence["device_count"]):
        raise RuntimeError("live Ascend device count differs from runtime evidence")
    _require_imported_class(
        str(evidence["runner_module"]), str(evidence["runner_class"])
    )
    _require_imported_class(
        str(evidence["attention_backend_module"]),
        str(evidence["attention_backend_class"]),
    )
    live_versions = {
        "sglang_version": _installed_version("sglang"),
        "torch_npu_version": _installed_version("torch-npu"),
        "cann_version": _live_cann_version(),
    }
    for key, actual in live_versions.items():
        if str(evidence[key]) != actual:
            raise RuntimeError(f"live {key} differs from runtime evidence")
    return evidence


@dataclass(frozen=True)
class CaptureAttestation:
    passed: bool
    token_count: int
    logical_layer_ids: tuple[int, ...]
    physical_layer_ids: tuple[int, ...]
    independent_tap_paths: tuple[str, ...]
    native_logits_path: str
    aux_max_abs_error: float
    aux_max_rel_error: float
    logits_max_abs_error: float
    logits_max_rel_error: float
    reason: str
    aux_atol: float = 0.02
    aux_rtol: float = 0.02
    logits_atol: float = 0.02
    logits_rtol: float = 0.02
    schema: str = CAPTURE_ATTESTATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CAPTURE_ATTESTATION_SCHEMA:
            raise ValueError("unsupported capture attestation schema")
        if self.token_count < 1:
            raise ValueError("capture attestation token_count must be positive")
        if not self.logical_layer_ids or len(self.logical_layer_ids) != len(
            self.physical_layer_ids
        ):
            raise ValueError("capture attestation layer mapping is incomplete")
        if self.physical_layer_ids != tuple(
            logical + 1 for logical in self.logical_layer_ids
        ):
            raise ValueError("capture attestation physical tap mapping is invalid")
        if len(self.independent_tap_paths) != len(self.logical_layer_ids) or any(
            not path for path in self.independent_tap_paths
        ):
            raise ValueError("capture attestation independent tap paths are incomplete")
        if not self.native_logits_path:
            raise ValueError("capture attestation has no native logits path")
        metrics = (
            self.aux_max_abs_error,
            self.aux_max_rel_error,
            self.logits_max_abs_error,
            self.logits_max_rel_error,
            self.aux_atol,
            self.aux_rtol,
            self.logits_atol,
            self.logits_rtol,
        )
        if any(not math.isfinite(value) or value < 0 for value in metrics):
            raise ValueError("capture attestation metrics must be finite and non-negative")
        if self.passed and self.reason:
            raise ValueError("passing capture attestation cannot contain a failure reason")
        if self.passed and (
            self.aux_max_abs_error > self.aux_atol
            or self.aux_max_rel_error > self.aux_rtol
            or self.logits_max_abs_error > self.logits_atol
            or self.logits_max_rel_error > self.logits_rtol
        ):
            raise ValueError("passing capture attestation exceeds its numeric tolerance")
        if not self.passed and not self.reason:
            raise ValueError("failed capture attestation requires an actionable reason")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["logical_layer_ids"] = list(self.logical_layer_ids)
        value["physical_layer_ids"] = list(self.physical_layer_ids)
        value["independent_tap_paths"] = list(self.independent_tap_paths)
        return value


@dataclass(frozen=True)
class CaptureTap:
    backend_namespace: str
    logical_layer_id: int
    concrete_tap: str
    tap_semantics: str

    def __post_init__(self) -> None:
        if not self.backend_namespace or not self.concrete_tap or not self.tap_semantics:
            raise ValueError("capture tap metadata cannot be empty")
        if self.logical_layer_id < 0:
            raise ValueError("logical layer ID must be non-negative")

    def as_tuple(self) -> tuple[str, int, str, str]:
        return (
            self.backend_namespace,
            self.logical_layer_id,
            self.concrete_tap,
            self.tap_semantics,
        )


@dataclass(frozen=True)
class TargetHiddenCapture:
    aux_hidden_states: torch.Tensor
    target_final_hidden: torch.Tensor
    capture_mapping: tuple[CaptureTap, ...]
    final_hidden_semantics: str = FINAL_HIDDEN_SEMANTICS
    attestation: CaptureAttestation | None = None

    def __post_init__(self) -> None:
        if self.aux_hidden_states.ndim != 3:
            raise ValueError("aux_hidden_states must have shape [tokens, layers, hidden]")
        if self.target_final_hidden.ndim != 2:
            raise ValueError("target_final_hidden must have shape [tokens, hidden]")
        if self.aux_hidden_states.shape[0] != self.target_final_hidden.shape[0]:
            raise ValueError("auxiliary and final token dimensions differ")
        if self.aux_hidden_states.shape[2] != self.target_final_hidden.shape[1]:
            raise ValueError("auxiliary and final hidden widths differ")
        if len(self.capture_mapping) != self.aux_hidden_states.shape[1]:
            raise ValueError("capture mapping does not match auxiliary layer dimension")
        logical = tuple(tap.logical_layer_id for tap in self.capture_mapping)
        if logical != tuple(sorted(logical)) or len(logical) != len(set(logical)):
            raise ValueError("capture mapping logical layers must be unique and ordered")
        if self.final_hidden_semantics != FINAL_HIDDEN_SEMANTICS:
            raise ValueError("target final hidden is not the post-final-norm LM-head input")
        if not bool(torch.isfinite(self.aux_hidden_states).all()):
            raise ValueError("auxiliary hidden states contain NaN or Inf")
        if not bool(torch.isfinite(self.target_final_hidden).all()):
            raise ValueError("target final hidden contains NaN or Inf")
        if self.attestation is not None:
            if self.attestation.token_count != self.aux_hidden_states.shape[0]:
                raise ValueError("capture attestation token count differs from tensors")
            if self.attestation.logical_layer_ids != logical:
                raise ValueError("capture attestation logical tap order differs")

    @property
    def logical_layer_ids(self) -> tuple[int, ...]:
        return tuple(tap.logical_layer_id for tap in self.capture_mapping)

    def cpu_bfloat16(self) -> "TargetHiddenCapture":
        return TargetHiddenCapture(
            aux_hidden_states=self.aux_hidden_states.detach().to(
                device="cpu", dtype=torch.bfloat16
            ),
            target_final_hidden=self.target_final_hidden.detach().to(
                device="cpu", dtype=torch.bfloat16
            ),
            capture_mapping=self.capture_mapping,
            final_hidden_semantics=self.final_hidden_semantics,
            attestation=self.attestation,
        )
