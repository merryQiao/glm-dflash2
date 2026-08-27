"""Fail-closed validation for A2/A3 hardware smoke reports."""

from __future__ import annotations

from typing import Any, Mapping


REQUIRED_MODALITIES = ("text", "image", "audio", "video")


class ParityError(ValueError):
    """Raised when hardware execution has not established data parity."""


def validate_attestation(report: Mapping[str, Any]) -> None:
    if report.get("hardware") not in {"a2", "a3"}:
        raise ParityError("hardware must be a2 or a3")
    modalities = report.get("modalities")
    if not isinstance(modalities, Mapping):
        raise ParityError("modalities are missing")
    for name in REQUIRED_MODALITIES:
        result = modalities.get(name)
        if not isinstance(result, Mapping):
            raise ParityError(f"{name} attestation is missing")
        if result.get("exact_tokens") is not True:
            raise ParityError(f"{name} exact-token parity failed")
        if result.get("finite_hidden") is not True:
            raise ParityError(f"{name} hidden-state finiteness failed")
    if report.get("final_normalized_hidden") is not True:
        raise ParityError("final normalized hidden-state capability is missing")
