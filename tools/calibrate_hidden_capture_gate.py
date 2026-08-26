#!/usr/bin/env python3
"""Calibrate immutable numerical bounds for the real hidden-capture gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


CAPTURE_KEYS = ("aux_hidden_states", "target_final_hidden")
METRIC_KEYS = ("cosine_error", "max_abs_error", "mean_abs_error")
IDENTITY_KEYS = (
    "target_fingerprint",
    "model_revision",
    "tokenizer_fingerprint",
    "cann_version",
    "torch_npu_version",
    "sglang_version",
)


def _capture_vector(capture: Mapping[str, torch.Tensor]) -> torch.Tensor:
    tensors = []
    for key in CAPTURE_KEYS:
        if key not in capture:
            raise ValueError(f"capture is missing {key}")
        value = torch.as_tensor(capture[key]).detach().cpu().float()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"capture {key} contains NaN or Inf")
        tensors.append(value.reshape(-1))
    return torch.cat(tensors)


def capture_error_metrics(
    reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    for key in CAPTURE_KEYS:
        if key not in reference or key not in candidate:
            raise ValueError(f"both captures must contain {key}")
        if tuple(torch.as_tensor(reference[key]).shape) != tuple(
            torch.as_tensor(candidate[key]).shape
        ):
            raise ValueError(f"capture shape differs for {key}")
    left = _capture_vector(reference)
    right = _capture_vector(candidate)
    if not bool(left.norm() > 0) or not bool(right.norm() > 0):
        raise ValueError("cosine parity requires non-zero captures")
    absolute = (left - right).abs()
    cosine_error = 1.0 - float(F.cosine_similarity(left[None], right[None]).item())
    return {
        "cosine_error": max(0.0, cosine_error),
        "max_abs_error": float(absolute.max().item()),
        "mean_abs_error": float(absolute.mean().item()),
    }


def _validated_identity(identity: Mapping[str, Any]) -> dict[str, str]:
    value = {key: str(identity.get(key, "")) for key in IDENTITY_KEYS}
    missing = [key for key, item in value.items() if not item]
    if missing:
        raise ValueError(f"capture identity is missing: {', '.join(missing)}")
    return value


def calibrate_parity_gate(
    *,
    direct_runs: Sequence[Mapping[str, torch.Tensor]],
    negative_controls: Mapping[str, Mapping[str, torch.Tensor]],
    floors: Mapping[str, float],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if len(direct_runs) != 3:
        raise ValueError("parity calibration requires exactly three direct runs")
    if set(negative_controls) != {"shifted_layer", "pre_norm"}:
        raise ValueError("negative controls must be shifted_layer and pre_norm")
    if set(floors) != set(METRIC_KEYS) or any(float(value) < 0 for value in floors.values()):
        raise ValueError("every parity metric needs a non-negative explicit floor")

    direct_metrics = []
    for left_index in range(len(direct_runs)):
        for right_index in range(left_index + 1, len(direct_runs)):
            direct_metrics.append(
                capture_error_metrics(direct_runs[left_index], direct_runs[right_index])
            )
    controls = {
        name: capture_error_metrics(direct_runs[0], capture)
        for name, capture in negative_controls.items()
    }
    metrics: dict[str, Any] = {}
    for name in METRIC_KEYS:
        worst = max(value[name] for value in direct_metrics)
        floor = float(floors[name])
        bound = max(floor, 2.0 * worst)
        negative = {key: value[name] for key, value in controls.items()}
        if any(bound >= error for error in negative.values()):
            raise ValueError(
                f"{name} bound is not strictly below every negative control error"
            )
        metrics[name] = {
            "floor": floor,
            "worst_direct_variation": worst,
            "bound": bound,
            "negative_controls": negative,
        }
    return {
        "schema": "glm-hidden-capture-parity-gate-v1",
        "calibration_runs": 3,
        "identity": _validated_identity(identity),
        "capture_keys": list(CAPTURE_KEYS),
        "metrics": metrics,
    }


def validate_capture_with_gate(
    *,
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    artifact: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if artifact.get("schema") != "glm-hidden-capture-parity-gate-v1":
        raise ValueError("unsupported parity gate artifact")
    expected_identity = _validated_identity(identity)
    if artifact.get("identity") != expected_identity:
        raise ValueError("parity gate identity differs from the active runtime identity")
    actual = capture_error_metrics(reference, candidate)
    results = {}
    passed = True
    for name in METRIC_KEYS:
        metric = artifact.get("metrics", {}).get(name)
        if not isinstance(metric, Mapping) or "bound" not in metric:
            raise ValueError(f"parity gate is missing metric {name}")
        bound = float(metric["bound"])
        ok = actual[name] <= bound
        results[name] = {"value": actual[name], "bound": bound, "passed": ok}
        passed &= ok
    return {"passed": passed, "results": results, "identity": expected_identity}


def _load_capture(path: str | Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"capture {path} is not a mapping")
    return {key: torch.as_tensor(value[key]) for key in CAPTURE_KEYS if key in value}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-run", action="append", required=True)
    parser.add_argument("--shifted-layer-control", required=True)
    parser.add_argument("--pre-norm-control", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cosine-floor", type=float, default=1e-7)
    parser.add_argument("--max-abs-floor", type=float, default=1e-3)
    parser.add_argument("--mean-abs-floor", type=float, default=1e-4)
    for key in IDENTITY_KEYS:
        parser.add_argument("--" + key.replace("_", "-"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = calibrate_parity_gate(
        direct_runs=[_load_capture(path) for path in args.direct_run],
        negative_controls={
            "shifted_layer": _load_capture(args.shifted_layer_control),
            "pre_norm": _load_capture(args.pre_norm_control),
        },
        floors={
            "cosine_error": args.cosine_floor,
            "max_abs_error": args.max_abs_floor,
            "mean_abs_error": args.mean_abs_floor,
        },
        identity={key: getattr(args, key) for key in IDENTITY_KEYS},
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
