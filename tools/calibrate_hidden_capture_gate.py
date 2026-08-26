#!/usr/bin/env python3
"""Calibrate immutable numerical bounds for the real hidden-capture gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


CAPTURE_KEYS = ("aux_hidden_states", "target_final_hidden", "target_logits")
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


def _capture_streams(capture: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    missing = [key for key in CAPTURE_KEYS if key not in capture]
    if missing:
        raise ValueError(f"capture is missing {', '.join(missing)}")
    aux = torch.as_tensor(capture["aux_hidden_states"]).detach().cpu().float()
    if aux.ndim < 2:
        raise ValueError("aux_hidden_states must expose a layer axis")
    values = {
        f"aux_hidden_states.layer_{index}": aux.select(-2, index)
        for index in range(aux.shape[-2])
    }
    values["target_final_hidden"] = (
        torch.as_tensor(capture["target_final_hidden"]).detach().cpu().float()
    )
    values["target_logits"] = (
        torch.as_tensor(capture["target_logits"]).detach().cpu().float()
    )
    for name, value in values.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"capture {name} contains NaN or Inf")
    return values


def _tensor_error_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError("capture stream shape differs")
    left = left.reshape(-1)
    right = right.reshape(-1)
    if not bool(left.norm() > 0) or not bool(right.norm() > 0):
        raise ValueError("cosine parity requires non-zero captures")
    absolute = (left - right).abs()
    cosine_error = 1.0 - float(F.cosine_similarity(left[None], right[None]).item())
    return {
        "cosine_error": max(0.0, cosine_error),
        "max_abs_error": float(absolute.max().item()),
        "mean_abs_error": float(absolute.mean().item()),
    }


def capture_stream_error_metrics(
    reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    left = _capture_streams(reference)
    right = _capture_streams(candidate)
    if tuple(left) != tuple(right):
        raise ValueError("capture stream sets differ")
    return {
        name: _tensor_error_metrics(left[name], right[name]) for name in left
    }


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
    return _tensor_error_metrics(_capture_vector(reference), _capture_vector(candidate))


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
                capture_stream_error_metrics(
                    direct_runs[left_index], direct_runs[right_index]
                )
            )
    controls = {
        name: capture_stream_error_metrics(direct_runs[0], capture)
        for name, capture in negative_controls.items()
    }
    streams: dict[str, Any] = {}
    for stream_name in direct_metrics[0]:
        metrics: dict[str, Any] = {}
        for metric_name in METRIC_KEYS:
            worst = max(value[stream_name][metric_name] for value in direct_metrics)
            floor = float(floors[metric_name])
            bound = max(floor, 2.0 * worst)
            negative = {
                key: value[stream_name][metric_name] for key, value in controls.items()
            }
            if any(bound >= error for error in negative.values()):
                raise ValueError(
                    f"{stream_name} {metric_name} bound is not strictly below "
                    "every negative control error"
                )
            metrics[metric_name] = {
                "floor": floor,
                "worst_direct_variation": worst,
                "bound": bound,
                "negative_controls": negative,
            }
        streams[stream_name] = {"metrics": metrics}
    return {
        "schema": "glm-hidden-capture-parity-gate-v2",
        "calibration_runs": 3,
        "identity": _validated_identity(identity),
        "capture_keys": list(CAPTURE_KEYS),
        "streams": streams,
        "target_top1_required": True,
    }


def validate_capture_with_gate(
    *,
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    artifact: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if artifact.get("schema") != "glm-hidden-capture-parity-gate-v2":
        raise ValueError("unsupported parity gate artifact")
    expected_identity = _validated_identity(identity)
    if artifact.get("identity") != expected_identity:
        raise ValueError("parity gate identity differs from the active runtime identity")
    actual = capture_stream_error_metrics(reference, candidate)
    stream_results = {}
    passed = True
    artifact_streams = artifact.get("streams")
    if not isinstance(artifact_streams, Mapping) or tuple(artifact_streams) != tuple(actual):
        raise ValueError("parity gate stream set differs from capture")
    for stream_name, actual_metrics in actual.items():
        metric_results = {}
        stream_passed = True
        configured = artifact_streams[stream_name].get("metrics", {})
        for metric_name in METRIC_KEYS:
            metric = configured.get(metric_name)
            if not isinstance(metric, Mapping) or "bound" not in metric:
                raise ValueError(
                    f"parity gate is missing metric {stream_name}.{metric_name}"
                )
            bound = float(metric["bound"])
            ok = actual_metrics[metric_name] <= bound
            metric_results[metric_name] = {
                "value": actual_metrics[metric_name],
                "bound": bound,
                "passed": ok,
            }
            stream_passed &= ok
        stream_results[stream_name] = {
            "passed": stream_passed,
            "metrics": metric_results,
        }
        passed &= stream_passed
    reference_top1 = torch.as_tensor(reference["target_logits"]).argmax(dim=-1)
    candidate_top1 = torch.as_tensor(candidate["target_logits"]).argmax(dim=-1)
    top1_passed = bool(torch.equal(reference_top1, candidate_top1))
    passed &= top1_passed
    return {
        "passed": passed,
        "streams": stream_results,
        "target_top1": {"passed": top1_passed},
        "identity": expected_identity,
    }


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
