#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHOD_TENSORS = {
    "dflash": ("backbone_logits", "final_path"),
    "dflash2": (
        "backbone_logits",
        "candidate_ids",
        "candidate_scores",
        "pair_scores",
        "final_path",
    ),
    "dspark": (
        "backbone_logits",
        "markov_scores",
        "confidence_logits",
        "final_path",
    ),
}
IDENTITY_TENSORS = (
    "input_ids",
    "anchor_positions",
    "position_ids",
    "cache_fingerprint",
)
EXACT_TENSORS = {
    *IDENTITY_TENSORS,
    "candidate_ids",
    "final_path",
}


def required_tensors_for_method(method: str) -> tuple[str, ...]:
    try:
        return METHOD_TENSORS[method]
    except KeyError as exc:
        raise ValueError("method must be dflash, dflash2, or dspark") from exc


def compare_captures(
    trainer_capture: str | Path,
    runtime_capture: str | Path,
    *,
    method: str,
    rtol: float,
    atol: float,
) -> dict:
    left = np.load(trainer_capture)
    right = np.load(runtime_capture)
    results = {}
    passed = True
    required = IDENTITY_TENSORS + required_tensors_for_method(method)
    for name in required:
        if name not in left or name not in right:
            raise KeyError(f"both captures must contain {name}")
        if name in EXACT_TENSORS:
            equal = bool(np.array_equal(left[name], right[name]))
            error = 0.0 if equal else float("inf")
        else:
            equal = bool(
                np.allclose(left[name], right[name], rtol=float(rtol), atol=float(atol))
            )
            error = float(
                np.max(
                    np.abs(
                        left[name].astype(np.float32) - right[name].astype(np.float32)
                    )
                )
            )
        results[name] = {"passed": equal, "max_abs_error": error}
        passed &= equal
    return {
        "schema": "glm-unified-runtime-parity-v2",
        "method": method,
        "passed": passed,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare offline trainer tensors with an Ascend serving capture"
    )
    parser.add_argument("--method", choices=tuple(METHOD_TENSORS), required=True)
    parser.add_argument("--trainer-capture", required=True)
    parser.add_argument("--sglang-capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    args = parser.parse_args(argv)
    payload = compare_captures(
        args.trainer_capture,
        args.sglang_capture,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
    )
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
