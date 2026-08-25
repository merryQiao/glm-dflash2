#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED = (
    "backbone_logits",
    "candidate_ids",
    "candidate_scores",
    "pair_scores",
    "final_path",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare captured offline-trainer tensors with an SGLang DFlash2 runtime capture"
    )
    parser.add_argument("--trainer-capture", required=True)
    parser.add_argument("--sglang-capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    args = parser.parse_args()
    left = np.load(args.trainer_capture)
    right = np.load(args.sglang_capture)
    results = {}
    passed = True
    for name in REQUIRED:
        if name not in left or name not in right:
            raise KeyError(f"both captures must contain {name}")
        if name in {"candidate_ids", "final_path"}:
            equal = bool(np.array_equal(left[name], right[name]))
            error = 0.0 if equal else float("inf")
        else:
            equal = bool(np.allclose(left[name], right[name], rtol=args.rtol, atol=args.atol))
            error = float(np.max(np.abs(left[name].astype(np.float32) - right[name].astype(np.float32))))
        results[name] = {"passed": equal, "max_abs_error": error}
        passed &= equal
    payload = {"schema": "glm-dflash2-runtime-parity-v1", "passed": passed, "results": results}
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
