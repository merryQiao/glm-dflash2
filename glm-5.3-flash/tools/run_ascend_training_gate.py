#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable


EXPECTED_LOGICAL = [1, 11, 22, 32, 42]
EXPECTED_CONCRETE = [2, 12, 23, 33, 43]


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evidence must be a JSON object")
    return value


def _tap_mapping(value: dict[str, Any]) -> None:
    if value.get("passed") is not True:
        raise ValueError("tap mapping did not pass")
    if value.get("logical_layer_ids") != EXPECTED_LOGICAL:
        raise ValueError("logical tap mapping differs from the GLM-5.3 contract")
    if value.get("concrete_hidden_state_indices") != EXPECTED_CONCRETE:
        raise ValueError("concrete tap mapping differs from the GLM-5.3 contract")


def _final_logit(value: dict[str, Any]) -> None:
    if value.get("passed") is not True:
        raise ValueError("final-logit parity did not pass")
    error = float(value["max_abs_error"])
    tolerance = float(value["tolerance"])
    if not math.isfinite(error) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("final-logit parity values must be finite")
    if error > tolerance:
        raise ValueError("final-logit parity exceeds tolerance")


def _fsdp_resume(value: dict[str, Any]) -> None:
    if value.get("passed") is not True:
        raise ValueError("FSDP2 resume parity did not pass")
    if value.get("backend") != "hccl" or value.get("fsdp2") is not True:
        raise ValueError("FSDP2 resume evidence is not from HCCL/FSDP2")
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "sampler_cursor",
        "global_step",
    ):
        if value.get(key) is not True:
            raise ValueError(f"FSDP2 resume evidence is missing {key} parity")
    if value.get("dflash_bool_sdpa") is not True:
        raise ValueError("A2 evidence did not exercise DFlash bool-mask SDPA")
    if value.get("training_window_tokens") != 4096:
        raise ValueError("A2 evidence did not exercise the fixed 4096-token window")


def _hbm(value: dict[str, Any]) -> None:
    if value.get("passed") is not True:
        raise ValueError("HBM headroom gate did not pass")
    if value.get("representative") is not True:
        raise ValueError("HBM evidence is not representative")
    peak = int(value["peak_bytes"])
    capacity = int(value["capacity_bytes"])
    if peak < 0 or capacity <= 0 or peak >= capacity:
        raise ValueError("HBM evidence has no positive headroom")


def _evaluate(
    path: Path, validator: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    try:
        value = _read_object(path)
        validator(value)
        return value
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        try:
            value = _read_object(path)
        except Exception:
            value = {}
        value["passed"] = False
        value["gate_error"] = f"{type(exc).__name__}: {exc}"
        return value


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the four fail-closed Ascend Stage B training gates."
    )
    parser.add_argument("--tap-mapping-evidence", type=Path, required=True)
    parser.add_argument("--final-logit-evidence", type=Path, required=True)
    parser.add_argument("--fsdp-resume-evidence", type=Path, required=True)
    parser.add_argument("--hbm-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = {
        "tap_mapping": _evaluate(args.tap_mapping_evidence, _tap_mapping),
        "final_logit": _evaluate(args.final_logit_evidence, _final_logit),
        "fsdp_resume": _evaluate(args.fsdp_resume_evidence, _fsdp_resume),
        "hbm": _evaluate(args.hbm_evidence, _hbm),
    }
    passed = all(item.get("passed") is True for item in evidence.values())
    record = {
        "schema": "glm53-ascend-training-gate-v1",
        "production_eligible": passed,
        "runtime_attested": False,
        "deployable_export": False,
        "evidence": evidence,
    }
    _atomic_write(args.output, record)
    if not passed:
        print(f"Ascend training gate failed; evidence written to {args.output}")
        return 1
    print(f"Ascend training gate passed; evidence written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
