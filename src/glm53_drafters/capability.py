from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


CANDIDATE_SCHEMA = "glm53-training-candidate-capability-v1"
ROLLBACK_SCHEMA = "glm53-runtime-rollback-attestation-v1"
STATE_COUNT = 34
ROLLBACK_STRATEGIES = frozenset(
    {
        "per_step_state_snapshots",
        "recompute_from_committed_checkpoint",
        "discard_and_reextend_from_committed_prefix",
    }
)


def candidate_capability(*, artifact_identity: str) -> dict[str, Any]:
    identity = str(artifact_identity).strip()
    if not identity:
        raise ValueError("artifact identity must be non-empty")
    return {
        "schema": CANDIDATE_SCHEMA,
        "artifact_identity": identity,
        "training_complete": True,
        "runtime_attested": False,
        "deployable_export": False,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_candidate_capability(
    path: str | Path, *, artifact_identity: str
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"candidate capability is immutable: {destination}")
    record = candidate_capability(artifact_identity=artifact_identity)
    _atomic_write_json(destination, record)
    return record


def validate_candidate_capability(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    if record.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("unsupported candidate capability schema")
    if record.get("runtime_attested") is not False:
        raise ValueError("candidate runtime_attested must remain false")
    if record.get("deployable_export") is not False:
        raise ValueError("candidate cannot contain a deployable export")
    if record.get("training_complete") is not True:
        raise ValueError("candidate training is not complete")
    if not str(record.get("artifact_identity", "")).strip():
        raise ValueError("candidate artifact identity is missing")
    return record


def assert_runtime_usable(
    candidate: Mapping[str, Any], *, rollback_attestation: Mapping[str, Any] | None
) -> None:
    validate_candidate_capability(candidate)
    if rollback_attestation is None:
        raise RuntimeError("runtime use requires a separate rollback attestation")
    attestation = dict(rollback_attestation)
    if attestation.get("schema") != ROLLBACK_SCHEMA:
        raise RuntimeError("runtime rollback attestation schema is invalid")
    if attestation.get("artifact_identity") != candidate.get("artifact_identity"):
        raise RuntimeError("runtime rollback attestation artifact identity differs")
    if attestation.get("strategy") not in ROLLBACK_STRATEGIES:
        raise RuntimeError("runtime rollback strategy is not attested")
    if attestation.get("state_count") != STATE_COUNT:
        raise RuntimeError("runtime use requires parity for all 34 target states")
    required_parity = (
        "all_state_parity",
        "recurrent_state_parity",
        "short_convolution_state_parity",
    )
    if any(attestation.get(key) is not True for key in required_parity):
        raise RuntimeError("runtime use requires all recurrent and short-convolution parity")
