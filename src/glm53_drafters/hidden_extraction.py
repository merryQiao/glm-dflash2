from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .contracts import TARGET_CONTRACT, validate_loss_mask, validate_token_ids
from .hidden_capture import TargetHiddenCapture
from .hidden_cache import HiddenCacheSpec, PackedHiddenWriter


class HiddenRunner(Protocol):
    hidden_size: int
    physical_layer_ids: Sequence[int]
    backend_metadata: Mapping[str, Any]
    capture_mapping: Sequence[Any]

    def extract(self, input_ids: Sequence[int]) -> TargetHiddenCapture: ...


def estimate_packed_cache_bytes(
    total_tokens: int, *, num_layers: int, hidden_size: int
) -> int:
    if total_tokens < 0 or num_layers < 1 or hidden_size < 1:
        raise ValueError("invalid cache-size dimensions")
    return int(total_tokens) * (
        2 * (int(num_layers) + 1) * int(hidden_size) + 8 + 1
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_manifest(
    path: Path,
    *,
    allow_smoke_unverified: bool = False,
    smoke_max_samples: int | None = None,
    expected_target_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"trajectory manifest is missing: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("trajectory manifest is not an object")
    if value.get("schema_version") != 3:
        raise ValueError("trajectory manifest must use schema version 3")
    status = value.get("status")
    if status == "smoke_failed":
        raise ValueError("smoke_failed Stage A artifacts are always rejected")
    if status == "smoke_unverified":
        if not allow_smoke_unverified:
            raise ValueError("smoke_unverified input requires explicit smoke opt-in")
        if smoke_max_samples is None or not 1 <= int(smoke_max_samples) <= 50:
            raise ValueError("explicit smoke_max_samples must be in 1..50")
        committed = int(value.get("committed_ids", -1))
        if committed < 0 or committed > 50 or committed > int(smoke_max_samples):
            raise ValueError("smoke_unverified artifact exceeds the 50-sample bound")
        if value.get("production_eligible") is not False:
            raise ValueError("smoke_unverified artifact must be production_eligible=false")
    elif status == "frozen":
        if "production_eligible" not in value:
            raise ValueError("trajectory manifest is missing production_eligible")
        if value.get("production_eligible") is not True:
            raise ValueError("trajectory manifest production eligible must be true")
    else:
        raise ValueError(
            f"trajectory manifest is not a frozen production artifact: {status!r}"
        )
    if int(value.get("unresolved_errors", 0)) != 0:
        raise ValueError("trajectory manifest contains unresolved generation errors")
    identity_labels = {
        "model_fingerprint": "model fingerprint",
        "model_revision": "model revision",
        "tokenizer_fingerprint": "tokenizer fingerprint",
        "vocab_size": "vocab size",
    }
    for key, label in identity_labels.items():
        if key not in value:
            raise ValueError(f"trajectory manifest is missing {key}")
        actual = value[key]
        if key == "vocab_size":
            if (
                not isinstance(actual, int)
                or isinstance(actual, bool)
                or actual != TARGET_CONTRACT.vocab_size
            ):
                raise ValueError(
                    f"trajectory manifest {label} must be {TARGET_CONTRACT.vocab_size}"
                )
        elif not isinstance(actual, str) or not actual.strip():
            raise ValueError(f"trajectory manifest {label} is empty")
        if expected_target_identity is not None:
            if key not in expected_target_identity:
                raise ValueError(f"expected target identity is missing {key}")
            if actual != expected_target_identity[key]:
                raise ValueError(f"trajectory manifest {label} differs from target")
    if not path.is_file():
        raise FileNotFoundError(path)
    if "jsonl_bytes" in value and path.stat().st_size != int(value["jsonl_bytes"]):
        raise ValueError("trajectory JSONL size differs from manifest")
    if "jsonl_sha256" in value and _file_sha256(path) != str(value["jsonl_sha256"]):
        raise ValueError("trajectory JSONL digest differs from manifest")
    return value


def read_frozen_trajectories(
    path: str | Path,
    *,
    allow_smoke_unverified: bool = False,
    smoke_max_samples: int | None = None,
    expected_target_identity: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    path = Path(path)
    manifest = _trajectory_manifest(
        path,
        allow_smoke_unverified=allow_smoke_unverified,
        smoke_max_samples=smoke_max_samples,
        expected_target_identity=expected_target_identity,
    )
    count = 0
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            if not value.get("stage_a_complete"):
                raise ValueError(f"{path}:{number} is not a complete Stage A record")
            if not str(value.get("id") or ""):
                raise ValueError(f"{path}:{number} has no sample ID")
            if not isinstance(value.get("generation_route"), str) or not value[
                "generation_route"
            ]:
                raise ValueError(f"{path}:{number} has no canonical generation route")
            ids = value.get("input_ids")
            mask = value.get("loss_mask")
            if (
                not isinstance(ids, list)
                or not isinstance(mask, list)
                or len(ids) != len(mask)
                or not ids
            ):
                raise ValueError(f"{path}:{number} has invalid token arrays")
            try:
                validate_token_ids(ids, vocab_size=int(manifest["vocab_size"]))
                validate_loss_mask(mask, expected_length=len(ids))
            except ValueError as exc:
                raise ValueError(f"{path}:{number} has invalid {exc}") from exc
            if value.get("token_contract", {}).get("mask_semantics") != (
                "dflash_target_token"
            ):
                raise ValueError(f"{path}:{number} has incompatible mask semantics")
            count += 1
            if manifest.get("status") == "smoke_unverified" and count > 50:
                raise ValueError("smoke_unverified artifact exceeds 50 records")
            yield value
    if count != int(manifest.get("committed_ids", count)):
        raise ValueError("trajectory record count differs from manifest committed_ids")


def _existing_ids(output_dir: Path) -> set[str]:
    path = output_dir / "index.jsonl"
    if not path.exists():
        return set()
    result: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.add(str(json.loads(line)["sample_id"]))
    return result


def extract_trajectory_cache(
    *,
    trajectory_path: str | Path,
    output_dir: str | Path,
    runner: HiddenRunner,
    logical_layer_ids: Sequence[int],
    expected_target_identity: Mapping[str, Any],
    max_segment_bytes: int = 64 << 30,
    max_samples: int | None = None,
    allow_smoke_unverified: bool = False,
    smoke_max_samples: int | None = None,
) -> int:
    trajectory_path = Path(trajectory_path)
    output_dir = Path(output_dir)
    logical = tuple(int(value) for value in logical_layer_ids)
    physical = tuple(int(value) for value in runner.physical_layer_ids)
    if len(logical) != len(physical):
        raise ValueError("logical and physical capture-layer counts differ")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    source_manifest = _trajectory_manifest(
        trajectory_path,
        allow_smoke_unverified=allow_smoke_unverified,
        smoke_max_samples=smoke_max_samples,
        expected_target_identity=expected_target_identity,
    )
    is_smoke = source_manifest.get("status") == "smoke_unverified"
    capture_mapping = tuple(tap.as_tuple() for tap in runner.capture_mapping)
    if tuple(item[1] for item in capture_mapping) != logical:
        raise ValueError("runner capture mapping differs from requested logical layers")
    ascend_a2_attestation = getattr(runner, "ascend_a2_attestation", None)
    provenance = {
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_sha256": _file_sha256(trajectory_path),
        "trajectory_manifest_sha256": _file_sha256(
            trajectory_path.with_suffix(trajectory_path.suffix + ".manifest.json")
        ),
        "logical_layer_ids": list(logical),
        "physical_layer_ids": list(physical),
        "capture_mapping": [list(item) for item in capture_mapping],
        "backend": dict(runner.backend_metadata),
        "model_fingerprint": source_manifest["model_fingerprint"],
        "model_revision": source_manifest["model_revision"],
        "tokenizer_fingerprint": source_manifest["tokenizer_fingerprint"],
        "vocab_size": source_manifest["vocab_size"],
        "target_hidden_dtype": "bfloat16",
        "source_status": source_manifest.get("status"),
        "production_eligible": not is_smoke,
    }
    if ascend_a2_attestation is not None:
        provenance["ascend_a2_runtime"] = dict(ascend_a2_attestation)
    spec = HiddenCacheSpec(
        layer_ids=logical,
        hidden_size=int(runner.hidden_size),
        capture_mapping=capture_mapping,
    )
    done = _existing_ids(output_dir)
    count = 0
    reached_eof = True
    with PackedHiddenWriter(
        output_dir,
        spec=spec,
        provenance=provenance,
        ascend_a2_attestation=ascend_a2_attestation,
        max_segment_bytes=max_segment_bytes,
    ) as writer:
        for row in read_frozen_trajectories(
            trajectory_path,
            allow_smoke_unverified=allow_smoke_unverified,
            smoke_max_samples=smoke_max_samples,
            expected_target_identity=expected_target_identity,
        ):
            sample_id = str(row["id"])
            if sample_id in done:
                continue
            if max_samples is not None and count >= max_samples:
                reached_eof = False
                break
            input_ids = list(
                validate_token_ids(
                    row["input_ids"], vocab_size=int(source_manifest["vocab_size"])
                )
            )
            capture = runner.extract(input_ids).cpu_bfloat16()
            if capture.logical_layer_ids != logical:
                raise ValueError("runner returned a different logical layer order")
            writer.append(
                sample_id=sample_id,
                source_index=int(
                    row.get("source_metadata", {}).get("selected_source_index", -1)
                ),
                input_ids=input_ids,
                loss_mask=row["loss_mask"],
                aux_hidden_states=capture.aux_hidden_states,
                target_final_hidden=capture.target_final_hidden,
                attestation=capture.attestation,
                metadata={
                    "generation_route": row.get("generation_route"),
                    "token_contract": row.get("token_contract", {}),
                    "source_metadata": row.get("source_metadata", {}),
                },
            )
            done.add(sample_id)
            count += 1
        if reached_eof:
            writer.freeze()
    return count
