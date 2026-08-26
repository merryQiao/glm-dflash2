from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch

from .hidden_capture import TargetHiddenCapture
from .hidden_cache import HiddenCacheSpec, PackedHiddenWriter


_VALIDATED_TRAJECTORY_FILES: set[tuple[str, int, int]] = set()


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
    # BF16 hidden + int64 token ID + uint8 loss mask. JSON index overhead is
    # intentionally excluded and covered by the caller's safety factor.
    return int(total_tokens) * (2 * (int(num_layers) + 1) * int(hidden_size) + 8 + 1)


def _trajectory_manifest(
    path: Path, *, allow_partial: bool = False
) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"trajectory manifest is missing: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_statuses = {"frozen", "partial"} if allow_partial else {"frozen"}
    if value.get("status") not in allowed_statuses:
        raise ValueError(f"trajectory manifest is not frozen: {value.get('status')!r}")
    if int(value.get("unresolved_errors", 0)) != 0:
        raise ValueError("trajectory manifest contains unresolved generation errors")
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if "jsonl_bytes" in value and stat.st_size != int(value["jsonl_bytes"]):
        raise ValueError("trajectory JSONL size differs from frozen manifest")
    if key not in _VALIDATED_TRAJECTORY_FILES:
        if "jsonl_sha256" in value and _file_sha256(path) != value["jsonl_sha256"]:
            raise ValueError("trajectory JSONL digest differs from frozen manifest")
        _VALIDATED_TRAJECTORY_FILES.add(key)
    return value


def read_frozen_trajectories(
    path: str | Path, *, allow_partial: bool = False
) -> Iterator[dict[str, Any]]:
    path = Path(path)
    _trajectory_manifest(path, allow_partial=allow_partial)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            if not value.get("stage_a_complete"):
                raise ValueError(f"{path}:{number} is not a complete Stage A record")
            ids = value.get("input_ids")
            mask = value.get("loss_mask")
            if not isinstance(ids, list) or not isinstance(mask, list) or len(ids) != len(mask):
                raise ValueError(f"{path}:{number} has invalid token arrays")
            if value.get("token_contract", {}).get("mask_semantics") != "dflash_target_token":
                raise ValueError(f"{path}:{number} has incompatible mask semantics")
            yield value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_ids(output_dir: Path) -> set[str]:
    path = output_dir / "index.jsonl"
    if not path.exists():
        return set()
    result = set()
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
    max_segment_bytes: int = 64 << 30,
    max_samples: int | None = None,
    allow_partial_trajectory: bool = False,
) -> int:
    trajectory_path = Path(trajectory_path)
    output_dir = Path(output_dir)
    logical = tuple(int(value) for value in logical_layer_ids)
    physical = tuple(int(value) for value in runner.physical_layer_ids)
    if len(logical) != len(physical):
        raise ValueError("logical and physical capture-layer counts differ")
    source_manifest = _trajectory_manifest(
        trajectory_path, allow_partial=allow_partial_trajectory
    )
    provenance = {
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_sha256": _file_sha256(trajectory_path),
        "trajectory_manifest": source_manifest,
        "logical_layer_ids": list(logical),
        "physical_layer_ids": list(physical),
        "capture_mapping": [tap.as_tuple() for tap in runner.capture_mapping],
        "backend": dict(runner.backend_metadata),
    }
    done = _existing_ids(output_dir)
    count = 0
    spec = HiddenCacheSpec(
        layer_ids=logical,
        hidden_size=int(runner.hidden_size),
        capture_mapping=tuple(tap.as_tuple() for tap in runner.capture_mapping),
    )
    with PackedHiddenWriter(
        output_dir,
        spec=spec,
        max_segment_bytes=max_segment_bytes,
        provenance=provenance,
    ) as writer:
        reached_eof = True
        for row in read_frozen_trajectories(
            trajectory_path, allow_partial=allow_partial_trajectory
        ):
            sample_id = str(row["id"])
            if sample_id in done:
                continue
            if max_samples is not None and count >= max_samples:
                reached_eof = False
                break
            input_ids = [int(value) for value in row["input_ids"]]
            capture = runner.extract(input_ids)
            if capture.logical_layer_ids != logical:
                raise ValueError("runner capture mapping differs from requested logical layers")
            source_index = int(row.get("source_metadata", {}).get("selected_source_index", -1))
            writer.append(
                sample_id=sample_id,
                source_index=source_index,
                input_ids=input_ids,
                loss_mask=row["loss_mask"],
                aux_hidden_states=capture.aux_hidden_states,
                target_final_hidden=capture.target_final_hidden,
                metadata={
                    "token_contract": row.get("token_contract", {}),
                    "source_metadata": row.get("source_metadata", {}),
                },
            )
            done.add(sample_id)
            count += 1
        if reached_eof and source_manifest.get("status") == "frozen":
            writer.freeze()
    return count
