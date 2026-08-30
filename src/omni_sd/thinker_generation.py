"""Shared records and shard checks for Thinker response and target-cache jobs."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from omni_sd.config import load_config
from omni_sd.data_io import atomic_write_json
from omni_sd.provenance import artifact_record, verify_artifact_record
from omni_sd.thinker_data import stable_hex


DEFAULT_CONFIG = "configs/generate_thinker_data.yaml"
TRAJECTORY_SCHEMA = pa.schema(
    [
        ("condition_id", pa.string()),
        ("source", pa.string()),
        ("source_subset", pa.string()),
        ("modality", pa.string()),
        ("language", pa.string()),
        ("task", pa.string()),
        ("messages_json", pa.large_string()),
        ("tools_json", pa.large_string()),
        ("media_json", pa.large_string()),
        ("prompt_token_ids", pa.large_list(pa.int32())),
        ("response_token_ids", pa.large_list(pa.int32())),
        ("response_text", pa.large_string()),
        ("prompt_tokens", pa.int32()),
        ("response_tokens", pa.int32()),
        ("finish_reason", pa.string()),
        ("sampling_seed", pa.int64()),
    ]
)


def read_config(path: str | Path) -> dict[str, Any]:
    return load_config(path)


def condition_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    return (
        Path(config["input"]["conditions"]),
        Path(config["input"]["success_marker"]),
    )


def trajectory_root(config: dict[str, Any]) -> Path:
    return Path(config["output"]["root"])


def generation_fingerprint(config: dict[str, Any]) -> str:
    """Prevent resumed output from mixing generation configurations."""

    return stable_hex(
        "vllm-ascend-thinker-v2",
        config["model"],
        config["generation"],
        config["runtime"],
        config["vllm_batch_size"],
        int(config["output"]["conditions_per_shard"]),
    )


def require_verified_conditions(config: dict[str, Any]) -> tuple[Path, int]:
    input_path, marker_path = condition_paths(config)
    if not input_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError("accepted Thinker conditions are not ready")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = int(config["input"]["expected_conditions"])
    if marker.get("status") != "PASS" or int(marker.get("conditions", -1)) != expected:
        raise ValueError(
            "accepted Thinker dataset has not passed exact-count verification"
        )
    rows = pq.ParquetFile(input_path).metadata.num_rows
    if rows != expected:
        raise ValueError(f"accepted condition count {rows} != {expected}")
    if "conditions_artifact" in marker:
        verify_artifact_record(marker["conditions_artifact"], root=marker_path.parent)
    return input_path, rows


def media_content(item: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    media_type = str(item["type"]).lower()
    path = Path(str(item.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(path)
    content: dict[str, Any] = {"type": media_type, media_type: str(path)}
    if media_type == "image":
        content["max_pixels"] = int(generation["image_max_pixels"])
    elif media_type == "video":
        content.update(
            fps=float(generation["video_fps"]),
            min_frames=int(generation["video_min_frames"]),
            max_frames=int(generation["video_max_frames"]),
            total_pixels=int(generation["max_model_tokens"] * 28**2 * 0.9),
        )
    elif media_type != "audio":
        raise ValueError(f"unsupported media type: {media_type}")
    return content


def model_conversation(
    row: dict[str, Any], generation: dict[str, Any]
) -> list[dict[str, Any]]:
    messages = json.loads(str(row["messages_json"]))
    media = json.loads(str(row["media_json"]))
    system_prompt = generation.get("system_prompt")
    if system_prompt:
        messages.insert(0, {"role": "system", "content": str(system_prompt)})
    if media:
        user_index = next(
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index]["role"] == "user"
        )
        text = str(messages[user_index]["content"])
        messages[user_index] = {
            "role": "user",
            "content": [
                *(media_content(item, generation) for item in media),
                {"type": "text", "text": text},
            ],
        }
    return messages


def batch_kind(row: dict[str, Any]) -> str:
    media = json.loads(str(row["media_json"]))
    types = [str(item.get("type", "")).lower() for item in media]
    if not types:
        return "text"
    if len(types) > 1:
        return "multi_image" if set(types) == {"image"} else "other"
    return {"audio": "audio", "image": "image", "video": "video"}.get(types[0], "other")


def modality_batches(
    rows: list[dict[str, Any]], sizes: dict[str, Any]
) -> Iterable[list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = buckets[batch_kind(row)]
        bucket.append(row)
        if len(bucket) == int(sizes[batch_kind(row)]):
            yield bucket[:]
            bucket.clear()
    for bucket in buckets.values():
        if bucket:
            yield bucket


def completed_conditions(output: Path) -> int:
    if not (output / "shards").is_dir():
        return 0
    return sum(
        int(json.loads(path.read_text(encoding="utf-8"))["conditions"])
        for path in (output / "shards").glob("*.success.json")
    )


def verify_trajectories(config: dict[str, Any]) -> None:
    _, expected = require_verified_conditions(config)
    output = trajectory_root(config)
    shard_rows = int(config["output"]["conditions_per_shard"])
    expected_shards = (expected + shard_rows - 1) // shard_rows
    condition_ids: set[str] = set()
    rows = 0
    finish_reasons: dict[str, int] = defaultdict(int)
    shards = []
    runtime = None
    for shard_id in range(expected_shards):
        parquet = output / "shards" / f"train-{shard_id:05d}.parquet"
        marker_path = output / "shards" / f"train-{shard_id:05d}.success.json"
        if not parquet.is_file() or not marker_path.is_file():
            raise FileNotFoundError(f"missing generated shard {shard_id}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("generation_fingerprint") != generation_fingerprint(config):
            raise ValueError(
                f"generated shard {shard_id} uses a different configuration"
            )
        if runtime is None:
            runtime = marker.get("runtime")
        elif marker.get("runtime") != runtime:
            raise ValueError("generated shards mix different runtime versions/topologies")
        for batch in pq.ParquetFile(parquet).iter_batches(
            columns=["condition_id", "response_token_ids", "finish_reason"]
        ):
            for row in batch.to_pylist():
                condition_id = str(row["condition_id"])
                if condition_id in condition_ids or not row["response_token_ids"]:
                    raise ValueError(f"invalid generated condition {condition_id}")
                condition_ids.add(condition_id)
                finish_reasons[str(row["finish_reason"])] += 1
                rows += 1
        record = artifact_record(parquet, relative_to=output)
        if marker.get("parquet_artifact") != record:
            raise ValueError(f"generated shard {shard_id} checksum mismatch")
        shards.append({**record, "rows": marker["conditions"]})
    if rows != expected:
        raise ValueError(f"generated condition count {rows} != {expected}")
    atomic_write_json(
        output / "manifest.json",
        {
            "status": "PASS",
            "dataset_name": config["dataset_name"],
            "conditions": rows,
            "backend": "vllm_ascend",
            "model": config["model"],
            "runtime": runtime,
            "generation": config["generation"],
            "generation_fingerprint": generation_fingerprint(config),
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "shards": shards,
        },
    )
    print(f"verification PASS: {rows:,} Thinker trajectories", flush=True)


def require_trajectories(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = trajectory_root(config)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Thinker responses are not complete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise ValueError("Thinker response manifest is not PASS")
    if manifest.get("generation_fingerprint") != generation_fingerprint(config):
        raise ValueError("Thinker responses use a different generation configuration")
    return root, manifest
