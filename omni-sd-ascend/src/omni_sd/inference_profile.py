"""Pure input and metric contracts for the Ascend Thinker profiler."""

from __future__ import annotations

import copy
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from omni_sd.thinker_data import canonical_json


MEDIA_TYPES = ("audio", "image", "video")


class ProfileContractError(ValueError):
    """Raised before reporting misleading inference performance."""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _media_path(value: Any, base_dir: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileContractError("media path must be a non-empty string")
    if value.startswith(("http://", "https://", "file://", "data:")):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _resolve_native_messages(
    messages: Any, base_dir: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(messages, list) or not messages:
        raise ProfileContractError("messages must be a non-empty list")
    result = copy.deepcopy(messages)
    modalities: list[str] = []
    for message in result:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ProfileContractError("every message must contain a role")
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                raise ProfileContractError("multimodal content items must be mappings")
            media_type = str(item.get("type", "")).lower()
            if media_type in MEDIA_TYPES:
                item[media_type] = _media_path(item.get(media_type), base_dir)
                modalities.append(media_type)
    return result, modalities


def _modality(types: Sequence[str]) -> str:
    unique = sorted(set(types))
    return "+".join(unique) if unique else "text"


def normalize_input_record(
    record: Mapping[str, Any], *, index: int, base_dir: Path
) -> dict[str, Any]:
    """Normalize a convenient inference record into Stage-A's row contract."""

    condition_id = str(record.get("id", record.get("condition_id", index)))
    if "messages" in record:
        messages, native_modalities = _resolve_native_messages(
            record["messages"], base_dir
        )
        media: list[dict[str, str]] = []
        modalities = native_modalities
    else:
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ProfileContractError(f"record {condition_id} has no text")
        messages = [{"role": "user", "content": text}]
        media = []
        modalities = []
        for media_type in MEDIA_TYPES:
            for value in _as_list(record.get(media_type)):
                media.append(
                    {
                        "type": media_type,
                        "path": _media_path(value, base_dir),
                    }
                )
                modalities.append(media_type)

    tools = record.get("tools", [])
    if tools is None:
        tools = []
    return {
        "condition_id": condition_id,
        "source": str(record.get("source", "inference_profile")),
        "source_subset": str(record.get("source_subset", "manual")),
        "modality": str(record.get("modality", _modality(modalities))),
        "language": str(record.get("language", "und")),
        "task": str(record.get("task", "inference_profile")),
        "messages_json": canonical_json(messages),
        "tools_json": canonical_json(tools),
        "media_json": canonical_json(media),
    }


def profile_batch_kind(row: Mapping[str, Any]) -> str:
    """Classify both Stage-A rows and native multimodal-message records."""

    media = json.loads(str(row.get("media_json", "[]")))
    types = [str(item.get("type", "")).lower() for item in media]
    if not types:
        messages = json.loads(str(row.get("messages_json", "[]")))
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            types.extend(
                str(item.get("type", "")).lower()
                for item in content
                if isinstance(item, dict)
                and str(item.get("type", "")).lower() in MEDIA_TYPES
            )
    types = [value for value in types if value in MEDIA_TYPES]
    if not types:
        return "text"
    if len(types) > 1:
        return "multi_image" if set(types) == {"image"} else "other"
    return types[0]


def profile_batches(
    rows: Sequence[dict[str, Any]], sizes: Mapping[str, int]
) -> Iterable[list[dict[str, Any]]]:
    """Batch by actual media payload rather than a user-supplied label."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        kind = profile_batch_kind(row)
        size = int(sizes[kind])
        bucket = buckets[kind]
        bucket.append(row)
        if len(bucket) == size:
            yield bucket[:]
            bucket.clear()
    for bucket in buckets.values():
        if bucket:
            yield bucket


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ProfileContractError("latency distribution is empty")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latency_summary(seconds: Sequence[float]) -> dict[str, float]:
    milliseconds = [float(value) * 1000.0 for value in seconds]
    if not milliseconds or any(value <= 0 for value in milliseconds):
        raise ProfileContractError("latencies must be positive")
    return {
        "mean": sum(milliseconds) / len(milliseconds),
        "p50": _percentile(milliseconds, 0.50),
        "p95": _percentile(milliseconds, 0.95),
        "p99": _percentile(milliseconds, 0.99),
        "max": max(milliseconds),
    }


def request_latency_seconds(output: Any, batch_seconds: float) -> float:
    """Prefer vLLM request timestamps and fall back to enclosing batch time."""

    metrics = getattr(output, "metrics", None)
    arrival = getattr(metrics, "arrival_time", None)
    finished = getattr(metrics, "finished_time", None)
    if isinstance(arrival, (int, float)) and isinstance(finished, (int, float)):
        elapsed = float(finished) - float(arrival)
        if elapsed > 0:
            return elapsed
    if batch_seconds <= 0:
        raise ProfileContractError("fallback batch latency must be positive")
    return float(batch_seconds)


def aggregate_performance(
    *,
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
    batch_seconds: Sequence[float],
    request_seconds: Sequence[float],
    model_load_seconds: float,
    warmup_seconds: float,
) -> dict[str, Any]:
    """Aggregate measured calls only; load and warmup remain separate fields."""

    requests = len(prompt_tokens)
    if requests == 0:
        raise ProfileContractError("no measured requests")
    if len(completion_tokens) != requests or len(request_seconds) != requests:
        raise ProfileContractError("request token/latency counts do not align")
    if not batch_seconds or any(float(value) <= 0 for value in batch_seconds):
        raise ProfileContractError("measured batch times must be positive")
    measured = sum(float(value) for value in batch_seconds)
    prompts = sum(int(value) for value in prompt_tokens)
    completions = sum(int(value) for value in completion_tokens)
    if completions <= 0:
        raise ProfileContractError("measured completions contain no tokens")
    return {
        "requests": requests,
        "batches": len(batch_seconds),
        "prompt_tokens": prompts,
        "completion_tokens": completions,
        "total_tokens": prompts + completions,
        "model_load_seconds": float(model_load_seconds),
        "warmup_seconds": float(warmup_seconds),
        "measured_seconds": measured,
        "requests_per_second": requests / measured,
        "completion_tokens_per_second": completions / measured,
        "total_tokens_per_second": (prompts + completions) / measured,
        "batch_latency_ms": _latency_summary(batch_seconds),
        "request_latency_ms": _latency_summary(request_seconds),
    }
