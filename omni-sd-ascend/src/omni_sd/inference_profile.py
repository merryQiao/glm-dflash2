"""Pure input and metric contracts for the Ascend Thinker profiler."""

from __future__ import annotations

import copy
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from omni_sd.thinker_data import canonical_json
from omni_sd.ascend_runtime import parse_visible_devices


MEDIA_TYPES = ("audio", "image", "video")
SCORE_VERSION = "omni_eval_v1"
EVALUATION_METRICS = {
    "exact_match",
    "normalized_exact_match",
    "multiple_choice_accuracy",
}
MULTIPLE_CHOICE_PATTERN = re.compile(
    r"^(?:ANSWER\s*[:=]\s*|OPTION\s+)?([A-Z])(?:[.)])?$"
)


class ProfileContractError(ValueError):
    """Raised before reporting misleading inference performance."""


HBM_FIELDS = (
    "allocated_current_bytes",
    "reserved_current_bytes",
    "allocated_peak_bytes",
    "reserved_peak_bytes",
)


def _worker_npu_context() -> tuple[Any, dict[str, int]]:
    """Resolve rank-local torch_npu state inside a vLLM worker process."""

    import torch
    import torch_npu  # noqa: F401 - registers torch.npu in worker processes

    distributed = getattr(torch, "distributed", None)
    distributed_available = (
        distributed is not None
        and (
            not hasattr(distributed, "is_available")
            or distributed.is_available()
        )
    )
    if (
        distributed_available
        and distributed.is_initialized()
    ):
        rank = int(distributed.get_rank())
    else:
        rank = int(os.environ.get("RANK", "0"))
    logical_device = int(torch.npu.current_device())
    visible_value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not visible_value:
        raise ProfileContractError(
            "ASCEND_RT_VISIBLE_DEVICES is required for physical NPU accounting"
        )
    try:
        visible_devices = parse_visible_devices(visible_value)
    except ValueError as error:
        raise ProfileContractError(str(error)) from error
    if logical_device >= len(visible_devices):
        raise ProfileContractError(
            "logical NPU index is outside ASCEND_RT_VISIBLE_DEVICES"
        )
    return torch, {
        "rank": rank,
        "logical_device": logical_device,
        "physical_device": int(visible_devices[logical_device]),
    }


def worker_reset_npu_peak(_worker: Any) -> dict[str, int]:
    """vLLM collective-RPC callback: reset this worker's allocator peak."""

    torch, identity = _worker_npu_context()
    torch.npu.synchronize(identity["logical_device"])
    torch.npu.reset_peak_memory_stats(identity["logical_device"])
    return identity


def worker_snapshot_npu_memory(_worker: Any) -> dict[str, int]:
    """vLLM collective-RPC callback: snapshot this worker's allocator state."""

    torch, identity = _worker_npu_context()
    device = identity["logical_device"]
    torch.npu.synchronize(device)
    return {
        **identity,
        "allocated_current_bytes": int(torch.npu.memory_allocated(device)),
        "reserved_current_bytes": int(torch.npu.memory_reserved(device)),
        "allocated_peak_bytes": int(torch.npu.max_memory_allocated(device)),
        "reserved_peak_bytes": int(torch.npu.max_memory_reserved(device)),
    }


def validate_hbm_worker_identities(
    snapshots: Sequence[Mapping[str, Any]], tensor_parallel_size: int
) -> list[dict[str, int]]:
    """Require one unique rank and physical NPU for every TP worker."""

    if len(snapshots) != tensor_parallel_size:
        raise ProfileContractError(
            f"HBM snapshot must contain exactly {tensor_parallel_size} worker ranks"
        )
    ranks = [int(snapshot["rank"]) for snapshot in snapshots if "rank" in snapshot]
    physical = [
        int(snapshot["physical_device"])
        for snapshot in snapshots
        if "physical_device" in snapshot
    ]
    if len(ranks) == tensor_parallel_size and len(set(ranks)) != tensor_parallel_size:
        raise ProfileContractError("HBM snapshot worker ranks must be unique")
    if (
        len(physical) == tensor_parallel_size
        and len(set(physical)) != tensor_parallel_size
    ):
        raise ProfileContractError(
            "HBM snapshot workers must map to a unique physical NPU"
        )
    identities: list[dict[str, int]] = []
    for snapshot in snapshots:
        required = ("rank", "logical_device", "physical_device")
        missing = [field for field in required if field not in snapshot]
        if missing:
            raise ProfileContractError(
                f"HBM snapshot is missing fields: {', '.join(missing)}"
            )
        identities.append({field: int(snapshot[field]) for field in required})
    return sorted(identities, key=lambda item: item["rank"])


def _validated_hbm_batch(
    snapshots: Sequence[Mapping[str, Any]], tensor_parallel_size: int
) -> list[dict[str, int]]:
    validate_hbm_worker_identities(snapshots, tensor_parallel_size)
    normalized: list[dict[str, int]] = []
    for snapshot in snapshots:
        required = ("rank", "logical_device", "physical_device", *HBM_FIELDS)
        missing = [field for field in required if field not in snapshot]
        if missing:
            raise ProfileContractError(
                f"HBM snapshot is missing fields: {', '.join(missing)}"
            )
        item = {field: int(snapshot[field]) for field in required}
        if any(item[field] < 0 for field in required):
            raise ProfileContractError("HBM snapshot values must be non-negative")
        normalized.append(item)
    return sorted(normalized, key=lambda item: item["rank"])


def reduce_hbm_measurements(
    batch_snapshots: Sequence[Sequence[Mapping[str, Any]]],
    *,
    tensor_parallel_size: int,
) -> dict[str, Any]:
    """Reduce rank-local allocator snapshots without conflating peak semantics."""

    if tensor_parallel_size <= 0:
        raise ProfileContractError("tensor_parallel_size must be positive")
    if not batch_snapshots:
        raise ProfileContractError("no HBM snapshots were collected")
    batches = [
        _validated_hbm_batch(batch, tensor_parallel_size)
        for batch in batch_snapshots
    ]
    expected_mapping = {
        item["rank"]: item["physical_device"] for item in batches[0]
    }
    for batch in batches[1:]:
        mapping = {item["rank"]: item["physical_device"] for item in batch}
        if mapping != expected_mapping:
            raise ProfileContractError("HBM worker/device mapping changed between batches")

    by_rank: dict[str, Any] = {}
    for rank, physical_device in sorted(expected_mapping.items()):
        samples = [
            next(item for item in batch if item["rank"] == rank)
            for batch in batches
        ]
        final = samples[-1]
        by_rank[str(rank)] = {
            "logical_device": final["logical_device"],
            "physical_device": physical_device,
            "final_current": {
                "allocated_bytes": final["allocated_current_bytes"],
                "reserved_bytes": final["reserved_current_bytes"],
            },
            "max_post_batch_current": {
                "allocated_bytes": max(
                    item["allocated_current_bytes"] for item in samples
                ),
                "reserved_bytes": max(
                    item["reserved_current_bytes"] for item in samples
                ),
            },
            "max_batch_peak": {
                "allocated_bytes": max(item["allocated_peak_bytes"] for item in samples),
                "reserved_bytes": max(item["reserved_peak_bytes"] for item in samples),
            },
        }

    return {
        "available": True,
        "source": "torch_npu_allocator",
        "tensor_parallel_size": tensor_parallel_size,
        "batches_observed": len(batches),
        "per_rank": by_rank,
        "max_rank_peak": {
            "allocated_bytes": max(
                item["allocated_peak_bytes"] for batch in batches for item in batch
            ),
            "reserved_bytes": max(
                item["reserved_peak_bytes"] for batch in batches for item in batch
            ),
        },
        "max_batch_sum_of_rank_peaks": {
            "allocated_bytes": max(
                sum(item["allocated_peak_bytes"] for item in batch)
                for batch in batches
            ),
            "reserved_bytes": max(
                sum(item["reserved_peak_bytes"] for item in batch)
                for batch in batches
            ),
        },
    }


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
    evaluation = record.get("evaluation")
    if evaluation is not None and not isinstance(evaluation, Mapping):
        raise ProfileContractError("evaluation must be a mapping")
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
        "evaluation_json": canonical_json(dict(evaluation)) if evaluation else None,
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
    unique_types = set(types)
    if unique_types == {"image"} and len(types) > 1:
        return "multi_image"
    if len(unique_types) == 1:
        return next(iter(unique_types))
    return "other"


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


def _latency_summary(
    seconds: Sequence[float], *, allow_zero: bool = False
) -> dict[str, float]:
    milliseconds = [float(value) * 1000.0 for value in seconds]
    invalid = (
        any(value < 0 for value in milliseconds)
        if allow_zero
        else any(value <= 0 for value in milliseconds)
    )
    if not milliseconds or invalid:
        requirement = "non-negative" if allow_zero else "positive"
        raise ProfileContractError(f"latencies must be {requirement}")
    return {
        "mean": sum(milliseconds) / len(milliseconds),
        "p50": _percentile(milliseconds, 0.50),
        "p95": _percentile(milliseconds, 0.95),
        "p99": _percentile(milliseconds, 0.99),
        "max": max(milliseconds),
    }


def request_stage_latencies_seconds(output: Any) -> dict[str, float | None]:
    """Return valid intervals from vLLM 0.23 engine-core monotonic timestamps."""

    metrics = getattr(output, "metrics", None)

    def interval(start_name: str, end_name: str) -> float | None:
        start = getattr(metrics, start_name, None)
        end = getattr(metrics, end_name, None)
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return None
        start_value = float(start)
        end_value = float(end)
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            return None
        if start_value <= 0 or end_value <= 0 or end_value < start_value:
            return None
        return end_value - start_value

    return {
        "queue_seconds": interval("queued_ts", "scheduled_ts"),
        "prefill_seconds": interval("scheduled_ts", "first_token_ts"),
        "decode_seconds": interval("first_token_ts", "last_token_ts"),
        "engine_inference_seconds": interval("scheduled_ts", "last_token_ts"),
    }


def _evaluation_payload(value: Any) -> dict[str, str] | None:
    if value in (None, "", "null"):
        return None
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, Mapping):
        raise ProfileContractError("evaluation metadata must be a mapping")
    metric = str(payload.get("metric", ""))
    reference = payload.get("reference")
    if metric not in EVALUATION_METRICS:
        raise ProfileContractError(f"unsupported evaluation metric: {metric}")
    if not isinstance(reference, str):
        raise ProfileContractError("evaluation reference must be a string")
    return {"metric": metric, "reference": reference}


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in value
    )
    return " ".join(value.split())


def score_prediction(metric: str, prediction: str, reference: str) -> bool:
    """Score one prediction with the frozen, dependency-free v1 contract."""

    if metric not in EVALUATION_METRICS:
        raise ProfileContractError(f"unsupported evaluation metric: {metric}")
    if not isinstance(prediction, str) or not isinstance(reference, str):
        raise ProfileContractError("prediction/reference must be strings")
    if metric == "exact_match":
        return prediction == reference
    if metric == "normalized_exact_match":
        return _normalized_text(prediction) == _normalized_text(reference)
    normalized_reference = unicodedata.normalize("NFKC", reference).upper().strip()
    if not re.fullmatch(r"[A-Z]", normalized_reference):
        raise ProfileContractError(
            "multiple-choice reference must be exactly one ASCII A-Z letter"
        )
    normalized_prediction = unicodedata.normalize("NFKC", prediction).upper().strip()
    match = MULTIPLE_CHOICE_PATTERN.fullmatch(normalized_prediction)
    return match is not None and match.group(1) == normalized_reference


def evaluation_result(
    evaluation_json: Any, prediction: str
) -> dict[str, Any] | None:
    """Return the serializable per-request v1 evaluation result."""

    metadata = _evaluation_payload(evaluation_json)
    if metadata is None:
        return None
    return {
        "metric": metadata["metric"],
        "correct": score_prediction(
            metadata["metric"], prediction, metadata["reference"]
        ),
        "scorer_version": SCORE_VERSION,
    }


def validate_evaluation_contract(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate scorer metadata before any model allocation occurs."""

    metrics: set[str] = set()
    for row in rows:
        metadata = _evaluation_payload(row.get("evaluation_json"))
        if metadata is None:
            continue
        metrics.add(metadata["metric"])
        if metadata["metric"] == "multiple_choice_accuracy":
            score_prediction(metadata["metric"], "", metadata["reference"])
    if len(metrics) > 1:
        raise ProfileContractError("mixed evaluation metrics are not supported")


def aggregate_evaluation(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parsed: list[tuple[str, dict[str, str] | None, bool | None]] = []
    metrics: set[str] = set()
    for record in records:
        metadata = _evaluation_payload(record.get("evaluation_json"))
        correct: bool | None = None
        if metadata is not None:
            metrics.add(metadata["metric"])
            correct = score_prediction(
                metadata["metric"],
                str(record.get("response_text", "")),
                metadata["reference"],
            )
        parsed.append((str(record["modality"]), metadata, correct))
    if len(metrics) > 1:
        raise ProfileContractError("mixed evaluation metrics are not supported")
    if not metrics:
        return {
            "available": False,
            "scorer_version": SCORE_VERSION,
            "evaluated": 0,
            "skipped": len(records),
            "reason": "no evaluation references",
        }

    metric = next(iter(metrics))

    def summarize(items: Sequence[tuple[str, dict[str, str] | None, bool | None]]):
        evaluated = sum(metadata is not None for _, metadata, _ in items)
        skipped = len(items) - evaluated
        if evaluated == 0:
            return {
                "available": False,
                "evaluated": 0,
                "skipped": skipped,
                "reason": "no evaluation references",
            }
        correct = sum(result is True for _, _, result in items)
        return {
            "available": True,
            "evaluated": evaluated,
            "skipped": skipped,
            "correct": correct,
            "accuracy": correct / evaluated,
        }

    modalities = sorted({modality for modality, _, _ in parsed})
    return {
        "available": True,
        "scorer_version": SCORE_VERSION,
        "metric": metric,
        "overall": summarize(parsed),
        "by_modality": {
            modality: summarize([item for item in parsed if item[0] == modality])
            for modality in modalities
        },
    }


def _optional_latency_summary(
    seconds: Sequence[float | None],
) -> dict[str, Any]:
    observed = [float(value) for value in seconds if value is not None]
    result: dict[str, Any] = {
        "available": bool(observed),
        "observed": len(observed),
        "missing": len(seconds) - len(observed),
    }
    if observed:
        result.update(_latency_summary(observed, allow_zero=True))
    else:
        result["reason"] = "vLLM request timestamps unavailable"
    return result


def _performance_slice(
    requests: Sequence[Mapping[str, Any]],
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not requests:
        raise ProfileContractError("no measured requests")
    if not batches:
        raise ProfileContractError("no measured batches")
    if sum(int(batch["requests"]) for batch in batches) != len(requests):
        raise ProfileContractError("batch/request counts do not align")

    prompt_tokens = sum(int(request["prompt_tokens"]) for request in requests)
    completion_tokens = sum(
        int(request["completion_tokens"]) for request in requests
    )
    engine_seconds = sum(float(batch["engine_seconds"]) for batch in batches)
    end_to_end_seconds = sum(
        float(batch["end_to_end_seconds"]) for batch in batches
    )
    preprocess_seconds = sum(
        float(request["preprocess_seconds"]) for request in requests
    )
    if completion_tokens <= 0:
        raise ProfileContractError("measured completions contain no tokens")
    if min(engine_seconds, end_to_end_seconds) <= 0:
        raise ProfileContractError("measured batch times must be positive")
    total_tokens = prompt_tokens + completion_tokens
    engine = {
        "requests_per_second": len(requests) / engine_seconds,
        "completion_tokens_per_second": completion_tokens / engine_seconds,
        "total_tokens_per_second": total_tokens / engine_seconds,
    }
    end_to_end = {
        "requests_per_second": len(requests) / end_to_end_seconds,
        "completion_tokens_per_second": completion_tokens / end_to_end_seconds,
        "total_tokens_per_second": total_tokens / end_to_end_seconds,
    }
    return {
        "requests": len(requests),
        "batches": len(batches),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "preprocess_seconds": preprocess_seconds,
        "engine_seconds": engine_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        # Backward-compatible names remain engine-only.
        "requests_per_second": engine["requests_per_second"],
        "completion_tokens_per_second": engine[
            "completion_tokens_per_second"
        ],
        "total_tokens_per_second": engine["total_tokens_per_second"],
        "engine": engine,
        "end_to_end": end_to_end,
        "request_preprocess_latency_ms": _latency_summary(
            [float(request["preprocess_seconds"]) for request in requests]
        ),
        "request_queue_latency_ms": _optional_latency_summary(
            [request.get("queue_seconds") for request in requests]
        ),
        "request_prefill_latency_ms": _optional_latency_summary(
            [request.get("prefill_seconds") for request in requests]
        ),
        "request_decode_latency_ms": _optional_latency_summary(
            [request.get("decode_seconds") for request in requests]
        ),
        "request_engine_inference_latency_ms": _optional_latency_summary(
            [request.get("engine_inference_seconds") for request in requests]
        ),
        "batch_engine_latency_ms": _latency_summary(
            [float(batch["engine_seconds"]) for batch in batches]
        ),
        "batch_end_to_end_latency_ms": _latency_summary(
            [float(batch["end_to_end_seconds"]) for batch in batches]
        ),
    }


def aggregate_profile_performance(
    *,
    requests: Sequence[Mapping[str, Any]],
    batches: Sequence[Mapping[str, Any]],
    model_load_seconds: float,
    warmup: Mapping[str, float],
) -> dict[str, Any]:
    """Aggregate measured batches with engine and outer wall clocks separated."""

    overall = _performance_slice(requests, batches)
    modalities = sorted({str(request["modality"]) for request in requests})
    return {
        "model_load_seconds": float(model_load_seconds),
        "warmup": {key: float(value) for key, value in warmup.items()},
        "overall": overall,
        "by_modality": {
            modality: _performance_slice(
                [request for request in requests if request["modality"] == modality],
                [batch for batch in batches if batch["modality"] == modality],
            )
            for modality in modalities
        },
    }


def aggregate_performance(
    *,
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
    batch_seconds: Sequence[float],
    request_seconds: Sequence[float],
    model_load_seconds: float,
    warmup_seconds: float,
) -> dict[str, Any]:
    """Legacy flat aggregation retained until callers adopt the richer report."""

    request_count = len(prompt_tokens)
    if request_count == 0:
        raise ProfileContractError("no measured requests")
    if len(completion_tokens) != request_count or len(request_seconds) != request_count:
        raise ProfileContractError("request token/latency counts do not align")
    if not batch_seconds or any(float(value) <= 0 for value in batch_seconds):
        raise ProfileContractError("measured batch times must be positive")
    measured = sum(float(value) for value in batch_seconds)
    prompts = sum(int(value) for value in prompt_tokens)
    completions = sum(int(value) for value in completion_tokens)
    if completions <= 0:
        raise ProfileContractError("measured completions contain no tokens")
    return {
        "requests": request_count,
        "batches": len(batch_seconds),
        "prompt_tokens": prompts,
        "completion_tokens": completions,
        "total_tokens": prompts + completions,
        "model_load_seconds": float(model_load_seconds),
        "warmup_seconds": float(warmup_seconds),
        "measured_seconds": measured,
        "requests_per_second": request_count / measured,
        "completion_tokens_per_second": completions / measured,
        "total_tokens_per_second": (prompts + completions) / measured,
        "batch_latency_ms": _latency_summary(batch_seconds),
        "request_latency_ms": _latency_summary(request_seconds),
    }


def component_availability(modalities: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Describe component execution separately from unavailable internal timing."""

    values = set(modalities)
    vision_executed = bool(values.intersection({"image", "multi_image", "video", "other"}))
    audio_executed = bool(values.intersection({"audio", "video", "other"}))
    return {
        "audio_encoder": {
            "loaded": True,
            "executed": audio_executed,
            "timing_available": False,
            "reason": "vLLM-Ascend does not expose request-scoped audio encoder events",
        },
        "vision_encoder": {
            "loaded": True,
            "executed": vision_executed,
            "timing_available": False,
            "reason": "vLLM-Ascend does not expose request-scoped vision encoder events",
        },
        "thinker": {
            "loaded": True,
            "executed": bool(values),
            "timing_available": False,
            "reason": "vLLM-Ascend exposes whole-engine time, not internal Thinker events",
        },
        "talker": {
            "loaded": False,
            "executed": False,
            "timing_available": False,
            "reason": "Thinker-only vLLM-Ascend engine does not load Talker",
        },
        "mtp": {
            "loaded": False,
            "executed": False,
            "timing_available": False,
            "reason": "Thinker-only vLLM-Ascend engine does not load MTP/code predictor",
        },
        "code2wav": {
            "loaded": False,
            "executed": False,
            "timing_available": False,
            "reason": "Thinker-only vLLM-Ascend engine does not load Code2Wav",
        },
    }
