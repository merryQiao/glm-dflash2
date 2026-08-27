"""Pure input and metric contracts for the Ascend Thinker profiler."""

from __future__ import annotations

import copy
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from omni_sd.thinker_data import canonical_json


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
        result.update(_latency_summary(observed))
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
        "request_engine_latency_ms": _optional_latency_summary(
            [request.get("engine_request_seconds") for request in requests]
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
