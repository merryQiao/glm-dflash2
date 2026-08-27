from __future__ import annotations

import json
import hashlib
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


_COUNTERS = {
    "vllm:spec_decode_num_drafts_total": "num_drafts",
    "vllm:spec_decode_num_draft_tokens_total": "num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "num_accepted_tokens",
}
_SAMPLE = re.compile(r"^(?P<name>[^\s{]+)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)(?:\s+\d+)?$")


def parse_spec_decode_metrics(text: str) -> dict[str, float]:
    result = {name: 0.0 for name in _COUNTERS.values()}
    found: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        output_name = _COUNTERS.get(match.group("name"))
        if output_name is None:
            continue
        value = float(match.group("value"))
        if not math.isfinite(value):
            raise ValueError(f"non-finite Prometheus counter {match.group('name')}")
        result[output_name] += value
        found.add(output_name)
    if found and found != set(result):
        missing = sorted(set(result) - found)
        raise ValueError(f"incomplete speculative-decoding metrics: missing {missing}")
    return result if found else {}


def summarize_spec_decode(
    before: Mapping[str, float], after: Mapping[str, float]
) -> dict[str, float]:
    names = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    delta = {name: float(after[name]) - float(before[name]) for name in names}
    if any(value < 0 for value in delta.values()):
        raise ValueError("speculative-decoding counters decreased during the run")
    drafts = delta["num_drafts"]
    drafted = delta["num_draft_tokens"]
    accepted = delta["num_accepted_tokens"]
    if drafts <= 0 or drafted <= 0:
        raise ValueError("the server reported no speculative draft steps")
    return {
        "drafts": drafts,
        "draft_tokens": drafted,
        "accepted_tokens": accepted,
        # vLLM convention includes the verifier bonus token.
        "mean_acceptance_length": 1.0 + accepted / drafts,
        "draft_acceptance_rate": accepted / drafted,
    }


def _read_json(url: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def load_prompts(path: str | Path, *, max_samples: int = 0) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, raw in enumerate(handle):
            if max_samples > 0 and len(prompts) >= max_samples:
                break
            if not raw.strip():
                continue
            row = json.loads(raw)
            sample_id = str(row.get("id", row.get("sample_id", index)))
            if isinstance(row.get("messages"), list):
                prompts.append({"sample_id": sample_id, "messages": row["messages"]})
            elif isinstance(row.get("prompt"), str):
                prompts.append({"sample_id": sample_id, "prompt": row["prompt"]})
            else:
                raise ValueError(f"row {index} must contain messages or prompt")
    if not prompts:
        raise ValueError("prompt file is empty")
    return prompts


def prompt_fixture_sha256(prompts: Sequence[Mapping[str, Any]]) -> str:
    raw = json.dumps(
        list(prompts), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _output_token_ids(response: Mapping[str, Any], choice: Mapping[str, Any]) -> list[int]:
    candidates = (
        choice.get("token_ids"),
        choice.get("output_token_ids"),
        response.get("output_token_ids"),
        (response.get("usage") or {}).get("output_token_ids"),
    )
    for value in candidates:
        if isinstance(value, list) and all(isinstance(token, int) for token in value):
            return [int(token) for token in value]
    raise ValueError(
        "vLLM response is missing raw output token IDs; enable the pinned return_token_ids extension"
    )


def benchmark_openai_server(
    *,
    base_url: str,
    model: str,
    prompts: Sequence[Mapping[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    warmup_requests: int = 2,
    timeout: float = 1800.0,
    rejection_mode: str = "none",
) -> dict[str, Any]:
    if max_tokens < 1 or warmup_requests < 0:
        raise ValueError("max_tokens must be positive and warmup_requests non-negative")
    base_url = base_url.rstrip("/")

    def request_one(row: Mapping[str, Any], request_seed: int) -> tuple[dict[str, Any], float]:
        is_chat = "messages" in row
        endpoint = "/v1/chat/completions" if is_chat else "/v1/completions"
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": request_seed,
            "stream": False,
            # Required by the pinned vLLM-Ascend response extension. Text is
            # insufficient for lossless parity because decoding is not injective.
            "return_token_ids": True,
        }
        payload["messages" if is_chat else "prompt"] = row["messages" if is_chat else "prompt"]
        started = time.perf_counter()
        response = _read_json(base_url + endpoint, payload, timeout)
        return response, time.perf_counter() - started

    for index in range(warmup_requests):
        request_one(prompts[index % len(prompts)], seed + index)
    before = parse_spec_decode_metrics(_read_text(base_url + "/metrics", timeout))
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(prompts):
        response, latency = request_one(row, seed + index)
        if response.get("error"):
            raise RuntimeError(f"vLLM request failed: {response['error']}")
        usage = response.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            raise ValueError("vLLM response is missing usage.completion_tokens")
        choice = (response.get("choices") or [{}])[0]
        output = (
            (choice.get("message") or {}).get("content", "")
            if "messages" in row
            else choice.get("text", "")
        )
        samples.append(
            {
                "sample_id": str(row["sample_id"]),
                "completion_tokens": int(completion_tokens),
                "latency_seconds": latency,
                "output_text": str(output or ""),
                "output_token_ids": _output_token_ids(response, choice),
            }
        )
    wall = time.perf_counter() - started
    after = parse_spec_decode_metrics(_read_text(base_url + "/metrics", timeout))
    completion_tokens = sum(item["completion_tokens"] for item in samples)
    result: dict[str, Any] = {
        "schema": "glm-vllm-ascend-benchmark-v2",
        "server": base_url,
        "model": model,
        "fixture_sha256": prompt_fixture_sha256(prompts),
        "rejection_mode": str(rejection_mode),
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_tokens": max_tokens,
        },
        "summary": {
            "samples": len(samples),
            "completion_tokens": completion_tokens,
            "wall_seconds": wall,
            "tps": completion_tokens / wall if wall > 0 else float("nan"),
            "mean_request_latency_seconds": sum(item["latency_seconds"] for item in samples) / len(samples),
        },
        "samples": samples,
    }
    if before and after:
        result["spec_decode"] = summarize_spec_decode(before, after)
    return result


def compare_benchmark_results(
    baseline: Mapping[str, Any],
    speculative: Mapping[str, Any],
    *,
    require_exact_outputs: bool,
) -> dict[str, Any]:
    baseline_samples = {str(row["sample_id"]): row for row in baseline["samples"]}
    speculative_samples = {str(row["sample_id"]): row for row in speculative["samples"]}
    if baseline_samples.keys() != speculative_samples.keys():
        raise ValueError("baseline and speculative sample sets differ")
    if baseline.get("fixture_sha256") != speculative.get("fixture_sha256"):
        raise ValueError("baseline and speculative prompt fixtures differ")
    if baseline.get("sampling") != speculative.get("sampling"):
        raise ValueError("baseline and speculative sampling settings differ")
    missing_ids = [
        key
        for key in baseline_samples
        if not isinstance(baseline_samples[key].get("output_token_ids"), list)
        or not isinstance(speculative_samples[key].get("output_token_ids"), list)
    ]
    if missing_ids:
        raise ValueError(f"raw output token IDs are missing for {len(missing_ids)} samples")
    matches = [
        baseline_samples[key].get("output_token_ids")
        == speculative_samples[key].get("output_token_ids")
        for key in baseline_samples
    ]
    exact = all(matches)
    if require_exact_outputs and not exact:
        mismatched = sum(not value for value in matches)
        raise ValueError(f"lossless greedy token-ID parity failed for {mismatched} samples")
    baseline_tps = float(baseline["summary"]["tps"])
    speculative_tps = float(speculative["summary"]["tps"])
    if baseline_tps <= 0:
        raise ValueError("baseline TPS must be positive")
    spec_decode = speculative.get("spec_decode")
    required_spec_metrics = {
        "drafts",
        "draft_tokens",
        "accepted_tokens",
        "mean_acceptance_length",
        "draft_acceptance_rate",
    }
    if not isinstance(spec_decode, Mapping) or not required_spec_metrics.issubset(
        spec_decode
    ):
        raise ValueError(
            "speculative run is missing active speculative-decoding metrics"
        )
    if float(spec_decode["drafts"]) <= 0 or float(spec_decode["draft_tokens"]) <= 0:
        raise ValueError("speculative run reported no active draft steps")
    temperature = float((speculative.get("sampling") or {}).get("temperature", 0.0))
    if temperature > 0 and speculative.get("rejection_mode") != "standard":
        raise ValueError("sampling evaluation requires standard rejection sampling")
    return {
        "schema": "glm-vllm-ascend-comparison-v2",
        "baseline_tps": baseline_tps,
        "speculative_tps": speculative_tps,
        "speedup": speculative_tps / baseline_tps,
        "exact_token_id_match": exact,
        "exact_token_id_match_rate": sum(matches) / len(matches),
        # Compatibility aliases for existing table scripts.
        "exact_output_match": exact,
        "exact_output_match_rate": sum(matches) / len(matches),
        "mean_acceptance_length": spec_decode["mean_acceptance_length"],
        "draft_acceptance_rate": spec_decode["draft_acceptance_rate"],
        "drafts": spec_decode["drafts"],
    }
