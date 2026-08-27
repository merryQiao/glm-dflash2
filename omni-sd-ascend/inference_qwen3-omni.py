#!/usr/bin/env python
"""Profile the production Qwen3-Omni Thinker path on vLLM-Ascend.

This entry point deliberately profiles only the Thinker used by Stage A. It
does not load or report Talker/Code2Wav timings. Generation uses the exact same
engine, processor, request builder, and sampling provider as trajectory
generation, so its throughput is representative of that production path.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from omni_sd.ascend_runtime import runtime_identity  # noqa: E402
from omni_sd.data_io import atomic_write_json  # noqa: E402
from omni_sd.inference_profile import (  # noqa: E402
    ProfileContractError,
    aggregate_performance,
    normalize_input_record,
    profile_batches,
    request_latency_seconds,
)
from omni_sd.thinker_generation import (  # noqa: E402
    DEFAULT_CONFIG,
    read_config,
)
from omni_sd.vllm_ascend_generation import (  # noqa: E402
    completion_payload,
    engine_kwargs,
    load_engine,
    prepare_request,
    sampling_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--conditions-parquet", type=Path)
    parser.add_argument("--audio", type=Path, action="append", default=[])
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--video", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("outputs/qwen3_omni_thinker_profile.jsonl"),
    )
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ProfileContractError("--limit must be positive")
    if args.warmup < 0:
        raise ProfileContractError("--warmup must be non-negative")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ProfileContractError("--batch-size must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ProfileContractError("--max-new-tokens must be positive")
    if args.text is None and (args.audio or args.image or args.video):
        raise ProfileContractError("--audio/--image/--video require --text")


def _jsonl_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProfileContractError(f"{path}:{line_number} is not an object")
            rows.append(
                normalize_input_record(
                    value,
                    index=line_number - 1,
                    base_dir=path.resolve().parent,
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _parquet_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "condition_id",
        "source",
        "source_subset",
        "modality",
        "language",
        "task",
        "messages_json",
        "tools_json",
        "media_json",
    }
    parquet = pq.ParquetFile(path)
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise ProfileContractError(f"condition Parquet missing {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=256):
        rows.extend(batch.to_pylist())
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
    return rows


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.text is not None:
        row = normalize_input_record(
            {
                "id": "command-line",
                "text": args.text,
                "audio": [str(path) for path in args.audio],
                "image": [str(path) for path in args.image],
                "video": [str(path) for path in args.video],
            },
            index=0,
            base_dir=Path.cwd(),
        )
        return [row]
    if args.input_jsonl is not None:
        rows = _jsonl_rows(args.input_jsonl, args.limit)
    else:
        assert args.conditions_parquet is not None
        rows = _parquet_rows(args.conditions_parquet, args.limit)
    if not rows:
        raise ProfileContractError("input contains no inference records")
    return rows


def effective_config(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    if args.max_new_tokens is not None:
        config = deepcopy(config)
        config["generation"]["max_new_tokens"] = int(args.max_new_tokens)
    return config


def batch_sizes(config: dict[str, Any], override: int | None) -> dict[str, int]:
    if override is None:
        return {key: int(value) for key, value in config["vllm_batch_size"].items()}
    return {key: int(override) for key in config["vllm_batch_size"]}


def _generate(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    llm: Any,
    processor: Any,
    sampling_params_class: Any,
) -> tuple[list[Any], float]:
    prepared = [prepare_request(row, config, processor) for row in rows]
    params = [
        sampling_params_class(**sampling_kwargs(config, str(row["condition_id"])))
        for row in rows
    ]
    started = time.perf_counter()
    outputs = llm.generate(
        [request for request, _ in prepared],
        sampling_params=params,
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - started
    if elapsed <= 0 or len(outputs) != len(rows):
        raise RuntimeError(f"vLLM returned {len(outputs)}/{len(rows)} requests")
    return list(outputs), elapsed


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def dry_run_profile(
    config: dict[str, Any], rows: list[dict[str, Any]], sizes: dict[str, int]
) -> dict[str, Any]:
    first_id = str(rows[0]["condition_id"])
    return {
        "status": "DRY_RUN",
        "component": "thinker",
        "backend": "vllm_ascend",
        "requests": len(rows),
        "batch_sizes": sizes,
        "engine": engine_kwargs(config),
        "sampling": sampling_kwargs(config, first_id),
        "model": config["model"],
        "generation": config["generation"],
    }


def run_profile(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    sizes: dict[str, int],
    warmup: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identity = runtime_identity(hardware=str(config["runtime"]["hardware"]))
    expected_devices = int(config["runtime"]["tensor_parallel_size"])
    if len(identity["visible_devices"]) != expected_devices:
        raise RuntimeError("visible Ascend device count must equal tensor_parallel_size")

    load_started = time.perf_counter()
    llm, processor, sampling_params_class = load_engine(config)
    model_load_seconds = time.perf_counter() - load_started

    warmup_seconds = 0.0
    for _ in range(warmup):
        _, elapsed = _generate(
            [rows[0]], config, llm, processor, sampling_params_class
        )
        warmup_seconds += elapsed

    records: list[dict[str, Any]] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    request_seconds: list[float] = []
    measured_batches: list[float] = []
    eos = int(config["generation"]["eos_token_id"])
    for batch_index, batch in enumerate(profile_batches(rows, sizes)):
        outputs, elapsed = _generate(
            batch, config, llm, processor, sampling_params_class
        )
        measured_batches.append(elapsed)
        for row, output in zip(batch, outputs, strict=True):
            payload = completion_payload(output, eos)
            latency = request_latency_seconds(output, elapsed)
            prompt_tokens.append(int(payload["prompt_tokens"]))
            completion_tokens.append(int(payload["response_tokens"]))
            request_seconds.append(latency)
            records.append(
                {
                    "condition_id": str(row["condition_id"]),
                    "modality": str(row["modality"]),
                    "response_text": payload["response_text"],
                    "prompt_token_ids": payload["prompt_token_ids"],
                    "response_token_ids": payload["response_token_ids"],
                    "prompt_tokens": payload["prompt_tokens"],
                    "response_tokens": payload["response_tokens"],
                    "finish_reason": payload["finish_reason"],
                    "request_latency_ms": latency * 1000.0,
                    "batch_index": batch_index,
                    "batch_wall_ms": elapsed * 1000.0,
                }
            )

    performance = aggregate_performance(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        batch_seconds=measured_batches,
        request_seconds=request_seconds,
        model_load_seconds=model_load_seconds,
        warmup_seconds=warmup_seconds,
    )
    profile = {
        "status": "PASS",
        "component": "thinker",
        "backend": "vllm_ascend",
        "model": config["model"],
        "generation": config["generation"],
        "engine": engine_kwargs(config),
        "runtime": identity,
        "performance": performance,
    }
    return records, profile


def print_summary(profile: dict[str, Any]) -> None:
    if profile["status"] == "DRY_RUN":
        print(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True))
        return
    perf = profile["performance"]
    print("Qwen3-Omni Thinker performance (vLLM-Ascend)")
    print(f"  requests:                {perf['requests']}")
    print(f"  model load:              {perf['model_load_seconds']:.3f} s")
    print(f"  warmup:                  {perf['warmup_seconds']:.3f} s")
    print(f"  measured:                {perf['measured_seconds']:.3f} s")
    print(f"  completion throughput:   {perf['completion_tokens_per_second']:.3f} tok/s")
    print(f"  total token throughput:  {perf['total_tokens_per_second']:.3f} tok/s")
    print(f"  request throughput:      {perf['requests_per_second']:.3f} req/s")
    print(f"  request latency p50/p95: {perf['request_latency_ms']['p50']:.3f} / "
          f"{perf['request_latency_ms']['p95']:.3f} ms")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = effective_config(args)
    rows = load_rows(args)
    sizes = batch_sizes(config, args.batch_size)
    profile_path = args.profile_json or args.output_jsonl.with_suffix(".profile.json")
    targets = [profile_path] if args.dry_run else [args.output_jsonl, profile_path]
    if not args.overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(f"output already exists: {existing[0]}")

    if args.dry_run:
        profile = dry_run_profile(config, rows, sizes)
    else:
        records, profile = run_profile(config, rows, sizes, args.warmup)
        atomic_write_jsonl(args.output_jsonl, records)
    atomic_write_json(profile_path, profile)
    print_summary(profile)
    print(f"Profile: {profile_path.resolve()}")
    if not args.dry_run:
        print(f"Results: {args.output_jsonl.resolve()}")


if __name__ == "__main__":
    main()
