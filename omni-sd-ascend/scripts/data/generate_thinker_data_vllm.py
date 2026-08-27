#!/usr/bin/env python
"""Generate exact-token Qwen3-Omni Thinker trajectories with vLLM-Ascend.

One process owns one TP/EP engine. Independent replicas must use disjoint
``ASCEND_RT_VISIBLE_DEVICES`` plus ``WORKER_ID``/``NUM_WORKERS``; this program
must not be wrapped in torchrun.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from omni_sd.ascend_runtime import runtime_identity  # noqa: E402
from omni_sd.data_io import ParquetBatchWriter, atomic_write_json  # noqa: E402
from omni_sd.provenance import artifact_record  # noqa: E402
from omni_sd.thinker_data import canonical_json  # noqa: E402
from omni_sd.thinker_generation import (  # noqa: E402
    DEFAULT_CONFIG,
    TRAJECTORY_SCHEMA,
    completed_conditions,
    generation_fingerprint,
    modality_batches,
    read_config,
    require_verified_conditions,
    trajectory_root,
    verify_trajectories,
)
from omni_sd.vllm_ascend_generation import (  # noqa: E402
    completion_payload,
    condition_seed,
    load_engine,
    prepare_request,
    sampling_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-shards", type=int)
    return parser.parse_args()


def generate_batch(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    llm: Any,
    processor: Any,
    sampling_params_class: Any,
) -> list[dict[str, Any]]:
    prepared = [prepare_request(row, config, processor) for row in rows]
    params = [
        sampling_params_class(**sampling_kwargs(config, str(row["condition_id"])))
        for row in rows
    ]
    outputs = llm.generate(
        [item[0] for item in prepared], sampling_params=params, use_tqdm=False
    )
    if len(outputs) != len(rows):
        raise RuntimeError(f"vLLM returned {len(outputs)}/{len(rows)} requests")
    result: list[dict[str, Any]] = []
    eos = int(config["generation"]["eos_token_id"])
    for row, (_, conversation), output in zip(rows, prepared, outputs, strict=True):
        payload = completion_payload(output, eos)
        conversation.append({"role": "assistant", "content": payload["response_text"]})
        result.append(
            {
                "condition_id": str(row["condition_id"]),
                "source": str(row["source"]),
                "source_subset": str(row["source_subset"]),
                "modality": str(row["modality"]),
                "language": str(row["language"]),
                "task": str(row["task"]),
                "messages_json": canonical_json(conversation),
                "tools_json": str(row["tools_json"]),
                "media_json": str(row["media_json"]),
                **payload,
                "sampling_seed": condition_seed(config, str(row["condition_id"])),
            }
        )
    return result


def append_error(path: Path, row: dict[str, Any], error: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "condition_id": str(row["condition_id"]),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "time": time.time(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def generate_with_recovery(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    llm: Any,
    processor: Any,
    sampling_params_class: Any,
    errors: Path,
) -> list[dict[str, Any]]:
    try:
        return generate_batch(rows, config, llm, processor, sampling_params_class)
    except Exception:
        if len(rows) == 1:
            raise
    recovered: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for row in rows:
        try:
            recovered.extend(
                generate_batch([row], config, llm, processor, sampling_params_class)
            )
        except Exception as error:
            append_error(errors, row, error)
            failures.append((str(row["condition_id"]), str(error)))
    if failures:
        raise RuntimeError(f"{len(failures)} Thinker request(s) failed: {failures[:3]}")
    return recovered


def process_shard(
    shard_id: int,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
    llm: Any,
    processor: Any,
    sampling_params_class: Any,
    identity: dict[str, Any],
) -> None:
    parquet = output / "shards" / f"train-{shard_id:05d}.parquet"
    errors = output / "errors" / f"train-{shard_id:05d}.jsonl"
    started = time.perf_counter()
    with ParquetBatchWriter(parquet, TRAJECTORY_SCHEMA) as writer:
        for batch in modality_batches(rows, config["vllm_batch_size"]):
            writer.extend(
                generate_with_recovery(
                    batch, config, llm, processor, sampling_params_class, errors
                )
            )
    atomic_write_json(
        output / "shards" / f"train-{shard_id:05d}.success.json",
        {
            "status": "PASS",
            "backend": "vllm_ascend",
            "shard_id": shard_id,
            "conditions": len(rows),
            "generation_fingerprint": generation_fingerprint(config),
            "parquet_artifact": artifact_record(parquet, relative_to=output),
            "runtime": identity,
            "generation_seconds": time.perf_counter() - started,
        },
    )


def run(config: dict[str, Any], max_shards: int | None) -> None:
    input_path, total = require_verified_conditions(config)
    output = trajectory_root(config)
    output.mkdir(parents=True, exist_ok=True)
    worker = int(os.environ.get("WORKER_ID", "0"))
    workers = int(os.environ.get("NUM_WORKERS", "1"))
    if workers <= 0 or not 0 <= worker < workers:
        raise ValueError("require 0 <= WORKER_ID < NUM_WORKERS")
    identity = runtime_identity(hardware=str(config["runtime"]["hardware"]))
    if len(identity["visible_devices"]) != int(config["runtime"]["tensor_parallel_size"]):
        raise ValueError("visible Ascend device count must equal tensor_parallel_size")
    shard_rows = int(config["output"]["conditions_per_shard"])
    count = math.ceil(total / shard_rows)
    pending: list[int] = []
    for shard_id in range(count):
        if shard_id % workers != worker:
            continue
        marker_path = output / "shards" / f"train-{shard_id:05d}.success.json"
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("generation_fingerprint") != generation_fingerprint(config):
                raise ValueError(f"shard {shard_id} belongs to another run")
            if marker.get("runtime") != identity:
                raise ValueError(f"shard {shard_id} uses another runtime build/topology")
            continue
        pending.append(shard_id)
        if max_shards is not None and len(pending) >= max_shards:
            break
    if not pending:
        print(f"worker={worker}: no pending trajectory shards", flush=True)
        if completed_conditions(output) == total:
            verify_trajectories(config)
        return
    print(
        f"host={socket.gethostname()} worker={worker}/{workers} loading "
        f"vLLM-Ascend for {len(pending)} shard(s)",
        flush=True,
    )
    llm, processor, sampling_params_class = load_engine(config)
    wanted = set(pending)
    for shard_id, batch in enumerate(
        pq.ParquetFile(input_path).iter_batches(batch_size=shard_rows)
    ):
        if shard_id not in wanted:
            continue
        rows = batch.to_pylist()
        process_shard(
            shard_id, rows, config, output, llm, processor,
            sampling_params_class, identity,
        )
        print(f"worker={worker}: committed trajectory shard {shard_id}", flush=True)
    if completed_conditions(output) == total:
        verify_trajectories(config)


def main() -> None:
    args = parse_args()
    run(read_config(args.config), args.max_shards)


if __name__ == "__main__":
    main()
