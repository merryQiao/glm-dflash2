#!/usr/bin/env python
"""Teacher-force saved trajectories and archive Thinker hidden states.

Production uses vLLM's native ``extract_hidden_states`` path. Exact Stage-A
token IDs are supplied as the full prompt, so Stage B performs no sampling and
cannot silently retokenize the target trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from omni_sd.ascend_runtime import runtime_identity  # noqa: E402
from omni_sd.data_io import ParquetBatchWriter, atomic_write_json  # noqa: E402
from omni_sd.provenance import artifact_record, verify_artifact_record  # noqa: E402
from omni_sd.thinker_data import canonical_json, stable_hex  # noqa: E402
from omni_sd.thinker_generation import (  # noqa: E402
    DEFAULT_CONFIG,
    modality_batches,
    read_config,
    require_trajectories,
)
from omni_sd.vllm_ascend_generation import prepare_request  # noqa: E402
from omni_sd.vllm_ascend_hidden import (  # noqa: E402
    cleanup_connector_artifact,
    load_connector_tensors,
    load_engine,
    load_final_normalizer,
    native_connector_loader,
    response_loss_mask,
)


INDEX_SCHEMA = pa.schema(
    [
        ("condition_id", pa.string()),
        ("start", pa.int64()),
        ("end", pa.int64()),
        ("prompt_tokens", pa.int32()),
        ("response_tokens", pa.int32()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-shards", type=int)
    return parser.parse_args()


def cache_fingerprint(
    config: dict[str, Any], trajectory_manifest: dict[str, Any]
) -> str:
    return stable_hex(
        "thinker-vllm-ascend-hidden-v2",
        trajectory_manifest["generation_fingerprint"],
        config["model"],
        config["runtime"],
        config["hidden_states"],
    )


def _prompt_only_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = json.loads(str(row["messages_json"]))
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"trajectory {row['condition_id']} has no assistant response")
    result = dict(row)
    result["messages_json"] = canonical_json(messages[:-1])
    return result


def hidden_request(
    row: dict[str, Any], config: dict[str, Any], processor: Any
) -> tuple[dict[str, Any], list[int], int]:
    request, _ = prepare_request(_prompt_only_row(row), config, processor)
    prompt_ids = [int(value) for value in row["prompt_token_ids"]]
    response_ids = [int(value) for value in row["response_token_ids"]]
    sequence = prompt_ids + response_ids
    if not prompt_ids or not response_ids:
        raise ValueError(f"trajectory {row['condition_id']} has empty token IDs")
    if len(sequence) > int(config["generation"]["max_model_tokens"]):
        raise ValueError(f"trajectory {row['condition_id']} exceeds model context")
    request.pop("prompt", None)
    request["prompt_token_ids"] = sequence
    return request, sequence, len(prompt_ids)


def extract_batch(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    llm: Any,
    processor: Any,
    sampling_params_class: Any,
    normalizer: Any,
    connector_root: Path,
) -> list[dict[str, Any]]:
    prepared = [hidden_request(row, config, processor) for row in rows]
    paths = [
        connector_root
        / f"{stable_hex('hidden-request', row['condition_id'], digest_size=16)}.safetensors"
        for row in rows
    ]
    params = [
        sampling_params_class(
            max_tokens=1,
            temperature=0.0,
            extra_args={
                "kv_transfer_params": {
                    "hidden_states_path": str(path),
                    "include_output_tokens": False,
                }
            },
        )
        for path in paths
    ]
    outputs = llm.generate(
        [item[0] for item in prepared], sampling_params=params, use_tqdm=False
    )
    if len(outputs) != len(rows):
        raise RuntimeError(f"extractor returned {len(outputs)}/{len(rows)} requests")
    samples: list[dict[str, Any]] = []
    for row, (_, sequence, prompt_tokens), requested_path, output in zip(
        rows, prepared, paths, outputs, strict=True
    ):
        actual_path = Path(
            str((output.kv_transfer_params or {}).get("hidden_states_path", ""))
        )
        if actual_path != requested_path:
            raise RuntimeError(
                f"connector ignored trusted output path for {row['condition_id']}"
            )
        tensors = load_connector_tensors(
            actual_path,
            sequence,
            config,
            normalizer=normalizer,
            tensor_loader=native_connector_loader,
        )
        samples.append(
            {
                "condition_id": str(row["condition_id"]),
                "input_ids": tensors["token_ids"].to(dtype=torch.int32),
                "loss_mask": response_loss_mask(len(sequence), prompt_tokens),
                "target_hidden_states": tensors["hidden_states"].contiguous(),
                "target_last_hidden_states": tensors[
                    "final_hidden_states"
                ].contiguous(),
                "prompt_tokens": prompt_tokens,
                "response_tokens": len(sequence) - prompt_tokens,
            }
        )
        cleanup_connector_artifact(actual_path)
    return samples


class HiddenShardWriter:
    def __init__(
        self,
        output: Path,
        source_shard_id: int,
        max_shard_bytes: int,
        layer_ids: list[int],
        hidden_size: int,
    ) -> None:
        self.output = output
        self.source_shard_id = source_shard_id
        self.max_shard_bytes = max_shard_bytes
        self.layer_ids = layer_ids
        self.hidden_size = hidden_size
        self.samples: list[dict[str, Any]] = []
        self.buffered_bytes = 0
        self.part = 0
        self.files: list[dict[str, Any]] = []

    @staticmethod
    def sample_bytes(sample: dict[str, Any]) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in sample.values()
            if hasattr(value, "numel")
        )

    def append(self, sample: dict[str, Any]) -> None:
        size = self.sample_bytes(sample)
        if self.samples and self.buffered_bytes + size > self.max_shard_bytes:
            self.flush()
        self.samples.append(sample)
        self.buffered_bytes += size

    def flush(self) -> None:
        if not self.samples:
            return
        import torch
        from safetensors.torch import save_file

        stem = f"train-{self.source_shard_id:05d}-{self.part:03d}"
        tensor_path = self.output / "shards" / f"{stem}.safetensors"
        index_path = self.output / "shards" / f"{stem}.index.parquet"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        offsets = [0]
        rows = []
        for sample in self.samples:
            start = offsets[-1]
            end = start + int(sample["input_ids"].shape[0])
            offsets.append(end)
            rows.append(
                {
                    "condition_id": sample["condition_id"],
                    "start": start,
                    "end": end,
                    "prompt_tokens": sample["prompt_tokens"],
                    "response_tokens": sample["response_tokens"],
                }
            )
        tensors = {
            "offsets": torch.tensor(offsets, dtype=torch.int64),
            "input_ids": torch.cat([sample["input_ids"] for sample in self.samples]),
            "loss_mask": torch.cat([sample["loss_mask"] for sample in self.samples]),
            "target_hidden_states": torch.cat(
                [sample["target_hidden_states"] for sample in self.samples]
            ),
            "target_last_hidden_states": torch.cat(
                [sample["target_last_hidden_states"] for sample in self.samples]
            ),
        }
        temporary = tensor_path.with_name(f".{tensor_path.name}.{os.getpid()}.tmp")
        save_file(
            tensors,
            str(temporary),
            metadata={
                "target_layer_ids": json.dumps(self.layer_ids),
                "hidden_layout": "tokens,layers,hidden",
                "hidden_size": str(self.hidden_size),
                "final_hidden_semantics": "post_final_norm_lm_head_input",
            },
        )
        os.replace(temporary, tensor_path)
        with ParquetBatchWriter(index_path, INDEX_SCHEMA) as writer:
            writer.extend(rows)
        data_record = artifact_record(tensor_path, relative_to=self.output)
        index_record = artifact_record(index_path, relative_to=self.output)
        self.files.append(
            {
                "data": data_record,
                "index": index_record,
                "conditions": len(self.samples),
                "tokens": offsets[-1],
            }
        )
        self.samples.clear()
        self.buffered_bytes = 0
        self.part += 1

    def close(self) -> None:
        self.flush()


def completed_conditions(output: Path) -> int:
    if not (output / "shards").is_dir():
        return 0
    return sum(
        int(json.loads(path.read_text(encoding="utf-8"))["conditions"])
        for path in (output / "shards").glob("train-?????.success.json")
    )


def verify_cache(
    config: dict[str, Any], trajectory_manifest: dict[str, Any], fingerprint: str
) -> None:
    output = Path(config["hidden_states"]["output_root"])
    files: list[dict[str, Any]] = []
    conditions = 0
    tokens = 0
    seen: set[str] = set()
    runtime = None
    for source_shard_id in range(len(trajectory_manifest["shards"])):
        marker_path = output / "shards" / f"train-{source_shard_id:05d}.success.json"
        if not marker_path.is_file():
            raise FileNotFoundError(marker_path)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("cache_fingerprint") != fingerprint:
            raise ValueError(f"hidden shard {source_shard_id} belongs to another run")
        if runtime is None:
            runtime = marker.get("runtime")
        elif marker.get("runtime") != runtime:
            raise ValueError("hidden shards mix different runtime versions/topologies")
        for item in marker["files"]:
            verify_artifact_record(item["data"], root=output)
            index_path = verify_artifact_record(item["index"], root=output)
            for condition_id in pq.read_table(index_path, columns=["condition_id"])[
                "condition_id"
            ].to_pylist():
                if condition_id in seen:
                    raise ValueError(f"duplicate cached condition {condition_id}")
                seen.add(condition_id)
        conditions += int(marker["conditions"])
        tokens += int(marker["tokens"])
        files.extend(marker["files"])
    expected = int(trajectory_manifest["conditions"])
    if conditions != expected or len(seen) != expected:
        raise ValueError(f"hidden condition count {conditions}/{len(seen)} != {expected}")
    hidden = config["hidden_states"]
    atomic_write_json(
        output / "manifest.json",
        {
            "status": "PASS",
            "conditions": conditions,
            "tokens": tokens,
            "model": config["model"],
            "runtime": runtime,
            "dtype": hidden["dtype"],
            "hidden_size": int(hidden["hidden_size"]),
            "target_layer_ids": [int(value) for value in hidden["layer_ids"]],
            "hidden_layout": "tokens,layers,hidden",
            "final_hidden_semantics": "post_final_norm_lm_head_input",
            "cache_fingerprint": fingerprint,
            "trajectory_generation_fingerprint": trajectory_manifest[
                "generation_fingerprint"
            ],
            "files": files,
        },
    )
    print(f"verification PASS: {conditions:,} hidden trajectories", flush=True)


def run(config: dict[str, Any], max_shards: int | None) -> None:
    trajectory_root, trajectory_manifest = require_trajectories(config)
    output = Path(config["hidden_states"]["output_root"])
    connector_root = Path(config["hidden_states"]["scratch_root"])
    output.mkdir(parents=True, exist_ok=True)
    connector_root.mkdir(parents=True, exist_ok=True)
    fingerprint = cache_fingerprint(config, trajectory_manifest)
    worker = int(os.environ.get("WORKER_ID", "0"))
    workers = int(os.environ.get("NUM_WORKERS", "1"))
    if workers <= 0 or not 0 <= worker < workers:
        raise ValueError("require 0 <= WORKER_ID < NUM_WORKERS")
    identity = runtime_identity(hardware=str(config["runtime"]["hardware"]))
    if len(identity["visible_devices"]) != int(config["runtime"]["tensor_parallel_size"]):
        raise ValueError("visible Ascend device count must equal tensor_parallel_size")
    pending: list[int] = []
    for source_shard_id in range(len(trajectory_manifest["shards"])):
        if source_shard_id % workers != worker:
            continue
        marker_path = output / "shards" / f"train-{source_shard_id:05d}.success.json"
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("cache_fingerprint") != fingerprint:
                raise ValueError(f"hidden shard {source_shard_id} belongs to another run")
            if marker.get("runtime") != identity:
                raise ValueError(
                    f"hidden shard {source_shard_id} uses another runtime build/topology"
                )
            continue
        pending.append(source_shard_id)
        if max_shards is not None and len(pending) >= max_shards:
            break
    if not pending:
        print(f"worker={worker}: no pending hidden shards", flush=True)
        if completed_conditions(output) == int(trajectory_manifest["conditions"]):
            verify_cache(config, trajectory_manifest, fingerprint)
        return

    # Loading one vector before the engine is intentional: unsupported/missing
    # final-state semantics fail before allocating the 30B target.
    normalizer = load_final_normalizer(config)
    llm, processor, sampling_params_class = load_engine(config)
    layer_ids = [int(value) for value in config["hidden_states"]["layer_ids"]]
    hidden_size = int(config["hidden_states"]["hidden_size"])
    for source_shard_id in pending:
        source_record = trajectory_manifest["shards"][source_shard_id]
        source = verify_artifact_record(source_record, root=trajectory_root)
        rows = pq.read_table(source).to_pylist()
        started = time.perf_counter()
        writer = HiddenShardWriter(
            output,
            source_shard_id,
            int(config["hidden_states"]["max_shard_bytes"]),
            layer_ids,
            hidden_size,
        )
        for batch in modality_batches(rows, config["hidden_states"]["batch_size"]):
            for sample in extract_batch(
                batch,
                config,
                llm,
                processor,
                sampling_params_class,
                normalizer,
                connector_root,
            ):
                writer.append(sample)
        writer.close()
        conditions = sum(int(item["conditions"]) for item in writer.files)
        tokens = sum(int(item["tokens"]) for item in writer.files)
        if conditions != len(rows):
            raise RuntimeError(f"hidden shard wrote {conditions}/{len(rows)} rows")
        atomic_write_json(
            output / "shards" / f"train-{source_shard_id:05d}.success.json",
            {
                "status": "PASS",
                "source_shard_id": source_shard_id,
                "conditions": conditions,
                "tokens": tokens,
                "cache_fingerprint": fingerprint,
                "runtime": identity,
                "files": writer.files,
                "generation_seconds": time.perf_counter() - started,
            },
        )
        print(f"worker={worker}: committed hidden shard {source_shard_id}", flush=True)
    if completed_conditions(output) == int(trajectory_manifest["conditions"]):
        verify_cache(config, trajectory_manifest, fingerprint)


def main() -> None:
    args = parse_args()
    run(read_config(args.config), args.max_shards)


if __name__ == "__main__":
    main()
