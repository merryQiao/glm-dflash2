#!/usr/bin/env python3
"""Stage B: teacher-force frozen trajectories and cache GLM-5.2 layer states."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm_dflash2.hidden_cache import HiddenCacheSpec, PackedHiddenWriter  # noqa: E402
from glm_dflash2.hidden_extraction import (  # noqa: E402
    estimate_packed_cache_bytes,
    read_frozen_trajectories,
)
from glm_dflash2.sglang_hidden_runner import (  # noqa: E402
    SGLangInternalHiddenRunner,
    one_batch_module,
)
from glm_dflash2.provenance import local_model_fingerprint  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _done_ids(output_dir: Path) -> set[str]:
    path = output_dir / "index.jsonl"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {str(json.loads(line)["sample_id"]) for line in handle if line.strip()}


def _worker(
    server_args: Any,
    port_args: Any,
    gpu_id: int,
    tp_rank: int,
    trajectory_path: str,
    output_dir: str,
    layer_ids: tuple[int, ...],
    max_segment_bytes: int,
    max_samples: int | None,
    model_fingerprint: str,
    allow_partial_trajectories: bool,
    source_is_frozen: bool,
) -> None:
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from sglang.srt.utils import configure_logger

    one_batch = one_batch_module()
    one_batch.initialize_moe_config(server_args)
    one_batch.initialize_fp8_gemm_config(server_args)
    one_batch.initialize_fp4_gemm_config(server_args)
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    runner = SGLangInternalHiddenRunner(
        server_args=server_args,
        port_args=port_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        logical_layer_ids=layer_ids,
    )
    trajectory = Path(trajectory_path)
    output = Path(output_dir)
    # Rank 0 is the single source of truth for resume/skip decisions.  Other
    # TP ranks must not inspect a potentially lagging shared-filesystem view.
    done = _done_ids(output) if tp_rank == 0 else set()
    provenance = {
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": _sha256(trajectory),
        "logical_layer_ids": list(layer_ids),
        "physical_layer_ids": list(runner.physical_layer_ids),
        "capture_mapping": [tap.as_tuple() for tap in runner.capture_mapping],
        "backend": dict(runner.backend_metadata),
        "topology": {
            "tp_size": int(server_args.tp_size),
            "ep_size": int(server_args.ep_size),
            "nnodes": int(server_args.nnodes),
            "node_rank": int(server_args.node_rank),
            "dist_init_addr": str(getattr(server_args, "dist_init_addr", "")),
            "nccl_port": int(getattr(port_args, "nccl_port", 0)),
        },
        "model_path": str(Path(server_args.model_path).resolve()),
        "model_fingerprint": model_fingerprint,
    }
    writer_context = (
        PackedHiddenWriter(
            output,
            spec=HiddenCacheSpec(
                layer_ids=layer_ids,
                hidden_size=runner.hidden_size,
                capture_mapping=tuple(tap.as_tuple() for tap in runner.capture_mapping),
            ),
            max_segment_bytes=max_segment_bytes,
            provenance=provenance,
        )
        if tp_rank == 0
        else nullcontext(None)
    )
    committed = 0
    reached_eof = False
    with writer_context as writer:
        source_iterator = (
            iter(
                read_frozen_trajectories(
                    trajectory, allow_partial=allow_partial_trajectories
                )
            )
            if tp_rank == 0
            else None
        )
        while True:
            if tp_rank == 0:
                source_row = next(source_iterator, None)
                if source_row is None:
                    control = {"action": "stop", "reason": "eof"}
                elif str(source_row["id"]) in done:
                    control = {"action": "skip", "id": str(source_row["id"])}
                elif max_samples is not None and committed >= max_samples:
                    control = {"action": "stop", "reason": "max_samples"}
                else:
                    control = {
                        "action": "process",
                        "id": source_row["id"],
                        "input_ids": source_row["input_ids"],
                        "loss_mask": source_row["loss_mask"],
                        "source_metadata": source_row.get("source_metadata", {}),
                        "token_contract": source_row.get("token_contract", {}),
                    }
            else:
                control = None
            if dist.is_initialized():
                payload = [control]
                dist.broadcast_object_list(payload, src=0)
                control = payload[0]
            action = control["action"]
            if action == "stop":
                reached_eof = control.get("reason") == "eof"
                break
            if action == "skip":
                continue
            row = control
            sample_id = str(row["id"])
            capture = runner.extract(row["input_ids"])
            write_error = None
            if tp_rank == 0:
                try:
                    source_index = int(
                        row.get("source_metadata", {}).get("selected_source_index", -1)
                    )
                    writer.append(
                        sample_id=sample_id,
                        source_index=source_index,
                        input_ids=row["input_ids"],
                        loss_mask=row["loss_mask"],
                        aux_hidden_states=capture.aux_hidden_states,
                        target_final_hidden=capture.target_final_hidden,
                        metadata={
                            "source_metadata": row.get("source_metadata", {}),
                            "token_contract": row.get("token_contract", {}),
                        },
                    )
                    done.add(sample_id)
                    print(
                        json.dumps(
                            {"sample_id": sample_id, "tokens": len(row["input_ids"])},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                except BaseException as exc:  # keep all TP ranks on one control path
                    write_error = f"{type(exc).__name__}: {exc}"
            if dist.is_initialized():
                status = [write_error]
                dist.broadcast_object_list(status, src=0)
                write_error = status[0]
            if write_error is not None:
                raise RuntimeError(
                    f"rank-0 cache commit failed for {sample_id}: {write_error}"
                )
            committed += 1
            if dist.is_initialized():
                dist.barrier()
        if tp_rank == 0 and reached_eof and source_is_frozen:
            writer.freeze()
    if server_args.tp_size > 1:
        destroy_model_parallel()
        destroy_distributed_environment()


def main() -> int:
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import kill_process_tree, maybe_reindex_device_id

    parser = argparse.ArgumentParser(description=__doc__)
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--trajectory-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--capture-layer-ids", default="1,20,38,56,75")
    parser.add_argument("--max-segment-gib", type=float, default=64.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--allow-partial-trajectories",
        action="store_true",
        help="Allow a Stage-A partial manifest for a small hardware gate.",
    )
    parser.add_argument("--storage-safety-factor", type=float, default=1.05)
    parser.add_argument("--skip-storage-preflight", action="store_true")
    cli = parser.parse_args()
    server_args = ServerArgs.from_cli_args(cli)
    if server_args.nnodes < 1 or server_args.tp_size % server_args.nnodes != 0:
        raise ValueError("tp_size must be divisible by a positive nnodes")
    if not 0 <= server_args.node_rank < server_args.nnodes:
        raise ValueError("node_rank must be in [0, nnodes)")
    if server_args.dp_size != 1:
        raise ValueError("Stage B currently supports DP=1 only")
    if server_args.pp_size != 1:
        raise ValueError("Stage B currently supports PP=1 only")
    server_args.disable_cuda_graph = True
    server_args.disable_radix_cache = True
    server_args.chunked_prefill_size = -1
    server_args.max_running_requests = 1
    if hasattr(server_args, "resolve_once"):
        server_args.resolve_once()
    _set_envs_and_config(server_args)
    layer_ids = tuple(int(value) for value in cli.capture_layer_ids.split(",") if value)
    if not layer_ids:
        raise ValueError("--capture-layer-ids is empty")
    trajectory_path = Path(cli.trajectory_jsonl)
    # Refuse an unfinished Stage A cache before allocating the model.
    if next(
        read_frozen_trajectories(
            trajectory_path, allow_partial=cli.allow_partial_trajectories
        ),
        None,
    ) is None:
        raise ValueError("frozen trajectory JSONL is empty")
    port_args = PortArgs.init_new(server_args)
    model_fingerprint = local_model_fingerprint(Path(server_args.model_path))
    trajectory_manifest_path = trajectory_path.with_suffix(
        trajectory_path.suffix + ".manifest.json"
    )
    trajectory_manifest = json.loads(trajectory_manifest_path.read_text(encoding="utf-8"))
    source_is_frozen = trajectory_manifest.get("status") == "frozen"
    source_model_fingerprint = trajectory_manifest.get("model_fingerprint")
    if source_model_fingerprint and source_model_fingerprint != model_fingerprint:
        raise ValueError(
            "Stage B model fingerprint differs from the model used by Stage A"
        )
    if not cli.skip_storage_preflight:
        done = _done_ids(Path(cli.output_dir))
        remaining_tokens = 0
        remaining_samples = 0
        for row in read_frozen_trajectories(
            trajectory_path, allow_partial=cli.allow_partial_trajectories
        ):
            if str(row["id"]) in done:
                continue
            if cli.max_samples is not None and remaining_samples >= cli.max_samples:
                break
            remaining_tokens += len(row["input_ids"])
            remaining_samples += 1
        required = estimate_packed_cache_bytes(
            remaining_tokens, num_layers=len(layer_ids), hidden_size=6144
        )
        target = Path(cli.output_dir).resolve()
        probe = target
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        if free < int(required * cli.storage_safety_factor):
            raise RuntimeError(
                f"insufficient cache storage: need about {required} bytes before "
                f"safety factor, free={free} at {probe}"
            )
    ranks_per_node = server_args.tp_size // server_args.nnodes
    local_start = server_args.node_rank * ranks_per_node
    local_end = local_start + ranks_per_node
    max_segment_bytes = int(cli.max_segment_gib * (1 << 30))
    logging.basicConfig(level=getattr(logging, server_args.log_level.upper()))
    try:
        if server_args.tp_size == 1:
            _worker(
                server_args,
                port_args,
                0,
                0,
                str(trajectory_path),
                cli.output_dir,
                layer_ids,
                max_segment_bytes,
                cli.max_samples,
                model_fingerprint,
                cli.allow_partial_trajectories,
                source_is_frozen,
            )
        else:
            workers = []
            for tp_rank in range(local_start, local_end):
                with maybe_reindex_device_id(tp_rank - local_start) as gpu_id:
                    process = multiprocessing.Process(
                        target=_worker,
                        args=(
                            server_args,
                            port_args,
                            gpu_id,
                            tp_rank,
                            str(trajectory_path),
                            cli.output_dir,
                            layer_ids,
                            max_segment_bytes,
                            cli.max_samples,
                            model_fingerprint,
                            cli.allow_partial_trajectories,
                            source_is_frozen,
                        ),
                    )
                    process.start()
                    workers.append(process)
            failures = []
            for process in workers:
                process.join()
                if process.exitcode:
                    failures.append(process.exitcode)
            if failures:
                raise RuntimeError(f"SGLang hidden workers failed: {failures}")
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
