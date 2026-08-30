#!/usr/bin/env python3
"""Teacher-force frozen GLM-5.3 Stage A IDs through an Ascend SGLang runner."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm53_drafters.contracts import TARGET_CONTRACT  # noqa: E402
from glm53_drafters.hidden_cache import HiddenCacheSpec, PackedHiddenWriter  # noqa: E402
from glm53_drafters.hidden_extraction import read_frozen_trajectories  # noqa: E402
from glm53_drafters.sglang_hidden_runner import (  # noqa: E402
    SGLangInternalHiddenRunner,
    validate_stage_b_server_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--trajectory-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-layer-ids", default="1,11,22,32,42")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--attention-backend", default="ascend")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--model-runner", default="torch")
    parser.add_argument("--tp-size", type=int, default=16)
    parser.add_argument("--ep-size", type=int)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--dist-init-addr", default="127.0.0.1")
    parser.add_argument("--dist-port", type=int, default=29500)
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--max-segment-gib", type=float, default=64.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--allow-smoke-unverified", action="store_true")
    parser.add_argument("--smoke-max-samples", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quantization")
    parser.add_argument("--load-format")
    parser.add_argument("--moe-a2a-backend")
    parser.add_argument("--deepep-mode")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _server_contract(cli: argparse.Namespace) -> SimpleNamespace:
    values = vars(cli).copy()
    values.update(
        {
            "model_path": str(cli.model_path),
            "ep_size": cli.ep_size or cli.tp_size,
            "chunked_prefill_size": -1,
            "disable_radix_cache": True,
            "disable_cuda_graph": True,
            "max_running_requests": 1,
            "dist_init_addr": f"{cli.dist_init_addr}:{cli.dist_port}",
        }
    )
    return SimpleNamespace(**values)


def _sglang_server_args(contract: SimpleNamespace) -> tuple[Any, Any]:
    """Create version-specific SGLang objects only after CLI validation."""

    from sglang.srt.server_args import PortArgs, ServerArgs

    server_args = ServerArgs(model_path=contract.model_path)
    for name, value in vars(contract).items():
        if hasattr(server_args, name):
            setattr(server_args, name, value)
    if hasattr(server_args, "resolve_once"):
        server_args.resolve_once()
    validate_stage_b_server_args(server_args)
    return server_args, PortArgs.init_new(server_args)


def _done_ids(output_dir: Path) -> set[str]:
    path = output_dir / "index.jsonl"
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["sample_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _next_source_control(
    rows: Any,
    *,
    done_ids: set[str],
    committed: int,
    max_samples: int | None,
) -> dict[str, Any]:
    """Produce the sole rank-0 decision that every TP rank must follow."""

    if max_samples is not None and committed >= max_samples:
        return {"action": "stop", "reason": "max_samples"}
    row = next(rows, None)
    if row is None:
        return {"action": "stop", "reason": "eof"}
    sample_id = str(row["id"])
    if sample_id in done_ids:
        return {"action": "skip", "id": sample_id}
    return {
        "action": "process",
        "id": sample_id,
        "input_ids": row["input_ids"],
        "loss_mask": row["loss_mask"],
        "generation_route": row.get("generation_route"),
        "source_metadata": row.get("source_metadata", {}),
        "token_contract": row.get("token_contract", {}),
    }


def _worker(
    contract: SimpleNamespace,
    gpu_id: int,
    tp_rank: int,
    layer_ids: tuple[int, ...],
    target_identity: dict[str, Any],
) -> None:
    import torch.distributed as dist

    server_args, port_args = _sglang_server_args(contract)
    runner = SGLangInternalHiddenRunner(
        server_args=server_args,
        port_args=port_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        logical_layer_ids=layer_ids,
    )
    trajectory = Path(contract.trajectory_jsonl)
    output = Path(contract.output_dir)
    source_manifest = json.loads(
        trajectory.with_suffix(trajectory.suffix + ".manifest.json").read_text(
            encoding="utf-8"
        )
    )
    is_smoke = source_manifest.get("status") == "smoke_unverified"
    ascend_a2_attestation = getattr(runner, "ascend_a2_attestation", None)
    if not is_smoke and ascend_a2_attestation is None:
        evidence = runner.backend_metadata.get("ascend_a2_runtime", {})
        raise RuntimeError(
            "production Stage B requires a live Ascend 910B A2 runtime "
            f"attestation: {evidence.get('reason', 'probe did not pass')}"
        )
    provenance = {
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": _sha256(trajectory),
        "trajectory_manifest_sha256": _sha256(
            trajectory.with_suffix(trajectory.suffix + ".manifest.json")
        ),
        "logical_layer_ids": list(layer_ids),
        "physical_layer_ids": list(runner.physical_layer_ids),
        "capture_mapping": [list(tap.as_tuple()) for tap in runner.capture_mapping],
        "backend": dict(runner.backend_metadata),
        "model_fingerprint": source_manifest["model_fingerprint"],
        "model_revision": source_manifest["model_revision"],
        "tokenizer_fingerprint": source_manifest["tokenizer_fingerprint"],
        "vocab_size": source_manifest["vocab_size"],
        "target_hidden_dtype": "bfloat16",
        "source_status": source_manifest.get("status"),
        "production_eligible": not is_smoke,
    }
    if ascend_a2_attestation is not None:
        provenance["ascend_a2_runtime"] = dict(ascend_a2_attestation)
    writer_context = (
        PackedHiddenWriter(
            output,
            spec=HiddenCacheSpec(
                layer_ids=layer_ids,
                hidden_size=runner.hidden_size,
                capture_mapping=tuple(tap.as_tuple() for tap in runner.capture_mapping),
            ),
            provenance=provenance,
            ascend_a2_attestation=ascend_a2_attestation,
            max_segment_bytes=int(contract.max_segment_gib * (1 << 30)),
        )
        if tp_rank == 0
        else nullcontext(None)
    )
    done = _done_ids(output) if tp_rank == 0 else set()
    committed = 0
    reached_eof = False
    with writer_context as writer:
        rows = (
            iter(
                read_frozen_trajectories(
                    trajectory,
                    allow_smoke_unverified=contract.allow_smoke_unverified,
                    smoke_max_samples=contract.smoke_max_samples,
                    expected_target_identity=target_identity,
                )
            )
            if tp_rank == 0
            else None
        )
        while True:
            control = (
                _next_source_control(
                    rows,
                    done_ids=done,
                    committed=committed,
                    max_samples=contract.max_samples,
                )
                if tp_rank == 0
                else None
            )
            if dist.is_initialized():
                payload = [control]
                dist.broadcast_object_list(payload, src=0)
                control = payload[0]
            elif tp_rank != 0:
                raise RuntimeError("multi-rank Stage B runner has no process group")
            action = control["action"]
            if action == "stop":
                reached_eof = control.get("reason") == "eof"
                break
            if action == "skip":
                continue
            sample_id = str(control["id"])
            capture = None
            capture_error = None
            try:
                capture = runner.extract(control["input_ids"])
            except BaseException as exc:
                capture_error = f"{type(exc).__name__}: {exc}"
            errors = [capture_error]
            if dist.is_initialized():
                errors = [None] * dist.get_world_size()
                dist.all_gather_object(errors, capture_error)
            failures = [
                f"rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            ]
            if failures:
                raise RuntimeError("SGLang hidden capture failed; " + "; ".join(failures))
            if capture is None:
                raise RuntimeError("capture failed without a synchronized error")
            write_error = None
            if tp_rank == 0:
                try:
                    writer.append(
                        sample_id=sample_id,
                        source_index=int(
                            control.get("source_metadata", {}).get(
                                "selected_source_index", -1
                            )
                        ),
                        input_ids=control["input_ids"],
                        loss_mask=control["loss_mask"],
                        aux_hidden_states=capture.aux_hidden_states,
                        target_final_hidden=capture.target_final_hidden,
                        attestation=capture.attestation,
                        metadata={
                            "generation_route": control.get("generation_route"),
                            "source_metadata": control.get("source_metadata", {}),
                            "token_contract": control.get("token_contract", {}),
                        },
                    )
                    done.add(sample_id)
                except BaseException as exc:
                    write_error = f"{type(exc).__name__}: {exc}"
            if dist.is_initialized():
                payload = [write_error]
                dist.broadcast_object_list(payload, src=0)
                write_error = payload[0]
            if write_error is not None:
                raise RuntimeError(
                    f"rank-0 cache commit failed for {sample_id}: {write_error}"
                )
            committed += 1
            if dist.is_initialized():
                dist.barrier()
        if tp_rank == 0 and reached_eof:
            writer.freeze()
    if contract.tp_size > 1:
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()


def main() -> int:
    cli = parse_args()
    contract = _server_contract(cli)
    validate_stage_b_server_args(contract)
    if contract.tp_size < 1 or contract.nnodes < 1 or contract.tp_size % contract.nnodes:
        raise ValueError("tp_size must be divisible by a positive nnodes")
    if not 0 <= contract.node_rank < contract.nnodes:
        raise ValueError("node_rank must be in [0, nnodes)")
    layer_ids = tuple(
        int(value) for value in cli.capture_layer_ids.split(",") if value.strip()
    )
    if layer_ids != TARGET_CONTRACT.logical_layer_ids:
        raise ValueError(
            f"capture layers must be {TARGET_CONTRACT.logical_layer_ids}, got {layer_ids}"
        )
    from glm53_drafters.target_io import local_model_identity

    target_identity = local_model_identity(cli.model_path)
    first = next(
        read_frozen_trajectories(
            cli.trajectory_jsonl,
            allow_smoke_unverified=cli.allow_smoke_unverified,
            smoke_max_samples=cli.smoke_max_samples,
            expected_target_identity=target_identity,
        ),
        None,
    )
    if first is None:
        raise ValueError("Stage A trajectory artifact is empty")
    source_manifest = json.loads(
        cli.trajectory_jsonl.with_suffix(
            cli.trajectory_jsonl.suffix + ".manifest.json"
        ).read_text(encoding="utf-8")
    )
    for key in (
        "model_fingerprint",
        "model_revision",
        "tokenizer_fingerprint",
        "vocab_size",
    ):
        if source_manifest.get(key) != target_identity[key]:
            raise ValueError(f"Stage B {key} differs from Stage A")
    ranks_per_node = contract.tp_size // contract.nnodes
    first_rank = contract.node_rank * ranks_per_node
    if contract.tp_size == 1:
        _worker(contract, 0, 0, layer_ids, target_identity)
        return 0
    processes = []
    for local_rank, tp_rank in enumerate(range(first_rank, first_rank + ranks_per_node)):
        process = multiprocessing.Process(
            target=_worker,
            args=(contract, local_rank, tp_rank, layer_ids, target_identity),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"Stage B worker exited with code {process.exitcode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
