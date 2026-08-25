from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class TrainingProgress:
    global_step: int
    micro_step: int
    epoch: int
    sample_cursor: int


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-rank{_rank():05d}-{os.getpid()}")


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = _temporary_path(path)
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = _temporary_path(path)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _fsync_tree(root: Path) -> None:
    """Durably commit a torch.distributed.checkpoint directory tree."""

    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_dir(path)
    _fsync_dir(root)


def _runtime_state(anchor_generator: torch.Generator) -> dict[str, Any]:
    value: dict[str, Any] = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "anchor_rng": anchor_generator.get_state(),
    }
    if torch.cuda.is_available():
        value["cuda_rng"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "npu") and torch.npu.is_available():
        value["npu_rng"] = torch.npu.get_rng_state_all()
    return value


def _restore_runtime_state(value: dict[str, Any], anchor_generator: torch.Generator) -> None:
    random.setstate(value["python_rng"])
    np.random.set_state(value["numpy_rng"])
    torch.set_rng_state(value["torch_rng"])
    anchor_generator.set_state(value["anchor_rng"])
    if "cuda_rng" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["cuda_rng"])
    if "npu_rng" in value and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_rng_state_all(value["npu_rng"])


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    progress: TrainingProgress,
    anchor_generator: torch.Generator,
    at_optimizer_boundary: bool,
    semantic_config: dict[str, Any] | None = None,
) -> None:
    if not at_optimizer_boundary:
        raise ValueError("checkpoints are only valid at an optimizer-step boundary")
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "COMPLETE").exists():
        raise FileExistsError(f"refusing to overwrite complete checkpoint {root}")
    multi_rank = _world_size() > 1
    if multi_rank:
        from torch.distributed.checkpoint import save as dcp_save
        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_state, optimizer_state = get_state_dict(model, optimizer)
        dcp_save(
            {"model": model_state, "optimizer": optimizer_state},
            checkpoint_id=root / "sharded",
        )
        _fsync_tree(root / "sharded")
    elif _rank() == 0:
        _atomic_torch_save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            root / "state.pt",
        )
    _atomic_torch_save(
        _runtime_state(anchor_generator), root / f"runtime-rank{_rank():05d}.pt"
    )
    if _rank() == 0:
        metadata = {
            "schema": "glm-dflash2-training-checkpoint-v1",
            "world_size": _world_size(),
            "progress": asdict(progress),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "semantic_config": semantic_config,
        }
        _atomic_write_text(root / "metadata.json", json.dumps(metadata, indent=2) + "\n")
    _barrier()
    if _rank() == 0:
        _atomic_write_text(root / "COMPLETE", "complete\n")
    _barrier()


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    anchor_generator: torch.Generator,
    expected_semantic_config: dict[str, Any] | None = None,
) -> TrainingProgress:
    root = Path(path)
    if not (root / "COMPLETE").is_file():
        raise ValueError(f"checkpoint is incomplete: {root}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "glm-dflash2-training-checkpoint-v1":
        raise ValueError("unsupported checkpoint schema")
    if int(metadata["world_size"]) != _world_size():
        raise ValueError("checkpoint world size differs from the current run")
    if expected_semantic_config is not None:
        recorded = metadata.get("semantic_config")
        if recorded != expected_semantic_config:
            raise ValueError(
                "checkpoint semantic configuration differs from the current run: "
                f"recorded={recorded!r}, current={expected_semantic_config!r}"
            )
    if _world_size() > 1:
        from torch.distributed.checkpoint import load as dcp_load
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        model_state, optimizer_state = get_state_dict(model, optimizer)
        state = {"model": model_state, "optimizer": optimizer_state}
        dcp_load(state, checkpoint_id=root / "sharded")
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )
    else:
        state = torch.load(root / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and metadata.get("scheduler") is not None:
        scheduler.load_state_dict(metadata["scheduler"])
    runtime = torch.load(
        root / f"runtime-rank{_rank():05d}.pt", map_location="cpu", weights_only=False
    )
    _restore_runtime_state(runtime, anchor_generator)
    return TrainingProgress(**{key: int(value) for key, value in metadata["progress"].items()})
