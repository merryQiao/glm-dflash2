from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


SCHEMA = "omni-stage-c-checkpoint-v1"


@dataclass(frozen=True)
class TrainingProgress:
    global_step: int
    epoch: int
    cursor: int


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _atomic_save(value: Any, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.rank{_rank()}.{os.getpid()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def save_checkpoint(root: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Any, progress: TrainingProgress,
                    semantics: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if (root / "COMPLETE").exists():
        raise FileExistsError(root)
    if _world() > 1:
        from torch.distributed.checkpoint import save
        from torch.distributed.checkpoint.state_dict import get_state_dict
        model_state, optimizer_state = get_state_dict(model, optimizer)
        save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=root / "sharded")
    elif _rank() == 0:
        _atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, root / "state.pt")
    if _rank() == 0:
        metadata = {"schema": SCHEMA, "world_size": _world(), "progress": asdict(progress),
                    "scheduler": scheduler.state_dict(), "semantics": semantics}
        tmp = root / ".metadata.tmp"
        tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, root / "metadata.json")
    if dist.is_initialized():
        dist.barrier()
    if _rank() == 0:
        (root / "COMPLETE").write_text("complete\n")
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(root: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Any, semantics: dict[str, Any]) -> TrainingProgress:
    if not (root / "COMPLETE").is_file():
        raise ValueError(f"incomplete checkpoint: {root}")
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("schema") != SCHEMA or metadata.get("semantics") != semantics:
        raise ValueError("checkpoint schema/semantics mismatch")
    if int(metadata["world_size"]) != _world():
        raise ValueError("checkpoint world size mismatch")
    if _world() > 1:
        from torch.distributed.checkpoint import load
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
        model_state, optimizer_state = get_state_dict(model, optimizer)
        state = {"model": model_state, "optimizer": optimizer_state}
        load(state, checkpoint_id=root / "sharded")
        set_state_dict(model, optimizer, model_state_dict=state["model"],
                       optim_state_dict=state["optimizer"])
    else:
        state = torch.load(root / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(metadata["scheduler"])
    return TrainingProgress(**metadata["progress"])
