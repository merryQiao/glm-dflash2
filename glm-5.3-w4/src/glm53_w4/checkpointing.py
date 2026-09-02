from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

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


def _world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _atomic_save(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-r{_rank()}-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    root: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    progress: TrainingProgress,
    semantic_config: Mapping[str, Any],
) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "COMPLETE").exists():
        raise FileExistsError(root)
    if _world() > 1:
        from torch.distributed.checkpoint import save as dcp_save
        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_state, optimizer_state = get_state_dict(model, optimizer)
        dcp_save(
            {"model": model_state, "optimizer": optimizer_state},
            checkpoint_id=root / "sharded",
        )
    elif _rank() == 0:
        _atomic_save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            root / "state.pt",
        )
    if _rank() == 0:
        metadata = {
            "schema": "glm53-w4-stage-c-checkpoint-v2",
            "world_size": _world(),
            "progress": asdict(progress),
            "scheduler": scheduler.state_dict(),
            "semantic_config": dict(semantic_config),
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if dist.is_initialized():
        dist.barrier()
    if _rank() == 0:
        (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(
    root: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    semantic_config: Mapping[str, Any],
) -> TrainingProgress:
    root = Path(root)
    if not (root / "COMPLETE").is_file():
        raise ValueError(f"incomplete checkpoint: {root}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "glm53-w4-stage-c-checkpoint-v2":
        raise ValueError("unsupported checkpoint schema")
    if int(metadata["world_size"]) != _world():
        raise ValueError("checkpoint world size differs")
    if metadata.get("semantic_config") != dict(semantic_config):
        raise ValueError("checkpoint training semantics differ")
    if _world() > 1:
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
    scheduler.load_state_dict(metadata["scheduler"])
    return TrainingProgress(**metadata["progress"])
