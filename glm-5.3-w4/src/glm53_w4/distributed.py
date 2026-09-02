from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class DistributedContext:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(device_kind: str) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if device_kind == "cpu":
        device, backend = torch.device("cpu"), "gloo"
    elif device_kind == "npu":
        importlib.import_module("torch_npu")
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("torch_npu is installed but no Ascend NPU is available")
        torch.npu.set_device(local_rank)
        device, backend = torch.device("npu", local_rank), "hccl"
    else:
        raise ValueError("device must be npu or cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=60),
        )
    return DistributedContext(device, rank, local_rank, world_size)


def apply_fsdp2(model: nn.Module, *, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    if not dist.is_initialized():
        raise RuntimeError("multi-rank FSDP2 requires an initialized process group")
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    for layer in model.layers:
        fully_shard(layer, mp_policy=policy, reshard_after_forward=True)
    selector = getattr(model, "candidate_selector", None)
    if selector is not None:
        fully_shard(selector, mp_policy=policy, reshard_after_forward=True)
    markov = getattr(model, "markov_head", None)
    if markov is not None:
        fully_shard(markov, mp_policy=policy, reshard_after_forward=False)
    confidence = getattr(model, "confidence_head", None)
    if confidence is not None:
        fully_shard(confidence, mp_policy=policy, reshard_after_forward=True)
    fully_shard(model, mp_policy=policy, reshard_after_forward=True)
    return model


def configure_accumulation(model: nn.Module, *, synchronize: bool) -> None:
    sync = getattr(model, "set_requires_gradient_sync", None)
    # FSDP2's dynamic accumulation control is a *backward* reshard toggle.
    # ``set_reshard_after_forward`` is the argument accepted by ``fully_shard``
    # at construction time, not a public method on the resulting module in
    # current PyTorch (including torch-npu releases based on FSDP2).
    reshard = getattr(model, "set_reshard_after_backward", None)
    # A few vendor forks exposed the old name; retain a narrow compatibility
    # fallback, but never prefer it over the standard API.
    if reshard is None:
        reshard = getattr(model, "set_reshard_after_forward", None)
    if sync is None and reshard is None:
        return
    if sync is None or reshard is None:
        raise TypeError("incomplete FSDP2 accumulation API")
    sync(bool(synchronize), recurse=True)
    reshard(bool(synchronize), recurse=True)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def shutdown_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
