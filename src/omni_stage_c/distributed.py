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


def initialize_distributed(device_kind: str, backend: str) -> DistributedContext:
    rank, local_rank = int(os.getenv("RANK", "0")), int(os.getenv("LOCAL_RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    if device_kind == "npu":
        importlib.import_module("torch_npu")
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("torch_npu is installed but no Ascend NPU is available")
        torch.npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
        if backend != "hccl":
            raise ValueError("NPU Stage C requires HCCL")
    elif device_kind == "cpu":
        device, backend = torch.device("cpu"), "gloo"
    else:
        raise ValueError("device must be npu or cpu")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://",
                                timeout=timedelta(minutes=30))
    return DistributedContext(device, rank, local_rank, world)


def apply_fsdp2(model: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    if not dist.is_initialized():
        raise RuntimeError("FSDP2 requires a distributed process group")
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                  reduce_dtype=torch.float32,
                                  output_dtype=torch.bfloat16)
    for layer in model.layers:
        fully_shard(layer, mp_policy=policy, reshard_after_forward=True)
    for name, reshard in (("candidate_selector", True), ("markov_head", False),
                          ("confidence_head", True)):
        module = getattr(model, name, None)
        if module is not None:
            fully_shard(module, mp_policy=policy, reshard_after_forward=reshard)
    fully_shard(model, mp_policy=policy, reshard_after_forward=True)
    return model


def configure_accumulation(model: nn.Module, synchronize: bool) -> None:
    sync, reshard = getattr(model, "set_requires_gradient_sync", None), getattr(
        model, "set_reshard_after_forward", None
    )
    if sync is not None:
        sync(bool(synchronize), recurse=True)
        reshard(bool(synchronize), recurse=True)


def global_denominator(local: torch.Tensor) -> tuple[torch.Tensor, int]:
    value, world = local.detach().float().clone(), 1
    if dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(value)
    return value, world


def shutdown_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
