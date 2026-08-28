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
    backend: str
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class FSDP2WrapTarget:
    name: str
    module: nn.Module
    reshard_after_forward: bool


def resolve_device_backend(kind: str, *, local_rank: int) -> tuple[torch.device, str]:
    normalized = str(kind).strip().lower()
    if normalized == "cpu":
        return torch.device("cpu"), "gloo"
    if normalized != "npu":
        raise ValueError("device must be either cpu or npu")
    try:
        importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError(
            "NPU training requires a matching torch_npu installation and CANN environment"
        ) from exc
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        raise RuntimeError("torch_npu imported but no Ascend NPU is available")
    npu.set_device(int(local_rank))
    try:
        device = torch.device("npu", int(local_rank))
    except RuntimeError:
        # A lightweight test double cannot register the private-use device name.
        device = torch.device("privateuseone", int(local_rank))
    return device, "hccl"


def initialize_distributed(
    device_kind: str, *, timeout_minutes: int = 30
) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device, backend = resolve_device_backend(device_kind, local_rank=local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=int(timeout_minutes)),
        )
    return DistributedContext(device, backend, rank, local_rank, world_size)


def fsdp2_wrap_policy(model: nn.Module) -> tuple[FSDP2WrapTarget, ...]:
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("FSDP2 draft model must expose decoder layers")
    targets = [
        FSDP2WrapTarget(f"layers.{index}", layer, True)
        for index, layer in enumerate(layers)
    ]
    for name, reshard in (
        ("candidate_selector", True),
        ("markov_head", False),
        ("confidence_head", True),
    ):
        module = getattr(model, name, None)
        if module is not None:
            targets.append(FSDP2WrapTarget(name, module, reshard))
    targets.append(FSDP2WrapTarget("<root>", model, True))
    return tuple(targets)


def apply_fsdp2(
    model: nn.Module, *, enabled: bool, dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    if not enabled:
        return model
    if not dist.is_initialized():
        raise RuntimeError("FSDP2 requires an initialized process group")
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(
        param_dtype=dtype, reduce_dtype=torch.float32, output_dtype=dtype
    )
    for target in fsdp2_wrap_policy(model):
        fully_shard(
            target.module,
            mp_policy=policy,
            reshard_after_forward=target.reshard_after_forward,
        )
    return model


def configure_accumulation(model: nn.Module, *, synchronize: bool) -> None:
    set_sync = getattr(model, "set_requires_gradient_sync", None)
    set_reshard = getattr(model, "set_reshard_after_forward", None)
    if set_sync is None and set_reshard is None:
        return
    if set_sync is None or set_reshard is None:
        raise TypeError("incomplete FSDP2 accumulation interface")
    set_sync(bool(synchronize), recurse=True)
    set_reshard(bool(synchronize), recurse=True)


def _global_denominator(local_denominator: torch.Tensor) -> tuple[torch.Tensor, int]:
    denominator = local_denominator.detach().to(dtype=torch.float32).clone()
    world_size = 1
    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    return denominator, world_size


def global_additive_mean(
    local_numerator: torch.Tensor, local_denominator: torch.Tensor
) -> torch.Tensor:
    denominator, world_size = _global_denominator(local_denominator)
    if not bool(denominator > 0):
        return local_numerator * 0.0
    return local_numerator * float(world_size) / denominator.to(
        device=local_numerator.device, dtype=local_numerator.dtype
    )


def scale_additive_loss_for_accumulation(
    local_numerator: torch.Tensor, local_denominator: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator, world_size = _global_denominator(local_denominator)
    if not bool(denominator > 0):
        return local_numerator * 0.0, denominator
    return local_numerator * float(world_size), denominator


def reduce_additive_metrics(
    metrics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    reduced = {name: value.detach().clone() for name, value in metrics.items()}
    if dist.is_initialized() and dist.get_world_size() > 1:
        for value in reduced.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return reduced


def shutdown_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
