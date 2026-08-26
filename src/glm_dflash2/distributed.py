from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Mapping

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


def rank_epoch_seed(base_seed: int, rank: int, epoch: int) -> int:
    return int(base_seed) + 1_000_003 * int(rank) + 10_000_019 * int(epoch)


def resolve_device_backend(kind: str, *, local_rank: int) -> tuple[torch.device, str]:
    normalized = kind.lower()
    if normalized == "cpu":
        return torch.device("cpu"), "gloo"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but torch.cuda is unavailable")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), "nccl"
    if normalized == "npu":
        try:
            importlib.import_module("torch_npu")
        except ImportError as exc:
            raise RuntimeError(
                "NPU training requires a matching torch_npu installation and CANN environment"
            ) from exc
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("torch_npu imported but no Ascend NPU is available")
        torch.npu.set_device(local_rank)
        return torch.device("npu", local_rank), "hccl"
    raise ValueError("device must be one of: cpu, cuda, npu")


def initialize_distributed(device_kind: str, *, timeout_minutes: int = 30) -> DistributedContext:
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


def fsdp2_method_modules(model: nn.Module) -> list[tuple[str, nn.Module, bool]]:
    """Return heads invoked outside the common backbone forward.

    The final boolean keeps gathered parameters resident across repeated calls.
    DSpark invokes its Markov head once per vocabulary chunk, so resharing after
    every chunk would add a collective to every chunk.
    """

    result: list[tuple[str, nn.Module, bool]] = []
    for name, keep_unsharded in (
        ("candidate_selector", False),
        ("markov_head", True),
        ("confidence_head", False),
    ):
        module = getattr(model, name, None)
        if module is not None:
            result.append((name, module, keep_unsharded))
    return result


def apply_fsdp2(model: nn.Module, *, enabled: bool, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    if not enabled:
        return model
    if not dist.is_initialized():
        raise RuntimeError("FSDP2 requires an initialized distributed process group")
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32, output_dtype=dtype)
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("FSDP2 draft model must expose decoder layers")
    for layer in layers:
        fully_shard(layer, mp_policy=policy, reshard_after_forward=True)
    for _, module, keep_unsharded in fsdp2_method_modules(model):
        fully_shard(
            module,
            mp_policy=policy,
            reshard_after_forward=not keep_unsharded,
        )
    fully_shard(model, mp_policy=policy, reshard_after_forward=True)
    return model


def configure_accumulation(model: nn.Module, *, synchronize: bool) -> None:
    """Use the FSDP2-native synchronization controls, never FSDP1 no_sync."""

    set_sync = getattr(model, "set_requires_gradient_sync", None)
    set_reshard = getattr(model, "set_reshard_after_forward", None)
    if set_sync is None and set_reshard is None:
        return
    if set_sync is None or set_reshard is None:
        raise TypeError("partially FSDP2-like model exposes an incomplete accumulation API")
    set_sync(bool(synchronize), recurse=True)
    set_reshard(bool(synchronize), recurse=True)


def reduce_additive_metrics(metrics: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    values = {name: value.detach().clone() for name, value in metrics.items()}
    if dist.is_initialized() and dist.get_world_size() > 1:
        for value in values.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return values


def global_weighted_mean(
    local_mean: torch.Tensor, local_weight: torch.Tensor
) -> torch.Tensor:
    """Scale a local mean so averaged DDP/FSDP gradients equal a global mean.

    Only the detached denominator is communicated.  FSDP/DDP subsequently
    averages gradients across ranks, hence the explicit world-size factor.
    """

    global_weight, world_size = _global_weight(
        local_weight, device=local_mean.device
    )
    if not bool(global_weight > 0):
        return local_mean * 0.0
    scale = (
        local_weight.detach().to(device=local_mean.device, dtype=torch.float32)
        * float(world_size)
        / global_weight
    )
    return local_mean * scale.to(dtype=local_mean.dtype)


def _global_weight(
    local_weight: torch.Tensor, *, device: torch.device
) -> tuple[torch.Tensor, int]:
    global_weight = local_weight.detach().to(device=device, dtype=torch.float32).clone()
    world_size = 1
    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        dist.all_reduce(global_weight, op=dist.ReduceOp.SUM)
    return global_weight, world_size


def scale_loss_for_accumulation(
    globally_normalized_loss: torch.Tensor,
    local_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover an additive numerator loss for exact gradient accumulation.

    ``global_weighted_mean`` accounts for gradient averaging across ranks.  By
    multiplying that loss by the global denominator, each backward contributes
    an unnormalized global numerator.  The caller divides gradients once by
    the sum of returned denominators at the optimizer boundary.
    """

    global_weight, _ = _global_weight(
        local_weight, device=globally_normalized_loss.device
    )
    if not bool(global_weight > 0):
        return globally_normalized_loss * 0.0, global_weight
    return (
        globally_normalized_loss * global_weight.to(globally_normalized_loss.dtype),
        global_weight,
    )


def distributed_any(value: bool, device: torch.device) -> bool:
    """Return one globally agreed boolean without rank-local control flow."""

    flag = torch.tensor(int(bool(value)), device=device, dtype=torch.int32)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def shutdown_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
