from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .blocks import build_training_batch, sample_anchor_positions, select_training_window
from .checkpointing import TrainingProgress, load_checkpoint, save_checkpoint
from .contracts import DEFAULT_MASK_TOKEN_ID
from .dflash_model import DFlashModel
from .dflash2_model import DFlash2Model
from .distributed import (
    apply_fsdp2, configure_accumulation, global_denominator,
    initialize_distributed, shutdown_distributed,
)
from .dspark_model import DSparkModel
from .hidden_cache import PackedThinkerHiddenCache
from .modeling_common import DraftModelConfig
from .offline_trainer import OfflineMethodTrainer
from .target_io import FrozenTargetIO


def _model(method: str, config: DraftModelConfig):
    if method == "dflash":
        return DFlashModel(config)
    if method == "dflash2":
        return DFlash2Model(config)
    if method == "dspark":
        return DSparkModel(config)
    raise ValueError(method)


def _lr_multiplier(step: int, total: int, warmup: int = 1000) -> float:
    if warmup and step <= warmup:
        return min(1.0, step / warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _semantic_identity(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _assignments(count: int, epoch: int, rank: int, world: int, seed: int = 42
                 ) -> list[tuple[int, bool]]:
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    padded = math.ceil(count / world) * world
    assignments = [(order[i] if i < count else order[0], i < count) for i in range(padded)]
    return assignments[rank::world]


def _sample_window(row: dict[str, Any], *, epoch: int, block_size: int,
                   max_tokens: int = 4096) -> dict[str, Any]:
    window = select_training_window(
        row["loss_mask"], sample_id=row["sample_id"], epoch=epoch,
        block_size=block_size, max_tokens=max_tokens, seed=42,
    )
    result = {"sample_id": row["sample_id"]}
    for key in ("input_ids", "loss_mask", "auxiliary_hidden",
                "target_final_hidden", "position_ids"):
        result[key] = row[key][window.start:window.end]
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _global_component_mean(
    component: Any,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return a differentiable global mean and a detached copy for logging.

    FSDP averages synchronized gradients across ranks. Multiplying the local
    numerator by ``world / global_denominator`` before backward therefore
    yields the gradient of the true cross-rank weighted mean. The detached
    numerator is reduced separately because collectives cannot operate on the
    autograd graph used by FSDP2.
    """

    denominator, world = global_denominator(component.denominator)
    if not bool(denominator > 0):
        zero = component.numerator * 0.0
        return zero, zero.detach().float(), False
    differentiable = component.numerator * (world / denominator)
    numerator = component.numerator.detach().float().clone()
    if dist.is_initialized():
        dist.all_reduce(numerator)
    return differentiable, numerator / denominator, True


def run_training(args: Any) -> None:
    from tools.train_thinker_drafter import recipe_for

    recipe = recipe_for(args.method, args.block_size)
    if args.device != "npu" or args.backend != "hccl" or args.strategy != "fsdp2":
        raise ValueError("production Stage C requires NPU/HCCL/FSDP2")
    if args.mask_token_id != DEFAULT_MASK_TOKEN_ID:
        raise ValueError("mask identity differs from the audited unused padded row 152063")
    context = initialize_distributed(args.device, args.backend)
    try:
        torch.manual_seed(recipe.seed + context.rank)
        cache = PackedThinkerHiddenCache(args.hidden_cache_dir)
        target_io = FrozenTargetIO.load(
            args.target_io_dir, cache_fingerprint=cache.manifest["cache_fingerprint"]
        )
        if int(target_io.manifest["mask_token_id"]) != args.mask_token_id:
            raise ValueError("target I/O mask identity differs from training")
        sample_count = len(cache) if args.max_samples is None else min(len(cache), args.max_samples)
        if sample_count < 1:
            raise ValueError("empty Stage B cache")

        config = DraftModelConfig.production()
        model = _model(args.method, config).to(context.device, dtype=torch.bfloat16)
        model = apply_fsdp2(model, enabled=True)
        target_io.embedding.to(context.device)
        target_io.lm_head.to(context.device)
        trainer = OfflineMethodTrainer(
            method=args.method, block_size=args.block_size, model=model,
            target_embedding=target_io.embedding, target_lm_head=target_io.lm_head,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                                      betas=(0.9, 0.95), weight_decay=0.0)
        local_per_epoch = math.ceil(sample_count / context.world_size)
        # Epoch tails are optimizer boundaries, so each epoch contributes its
        # own ceil rather than allowing accumulation to cross epoch identity.
        total_steps = recipe.epochs * math.ceil(
            local_per_epoch / recipe.gradient_accumulation
        )
        if args.max_steps is not None:
            total_steps = min(total_steps, args.max_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: _lr_multiplier(step, total_steps, recipe.warmup_steps)
        )
        semantics = {
            "schema": "omni-stage-c-semantics-v1",
            "method": args.method, "block_size": args.block_size,
            "model": config.__dict__, "recipe": recipe.__dict__,
            "mask_token_id": args.mask_token_id,
            "cache_fingerprint": cache.manifest["cache_fingerprint"],
            "target_io_sha256": target_io.manifest["tensor_sha256"],
            "sample_count": sample_count, "world_size": context.world_size,
            "identity": "",
        }
        semantics["identity"] = _semantic_identity(semantics)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if context.is_main:
            _write_json(args.output_dir / "semantic_config.json", semantics)
        if dist.is_initialized():
            dist.barrier()

        progress = TrainingProgress(0, 0, 0)
        if args.resume:
            progress = load_checkpoint(args.resume, model=model, optimizer=optimizer,
                                       scheduler=scheduler, semantics=semantics)
        optimizer.zero_grad(set_to_none=True)
        denominator_accum = torch.zeros((), device=context.device, dtype=torch.float32)
        numerator_accum = torch.zeros((), device=context.device, dtype=torch.float32)
        dflash2_loss_accum = torch.zeros((), device=context.device, dtype=torch.float32)
        dflash2_valid_accum = False
        metric_accum: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        micro = 0
        global_step = progress.global_step
        metrics_path = args.output_dir / "train_metrics.jsonl"
        for epoch in range(progress.epoch, recipe.epochs):
            assignments = _assignments(sample_count, epoch, context.rank, context.world_size)
            start = progress.cursor if epoch == progress.epoch else 0
            for cursor in range(start, len(assignments)):
                index, include = assignments[cursor]
                row = _sample_window(cache[index], epoch=epoch, block_size=args.block_size)
                anchors = sample_anchor_positions(
                    row["loss_mask"], sample_id=row["sample_id"], epoch=epoch,
                    block_size=args.block_size, count=recipe.anchors_per_sample, seed=recipe.seed,
                )
                if not include:
                    anchors = type(anchors)(anchors.positions, torch.zeros_like(anchors.keep_mask))
                batch = build_training_batch(
                    row, anchors, block_size=args.block_size,
                    mask_token_id=args.mask_token_id, device=context.device,
                )
                boundary = (micro + 1 == recipe.gradient_accumulation) or (cursor + 1 == len(assignments))
                configure_accumulation(model, boundary)
                result = trainer.compute_loss(batch)
                for name, component in result.metrics.items():
                    if name == "total":
                        continue
                    numerator, denominator = metric_accum.get(
                        name,
                        (
                            torch.zeros((), device=context.device, dtype=torch.float32),
                            torch.zeros((), device=context.device, dtype=torch.float32),
                        ),
                    )
                    metric_accum[name] = (
                        numerator + component.numerator.detach().float(),
                        denominator + component.denominator.detach().float(),
                    )
                if args.method == "dflash2":
                    base_loss, base_log, base_valid = _global_component_mean(
                        result.metrics["base"]
                    )
                    selector_loss, selector_log, selector_valid = _global_component_mean(
                        result.metrics["selector"]
                    )
                    (base_loss + selector_loss).backward()
                    dflash2_loss_accum += base_log + selector_log
                    dflash2_valid_accum = (
                        dflash2_valid_accum or base_valid or selector_valid
                    )
                else:
                    total = result.metrics["total"]
                    global_den, world = global_denominator(total.denominator)
                    (total.numerator * world).backward()
                    denominator_accum += global_den
                    numerator_accum += total.numerator.detach().float()
                micro += 1
                if not boundary:
                    continue
                did_step = (
                    dflash2_valid_accum if args.method == "dflash2"
                    else bool(denominator_accum > 0)
                )
                if did_step:
                    gradient_divisor = (
                        torch.as_tensor(float(micro), device=context.device)
                        if args.method == "dflash2" else denominator_accum
                    )
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(gradient_divisor)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                if args.method == "dflash2":
                    step_loss = float((dflash2_loss_accum / max(1, micro)).cpu())
                else:
                    reduced_numerator = numerator_accum.clone()
                    if dist.is_initialized():
                        dist.all_reduce(reduced_numerator)
                    step_loss = (
                        float((reduced_numerator / denominator_accum).cpu())
                        if did_step else 0.0
                    )
                optimizer.zero_grad(set_to_none=True)
                denominator_accum.zero_()
                numerator_accum.zero_()
                dflash2_loss_accum.zero_()
                dflash2_valid_accum = False
                logged_metrics: dict[str, float] = {}
                for name, (numerator, denominator) in metric_accum.items():
                    pair = torch.stack((numerator, denominator))
                    if dist.is_initialized():
                        dist.all_reduce(pair)
                    logged_metrics[name] = (
                        float((pair[0] / pair[1]).cpu()) if bool(pair[1] > 0) else 0.0
                    )
                metric_accum.clear()
                micro = 0
                next_progress = TrainingProgress(
                    global_step, epoch + 1 if cursor + 1 == len(assignments) else epoch,
                    0 if cursor + 1 == len(assignments) else cursor + 1,
                )
                if did_step and context.is_main:
                    with metrics_path.open("a") as handle:
                        handle.write(json.dumps({
                            "step": global_step, "epoch": epoch,
                            "loss": step_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                            **logged_metrics,
                        }) + "\n")
                if did_step and global_step % args.checkpoint_every == 0:
                    save_checkpoint(
                        args.output_dir / "checkpoints" / f"step-{global_step:08d}",
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        progress=next_progress, semantics=semantics,
                    )
                progress = next_progress
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break
        final = args.output_dir / "checkpoints" / f"final-step-{global_step:08d}"
        if not (final / "COMPLETE").exists():
            save_checkpoint(final, model=model, optimizer=optimizer, scheduler=scheduler,
                            progress=progress, semantics=semantics)
    finally:
        shutdown_distributed()
