#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm53_w4.checkpointing import (  # noqa: E402
    TrainingProgress,
    load_checkpoint,
    save_checkpoint,
)
from glm53_w4.contracts import TARGET_CONTRACT  # noqa: E402
from glm53_w4.data import expand_training_window, select_training_window  # noqa: E402
from glm53_w4.dflash2 import DFlash2Model  # noqa: E402
from glm53_w4.distributed import (  # noqa: E402
    apply_fsdp2,
    configure_accumulation,
    initialize_distributed,
    shutdown_distributed,
)
from glm53_w4.dspark import DSparkModel  # noqa: E402
from glm53_w4.hidden_cache import HiddenCacheDataset  # noqa: E402
from glm53_w4.modeling import DraftModelConfig  # noqa: E402
from glm53_w4.target_io import load_frozen_target_io  # noqa: E402
from glm53_w4.trainer import OfflineDrafterTrainer, recipe_for  # noqa: E402
from glm53_w4.trainer import (  # noqa: E402
    accumulation_real_count,
    rank_loss_scale,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline BF16 DSpark/DFlash2 training for formal GLM-5.3 W4A8."
    )
    parser.add_argument("--method", choices=("dspark", "dflash2"), required=True)
    parser.add_argument("--block-size", type=int, choices=(8, 16), required=True)
    parser.add_argument("--hidden-cache", type=Path, required=True)
    parser.add_argument("--target-io", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-token-id", type=int, required=True)
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--strategy", choices=("fsdp2", "none"), default="fsdp2")
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--anchor-chunk-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--training-window", type=int, default=4096)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="recompute draft layers during backward to reduce NPU activation memory",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _cosine_multiplier(step: int, total: int, warmup: int, minimum: float = 0.1) -> float:
    if warmup and step <= warmup:
        return min(1.0, step / warmup)
    if total <= warmup:
        return 1.0
    progress = min(1.0, (step - warmup) / (total - warmup))
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _assignments(length: int, *, epoch: int, rank: int, world: int, seed: int) -> list[tuple[int, bool]]:
    generator = torch.Generator().manual_seed(seed + epoch)
    values = torch.randperm(length, generator=generator).tolist()
    total = math.ceil(length / world) * world
    values.extend(values[index % length] for index in range(total - length))
    return [(values[index], index < length) for index in range(rank, total, world)]


def _validate_cache_io(cache: HiddenCacheDataset, io: Any) -> None:
    manifest = cache.manifest
    if tuple(manifest.get("layer_ids", ())) != TARGET_CONTRACT.layer_ids:
        raise ValueError("hidden cache layer order differs from [1,20,38,56,75]")
    if int(manifest.get("hidden_size", -1)) != TARGET_CONTRACT.hidden_size:
        raise ValueError("hidden cache width differs from formal GLM-5.3")
    provenance = manifest.get("provenance") or {}
    if provenance.get("logical_layer_ids") != list(TARGET_CONTRACT.layer_ids):
        raise ValueError("hidden cache logical layer provenance is incomplete")
    if provenance.get("requested_layer_ids") != [
        *TARGET_CONTRACT.layer_ids,
        TARGET_CONTRACT.num_hidden_layers,
    ]:
        raise ValueError("hidden cache requested layer provenance is incomplete")
    if provenance.get("layer_indexing") != "vllm-layer-id":
        raise ValueError("hidden cache layer indexing convention is unknown")
    if provenance.get("target_quantization") != "W4A8":
        raise ValueError("hidden cache was not extracted from W4A8 target")
    if provenance.get("runtime_backend") != "vllm-ascend":
        raise ValueError("hidden cache was not extracted by vLLM-Ascend")
    for key in (
        "source_model_fingerprint",
        "tokenizer_fingerprint",
        "target_io_weights_sha256",
    ):
        if not provenance.get(key) or not io.manifest.get(key):
            raise ValueError("hidden cache or target I/O is missing identity provenance")
    for key in (
        "source_model_fingerprint",
        "tokenizer_fingerprint",
        "target_io_weights_sha256",
    ):
        if provenance.get(key) != io.manifest.get(key):
            raise ValueError(f"hidden cache and target I/O {key} differ")


def _batch(
    cache: HiddenCacheDataset,
    index: int,
    *,
    epoch: int,
    max_tokens: int,
    block_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    token_fields = cache.token_fields(index)
    core = select_training_window(
        token_fields["loss_mask"],
        sample_id=token_fields["sample_id"],
        epoch=epoch,
        max_tokens=max_tokens,
        block_size=block_size,
        seed=seed,
    )
    read = expand_training_window(
        core,
        total_tokens=int(token_fields["input_ids"].numel()),
        sliding_window=2048,
        block_size=block_size,
    )
    row = cache.get_window(index, read.start, read.end)
    # The left halo supplies history but must never become an anchor. The
    # right tail supplies successors for anchors near the core boundary.
    anchor_mask = row["loss_mask"].clone()
    anchor_mask[: read.anchor_start] = False
    anchor_mask[read.anchor_end :] = False
    return {
        "input_ids": row["input_ids"].unsqueeze(0).to(device),
        "loss_mask": row["loss_mask"].unsqueeze(0).to(device),
        "anchor_mask": anchor_mask.unsqueeze(0).to(device),
        "attention_mask": row["attention_mask"].unsqueeze(0).to(device),
        "aux_hidden_states": row["aux_hidden_states"].unsqueeze(0).to(device),
        "target_final_hidden": row["target_final_hidden"].unsqueeze(0).to(device),
        "position_offset": torch.tensor([row["position_offset"]], device=device),
        "sample_ids": [row["sample_id"]],
    }


def main() -> None:
    args = _parse_args()
    recipe = recipe_for(args.method, args.block_size)
    if args.anchors < 1 or args.anchor_chunk_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("anchors/chunk/accumulation must be positive")
    if args.training_window < 2 * args.block_size:
        raise ValueError("training window is too short")
    context = initialize_distributed(args.device)
    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.manual_seed_all(args.seed)
        cache = HiddenCacheDataset(args.hidden_cache, verify_checksums=True)
        if len(cache) == 0:
            raise ValueError("hidden cache contains no samples")
        target_io = load_frozen_target_io(args.target_io, device=context.device)
        _validate_cache_io(cache, target_io)
        config = DraftModelConfig(
            block_size=args.block_size,
            mask_token_id=args.mask_token_id,
            anchor_chunk_size=args.anchor_chunk_size,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        if not 0 <= args.mask_token_id < config.vocab_size:
            raise ValueError("mask token ID is outside the target vocabulary")
        model = (
            DSparkModel(config)
            if args.method == "dspark"
            else DFlash2Model(config)
        ).to(device=context.device, dtype=torch.bfloat16)
        use_fsdp = args.strategy == "fsdp2"
        if context.world_size > 1 and not use_fsdp:
            raise ValueError("multi-rank production training requires FSDP2")
        model = apply_fsdp2(model, enabled=use_fsdp)
        trainer = OfflineDrafterTrainer(
            model,
            target_io,
            method=args.method,
            gamma=recipe.gamma,
            anchors_per_sample=args.anchors,
            global_seed=args.seed,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=recipe.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        micro_per_epoch = math.ceil(len(cache) / context.world_size)
        total_micro = micro_per_epoch * recipe.epochs
        total_steps = math.ceil(total_micro / args.gradient_accumulation)
        if args.max_steps is not None:
            total_steps = min(total_steps, args.max_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _cosine_multiplier(
                step, total_steps, recipe.warmup_steps, 0.1
            ),
        )
        semantic = {
            "schema": "formal-glm53-w4-stage-c-v2",
            "method": args.method,
            "model_config": config.to_dict(),
            "recipe": recipe.__dict__,
            "cache_provenance": cache.manifest["provenance"],
            "target_io_weights_sha256": target_io.manifest["weights_sha256"],
            "mask_token_id": args.mask_token_id,
            "training_window": args.training_window,
            "anchors": args.anchors,
            "anchor_chunk_size": args.anchor_chunk_size,
            "gradient_accumulation": args.gradient_accumulation,
            "vocab_chunk_size": args.vocab_chunk_size,
            "seed": args.seed,
            "device": args.device,
            "strategy": args.strategy,
            "world_size": context.world_size,
            "sample_from_anchor": False,
            "num_speculative_tokens": config.block_size - 1,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if context.is_main:
            config_path = args.output_dir / "training_config.json"
            if config_path.exists() and json.loads(config_path.read_text()) != semantic:
                raise ValueError("output directory contains different training semantics")
            config_path.write_text(
                json.dumps(semantic, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        progress = TrainingProgress(0, 0, 0, 0)
        if args.resume is not None:
            progress = load_checkpoint(
                args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                semantic_config=semantic,
            )
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        real_count = 0
        stop = False
        for epoch in range(progress.epoch, recipe.epochs):
            assignments = _assignments(
                len(cache),
                epoch=epoch,
                rank=context.rank,
                world=context.world_size,
                seed=args.seed,
            )
            cursor_start = progress.sample_cursor if epoch == progress.epoch else 0
            for cursor in range(cursor_start, len(assignments)):
                index, include = assignments[cursor]
                is_last = epoch + 1 == recipe.epochs and cursor + 1 == len(assignments)
                boundary = accumulated + 1 == args.gradient_accumulation or is_last
                configure_accumulation(model, synchronize=boundary)
                batch = _batch(
                    cache,
                    index,
                    epoch=epoch,
                    max_tokens=args.training_window,
                    block_size=args.block_size,
                    seed=args.seed,
                    device=context.device,
                )
                output = trainer(batch, epoch=epoch)
                real_ranks = torch.tensor(int(include), device=context.device, dtype=torch.float32)
                if dist.is_initialized():
                    dist.all_reduce(real_ranks, op=dist.ReduceOp.SUM)
                rank_scale = rank_loss_scale(
                    include=include, world_size=context.world_size
                )
                (output.loss * rank_scale).backward()
                accumulated += 1
                real_count += int(real_ranks.item())
                if not boundary:
                    continue
                denominator = accumulation_real_count([real_count])
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(float(denominator))
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                consumed_micro = accumulated
                accumulated = 0
                real_count = 0
                next_epoch = epoch
                next_cursor = cursor + 1
                if next_cursor == len(assignments):
                    next_epoch, next_cursor = epoch + 1, 0
                progress = TrainingProgress(
                    progress.global_step + 1,
                    # ``micro_step`` counts consumed samples, not optimizer
                    # updates.  Persist the full accumulation interval so a
                    # resume cannot replay or skip microbatches.
                    progress.micro_step + consumed_micro,
                    next_epoch,
                    next_cursor,
                )
                if context.is_main:
                    metrics = " ".join(
                        f"{name}={float(value):.5f}" for name, value in output.metrics.items()
                    )
                    line = (
                        f"step={progress.global_step} epoch={epoch} "
                        f"loss={float(output.loss.detach()):.6f} lr={scheduler.get_last_lr()[0]:.3e} "
                        f"grad_norm={float(grad_norm):.4f} {metrics}"
                    )
                    print(line, flush=True)
                    with (args.output_dir / "train.log").open("a", encoding="utf-8") as handle:
                        handle.write(line + "\n")
                should_save = progress.global_step % args.checkpoint_every == 0
                reached_limit = args.max_steps is not None and progress.global_step >= args.max_steps
                if should_save or reached_limit or is_last:
                    save_checkpoint(
                        args.output_dir / f"step-{progress.global_step}",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        progress=progress,
                        semantic_config=semantic,
                    )
                if reached_limit:
                    stop = True
                    break
            if stop:
                break
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
