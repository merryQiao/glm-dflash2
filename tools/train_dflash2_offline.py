#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, DistributedSampler

from glm_dflash2.checkpointing import (
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)
from glm_dflash2.dflash2_blocks import NoValidAnchorsError, sample_anchor_positions
from glm_dflash2.dflash2_model import Qwen3DFlash2DraftModel, build_glm52_dflash2_config
from glm_dflash2.distributed import (
    apply_fsdp2,
    barrier,
    configure_accumulation,
    distributed_any,
    initialize_distributed,
    rank_epoch_seed,
    reduce_additive_metrics,
    shutdown_distributed,
)
from glm_dflash2.hidden_cache import DFlashHiddenCollator, PackedHiddenDataset
from glm_dflash2.offline_trainer import OfflineDFlash2Trainer
from glm_dflash2.target_io import load_frozen_target_io


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline GLM-5.2 DFlash2 training")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--target-io-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--mask-token-id", type=int, required=True)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--device", choices=("npu", "cuda", "cpu"), default="npu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--token-chunk-size", type=int, default=256)
    parser.add_argument("--vocab-chunk-size", type=int, default=4096)
    parser.add_argument("--block-size", type=int, choices=(16,), default=16)
    parser.add_argument("--num-anchors", type=int, choices=(64,), default=64)
    parser.add_argument("--gamma", type=float, choices=(7.0,), default=7.0)
    parser.add_argument("--selector-rank", type=int, choices=(256,), default=256)
    parser.add_argument("--selector-top-k", type=int, choices=(16,), default=16)
    parser.add_argument("--hidden-size", type=int, choices=(6144,), default=6144)
    parser.add_argument("--intermediate-size", type=int, choices=(12288,), default=12288)
    parser.add_argument("--num-draft-layers", type=int, choices=(5,), default=5)
    parser.set_defaults(fsdp2=True)
    return parser


def _scheduler(optimizer, *, warmup: int, total: int, min_ratio: float):
    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(step, 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _sample_or_dummy_anchors(batch, trainer):
    try:
        anchors, keep = sample_anchor_positions(
            batch["loss_mask"],
            attention_mask=batch.get("attention_mask"),
            block_size=trainer.draft_model.block_size,
            num_anchors=trainer.num_anchors,
            generator=trainer.anchor_generator,
        )
        return anchors, keep, True
    except NoValidAnchorsError:
        shape = (int(batch["input_ids"].shape[0]), 1)
        anchors = torch.zeros(shape, dtype=torch.long, device=batch["input_ids"].device)
        keep = torch.zeros(shape, dtype=torch.bool, device=batch["input_ids"].device)
        return anchors, keep, False


def _semantic_config(args, dataset, target_io, total_steps: int) -> dict:
    value = {
        "schema": "glm-dflash2-train-semantics-v1",
        "cache": {
            "samples": len(dataset),
            "spec": dataset.manifest.get("spec"),
            "provenance": dataset.manifest.get("provenance"),
        },
        "target_io": {
            "source_model_fingerprint": target_io.manifest.get("source_model_fingerprint"),
            "weights_sha256": target_io.manifest.get("weights_sha256"),
            "vocab_size": target_io.manifest.get("vocab_size"),
            "hidden_size": target_io.manifest.get("hidden_size"),
        },
        "recipe": {
            key: getattr(args, key)
            for key in (
                "mask_token_id", "pad_token_id", "epochs", "batch_size", "grad_accum",
                "lr", "beta1", "beta2", "min_lr_ratio", "warmup_steps", "weight_decay",
                "seed", "block_size", "num_anchors", "gamma", "selector_rank",
                "selector_top_k", "hidden_size", "intermediate_size", "num_draft_layers",
            )
        },
        "total_optimizer_steps": int(total_steps),
    }
    # Canonicalize tuples and NumPy-like scalars exactly as checkpoint JSON will.
    return json.loads(json.dumps(value, sort_keys=True))


def _add_metrics(accumulator: dict[str, torch.Tensor], step) -> None:
    values = {
        "base_numerator": step.base_numerator,
        "base_denominator": step.base_denominator,
        "selector_numerator": step.selector_numerator,
        "selector_denominator": step.selector_denominator,
        "base_correct": step.base_correct,
        "selector_correct": step.selector_correct,
        "valid_tokens": step.valid_tokens,
        "base_accept_total": step.base_accept_total,
        "selector_accept_total": step.selector_accept_total,
        "valid_blocks": step.valid_blocks,
        "candidate_hits": step.candidate_hits,
        "candidate_total": step.candidate_total,
    }
    for key, value in values.items():
        detached = value.detach().to(torch.float32)
        accumulator[key] = accumulator.get(key, torch.zeros_like(detached)) + detached


def _ratios(reduced: dict[str, torch.Tensor]) -> dict[str, float]:
    def ratio(numerator: str, denominator: str) -> float:
        den = float(reduced[denominator].item())
        return float(reduced[numerator].item()) / den if den > 0 else 0.0

    base_loss = ratio("base_numerator", "base_denominator")
    selector_loss = ratio("selector_numerator", "selector_denominator")
    return {
        "loss": base_loss + selector_loss,
        "base_loss": base_loss,
        "selector_loss": selector_loss,
        "base_accuracy": ratio("base_correct", "valid_tokens"),
        "selector_accuracy": ratio("selector_correct", "valid_tokens"),
        "base_accept": ratio("base_accept_total", "valid_blocks"),
        "selector_accept": ratio("selector_accept_total", "valid_blocks"),
        "candidate_recall": ratio("candidate_hits", "candidate_total"),
        "valid_tokens": float(reduced["valid_tokens"].item()),
        "valid_blocks": float(reduced["valid_blocks"].item()),
    }


def _export(model, config, output_dir: Path, is_main: bool) -> None:
    barrier()
    state = None
    if hasattr(model, "set_requires_gradient_sync"):
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        state = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    elif is_main:
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if is_main:
        export = output_dir / "export"
        export.mkdir(parents=True, exist_ok=True)
        save_file({key: value.contiguous() for key, value in state.items()}, export / "model.safetensors")
        config.to_json_file(export / "config.json")
        (export / "EXPORT_NOTE.txt").write_text(
            "Draft-only artifact: target embed_tokens/lm_head are intentionally shared at runtime.\n",
            encoding="utf-8",
        )
    barrier()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = initialize_distributed(args.device)
    seed = rank_epoch_seed(args.seed, context.rank, 0)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    output_dir = Path(args.output_dir)
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    try:
        dataset = PackedHiddenDataset(args.cache_dir, require_frozen=True)
        target_io = load_frozen_target_io(
            args.target_io_dir, device=context.device, dtype=torch.bfloat16
        )
        config = build_glm52_dflash2_config(
            vocab_size=int(target_io.manifest["vocab_size"]),
            mask_token_id=args.mask_token_id,
        )
        draft = Qwen3DFlash2DraftModel(config).to(context.device, dtype=torch.bfloat16)
        draft = apply_fsdp2(draft, enabled=args.fsdp2, dtype=torch.bfloat16)
        trainer = OfflineDFlash2Trainer(
            draft,
            target_io,
            cache_manifest=dataset.manifest,
            num_anchors=args.num_anchors,
            gamma=args.gamma,
            selector_loss_weight=1.0,
            token_chunk_size=args.token_chunk_size,
            vocab_chunk_size=args.vocab_chunk_size,
            anchor_seed=seed,
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=DFlashHiddenCollator(args.pad_token_id),
            num_workers=args.num_workers,
            pin_memory=context.device.type != "cpu",
        )
        optimizer = torch.optim.AdamW(
            trainer.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        total_steps = args.epochs * math.ceil(len(loader) / args.grad_accum)
        scheduler = _scheduler(
            optimizer,
            warmup=args.warmup_steps,
            total=max(1, total_steps),
            min_ratio=args.min_lr_ratio,
        )
        semantic_config = _semantic_config(args, dataset, target_io, total_steps)
        progress = TrainingProgress(0, 0, 0, 0)
        resumed = False
        if args.resume:
            progress = load_training_checkpoint(
                args.resume,
                model=draft,
                optimizer=optimizer,
                scheduler=scheduler,
                anchor_generator=trainer.anchor_generator,
                expected_semantic_config=semantic_config,
            )
            resumed = True

        if context.is_main:
            with (output_dir / "launch_args.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(vars(args), sort_keys=True) + "\n")

        log_handle = (output_dir / "train.jsonl").open("a", encoding="utf-8") if context.is_main else None
        accumulation = 0
        metric_accumulator: dict[str, torch.Tensor] = {}
        stop = False
        for epoch in range(progress.epoch, args.epochs):
            sampler.set_epoch(epoch)
            if not (resumed and epoch == progress.epoch):
                trainer.anchor_generator.manual_seed(rank_epoch_seed(args.seed, context.rank, epoch))
            cursor = progress.sample_cursor if resumed and epoch == progress.epoch else 0
            resumed = False
            for batch_index, batch in enumerate(loader):
                if batch_index < cursor:
                    continue
                batch = _move_batch(batch, context.device)
                anchors, keep, local_has_anchors = _sample_or_dummy_anchors(batch, trainer)
                global_has_anchors = distributed_any(local_has_anchors, context.device)
                is_last_batch = batch_index + 1 == len(loader)
                if not global_has_anchors and not (is_last_batch and accumulation > 0):
                    progress = TrainingProgress(
                        progress.global_step, progress.micro_step + 1, epoch, batch_index + 1
                    )
                    continue
                contributes = global_has_anchors
                next_accumulation = accumulation + int(contributes)
                should_step = next_accumulation == args.grad_accum or (
                    is_last_batch and next_accumulation > 0
                )
                configure_accumulation(draft, synchronize=should_step)
                started = time.perf_counter()
                with torch.autocast(
                    device_type=context.device.type,
                    dtype=torch.bfloat16,
                    enabled=context.device.type != "cpu",
                ):
                    step = trainer(
                        batch, anchor_positions=anchors, block_keep_mask=keep
                    )
                if not bool(torch.isfinite(step.loss)):
                    raise FloatingPointError("non-finite training loss")
                step.loss.backward()
                if contributes:
                    accumulation += 1
                    _add_metrics(metric_accumulator, step)
                micro_step = progress.micro_step + 1
                global_step = progress.global_step
                if should_step:
                    for parameter in trainer.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulation)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(trainer.parameters()), args.max_grad_norm
                    )
                    if not bool(torch.isfinite(grad_norm)):
                        raise FloatingPointError("non-finite trainable gradient norm")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulation = 0
                    global_step += 1
                progress = TrainingProgress(global_step, micro_step, epoch, batch_index + 1)

                if should_step and global_step % args.log_every == 0:
                    reduced = reduce_additive_metrics(metric_accumulator)
                    if context.is_main:
                        record = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "sample_cursor": batch_index + 1,
                            **_ratios(reduced),
                            "lr": scheduler.get_last_lr()[0],
                            "grad_norm": float(grad_norm),
                            "step_seconds": time.perf_counter() - started,
                        }
                        line = json.dumps(record, ensure_ascii=False)
                        print(line, flush=True)
                        log_handle.write(line + "\n")
                        log_handle.flush()
                if should_step:
                    metric_accumulator = {}
                if should_step and args.save_every > 0 and global_step % args.save_every == 0:
                    save_training_checkpoint(
                        output_dir / f"step-{global_step}",
                        model=draft,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        progress=progress,
                        anchor_generator=trainer.anchor_generator,
                        at_optimizer_boundary=True,
                        semantic_config=semantic_config,
                    )
                if should_step and args.max_steps > 0 and global_step >= args.max_steps:
                    stop = True
                    break
            if stop:
                break
            progress = TrainingProgress(progress.global_step, progress.micro_step, epoch + 1, 0)
        if accumulation:
            raise RuntimeError("training ended away from an optimizer-step boundary")
        final_path = output_dir / f"step-{progress.global_step}"
        if not (final_path / "COMPLETE").exists():
            save_training_checkpoint(
                final_path,
                model=draft,
                optimizer=optimizer,
                scheduler=scheduler,
                progress=progress,
                anchor_generator=trainer.anchor_generator,
                at_optimizer_boundary=True,
                semantic_config=semantic_config,
            )
        trainer.assert_frozen_io_unchanged()
        _export(draft, config, output_dir, context.is_main)
        if log_handle is not None:
            log_handle.close()
        return 0
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    sys.exit(main())
