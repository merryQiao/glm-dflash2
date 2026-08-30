#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from glm53_drafters.blocks import (
    build_physical_blocks,
    sample_anchor_positions,
    select_training_window,
)
from glm53_drafters.capability import (
    validate_candidate_capability,
    write_candidate_capability,
)
from glm53_drafters.checkpointing import (
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)
from glm53_drafters.contracts import METHOD_BLOCK_SIZES, validate_method_block
from glm53_drafters.dflash2_model import DFlash2Model
from glm53_drafters.dflash_model import DFlashModel
from glm53_drafters.distributed import (
    apply_fsdp2,
    configure_accumulation,
    initialize_distributed,
    reduce_additive_metrics,
    scale_additive_loss_for_accumulation,
    shutdown_distributed,
)
from glm53_drafters.dspark_model import DSparkModel
from glm53_drafters.hidden_cache import PackedHiddenDataset
from glm53_drafters.modeling_common import DraftModelConfig
from glm53_drafters.offline_trainer import (
    OfflineMethodTrainer,
    TrainingBatch,
)
from glm53_drafters.target_io import (
    FrozenTargetIO,
    TARGET_IO_SCHEMA,
    load_frozen_target_io,
    validate_cache_io_compatibility,
)


@dataclass(frozen=True)
class TrainingRecipe:
    method: str
    block_size: int
    gamma: float
    anchors_per_sample: int = 64
    epochs: int = 3
    learning_rate: float = 6e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1
    per_rank_batch: int = 1
    gradient_accumulation: int = 8
    gradient_clip: float = 1.0
    seed: int = 42
    validation_samples: int = 128
    validation_every: int = 1000


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    train_route_counts: Mapping[str, int]
    validation_route_counts: Mapping[str, int]
    identity: str
    seed: int
    requested_validation_samples: int

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "glm53-heldout-split-v1",
            "identity": self.identity,
            "train_samples": len(self.train_indices),
            "validation_samples": len(self.validation_indices),
            "requested_validation_samples": self.requested_validation_samples,
            "seed": self.seed,
            "train_route_counts": dict(self.train_route_counts),
            "validation_route_counts": dict(self.validation_route_counts),
            "selection": "lowest-stable-sha256-by-sample-id",
            "validation_window_epoch": 0,
        }


class StepJsonlLogger:
    """Append optimizer-step records with conflict-safe resume reconciliation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[tuple[str, int], dict[str, Any]] = {}
        if self.path.exists():
            for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = self._key(record)
                if key in self._records and self._records[key] != record:
                    raise ValueError(f"conflicting metric record at {self.path}:{number}")
                self._records[key] = record

    @staticmethod
    def _key(record: Mapping[str, Any]) -> tuple[str, int]:
        split = record.get("split")
        step = record.get("step")
        if not isinstance(split, str) or not split:
            raise ValueError("metric record split must be a non-empty string")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("metric record step must be a non-negative integer")
        return split, step

    def append(self, record: Mapping[str, Any]) -> bool:
        value = json.loads(_canonical_json(record))
        key = self._key(value)
        existing = self._records.get(key)
        if existing is not None:
            if existing != value:
                raise ValueError(f"conflicting metric record for {key}")
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records[key] = value
        return True

    def reconcile(self, checkpoint_step: int) -> None:
        """Discard metrics newer than the checkpoint being resumed."""

        limit = int(checkpoint_step)
        kept = [
            record
            for (_, step), record in self._records.items()
            if step <= limit
        ]
        kept.sort(key=lambda row: (int(row["step"]), str(row["split"])))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(_canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        self._records = {self._key(record): record for record in kept}


def recipe_for(method: str, block_size: int) -> TrainingRecipe:
    method = str(method).lower()
    block_size = int(block_size)
    validate_method_block(method, block_size)
    return TrainingRecipe(
        method=method,
        block_size=block_size,
        gamma=7.0 if block_size == 16 else 4.0,
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _json_value(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json(value))


def semantic_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def cosine_warmup_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int = 1000,
    min_ratio: float = 0.1,
) -> float:
    step = max(0, int(step))
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    if not 0.0 <= float(min_ratio) <= 1.0:
        raise ValueError("min_ratio must be in [0,1]")
    if warmup_steps and step <= warmup_steps:
        return min(1.0, step / warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio) + (1.0 - float(min_ratio)) * cosine


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GLM-5.3 DFlash-family candidate from one frozen cache."
    )
    parser.add_argument("--method", choices=tuple(METHOD_BLOCK_SIZES), required=True)
    parser.add_argument("--block-size", type=int, choices=(8, 16), required=True)
    parser.add_argument("--hidden-cache", type=Path, required=True)
    parser.add_argument("--target-io", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--allow-smoke-unverified", action="store_true")
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--backend", choices=("hccl", "gloo"), default="hccl")
    parser.add_argument("--strategy", choices=("fsdp2", "none"), default="fsdp2")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--per-rank-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> TrainingRecipe:
    recipe = recipe_for(args.method, args.block_size)
    if args.device != "npu" or args.backend != "hccl" or args.strategy != "fsdp2":
        raise ValueError("production training requires NPU/HCCL/FSDP2")
    if args.dtype != "bfloat16":
        raise ValueError("production training requires BF16")
    if args.per_rank_batch != recipe.per_rank_batch:
        raise ValueError("production per-rank batch is fixed at 1")
    if args.gradient_accumulation != recipe.gradient_accumulation:
        raise ValueError("production gradient accumulation is fixed at 8")
    if args.mask_token_id is None or args.mask_token_id < 0:
        raise ValueError("--mask-token-id is required for production training")
    if args.checkpoint_every < 1 or args.vocab_chunk_size < 1:
        raise ValueError("checkpoint and vocabulary chunk sizes must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.allow_smoke_unverified and (
        args.max_samples is None or args.max_samples > 50
    ):
        raise ValueError("smoke_unverified training requires a total 1-50 sample bound")
    return recipe


def _build_model(config: DraftModelConfig, method: str) -> nn.Module:
    if method == "dflash":
        return DFlashModel(config)
    if method == "dflash2":
        return DFlash2Model(config)
    if method == "dspark":
        return DSparkModel(config)
    raise ValueError(f"unknown method: {method}")


def _validate_cache_and_io(
    cache: PackedHiddenDataset,
    target_io: FrozenTargetIO,
    *,
    allow_smoke_unverified: bool,
) -> None:
    if not allow_smoke_unverified:
        validate_cache_io_compatibility(cache.manifest, target_io.manifest)
        return
    if cache.manifest.get("status") != "smoke_unverified":
        raise ValueError("smoke opt-in only applies to a smoke_unverified cache")
    if cache.manifest.get("production_eligible") is not False:
        raise ValueError("smoke cache must remain production_eligible=false")
    spec = cache.manifest.get("spec", {})
    provenance = cache.manifest.get("provenance", {})
    if int(spec.get("hidden_size", -1)) != int(target_io.manifest["hidden_size"]):
        raise ValueError("smoke cache width differs from target I/O")
    if int(provenance.get("vocab_size", -1)) != int(target_io.manifest["vocab_size"]):
        raise ValueError("smoke cache vocabulary differs from target I/O")
    if provenance.get("model_fingerprint") != target_io.manifest.get(
        "source_model_fingerprint"
    ):
        raise ValueError("smoke cache model fingerprint differs from target I/O")
    bound = target_io.manifest.get("hidden_cache_identity")
    if bound not in (None, cache.cache_identity):
        raise ValueError("target I/O is bound to another hidden cache")


def _epoch_indices(
    length: int, *, epoch: int, rank: int, world_size: int, seed: int
) -> list[int]:
    return [
        index
        for index, _ in _epoch_assignments(
            tuple(range(length)),
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            seed=seed,
        )
    ]


def _epoch_assignments(
    indices: Sequence[int],
    *,
    epoch: int,
    rank: int,
    world_size: int,
    seed: int,
) -> list[tuple[int, bool]]:
    """Equalize rank work while marking repeated FSDP padding as zero-weight."""

    if not indices:
        raise ValueError("training split is empty")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank/world size")
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(epoch))
    permutation = torch.randperm(len(indices), generator=generator).tolist()
    shuffled = [(int(indices[position]), True) for position in permutation]
    total = math.ceil(len(shuffled) / world_size) * world_size
    padding = total - len(shuffled)
    if padding:
        shuffled.extend(
            (shuffled[position % len(shuffled)][0], False)
            for position in range(padding)
        )
    return shuffled[rank:total:world_size]


def _distributed_assignments(
    indices: Sequence[int], *, rank: int, world_size: int
) -> list[tuple[int, bool]]:
    """Shard fixed validation rows without counting collective padding twice."""

    if not indices:
        return []
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank/world size")
    assigned = [(int(index), True) for index in indices]
    total = math.ceil(len(assigned) / world_size) * world_size
    assigned.extend(
        (int(indices[position % len(indices)]), False)
        for position in range(total - len(assigned))
    )
    return assigned[rank:total:world_size]


def _row_generation_route(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("hidden-cache row has no metadata mapping")
    route = metadata.get("generation_route")
    if not route:
        source = metadata.get("source_metadata")
        if isinstance(source, Mapping):
            route = source.get("route")
    if not isinstance(route, str) or not route:
        raise ValueError("hidden-cache row has no canonical generation route")
    return route


def _dataset_split(
    rows: Sequence[Mapping[str, Any]], *, validation_samples: int, seed: int
) -> DatasetSplit:
    """Create a row-order-independent, disjoint fixed held-out split."""

    if not rows:
        raise ValueError("hidden cache is empty")
    if validation_samples < 0:
        raise ValueError("validation_samples cannot be negative")
    records: list[tuple[int, str, str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            raise ValueError(f"duplicate or empty hidden-cache sample ID: {sample_id!r}")
        seen.add(sample_id)
        route = _row_generation_route(row)
        key = hashlib.sha256(
            f"{int(seed)}\0heldout\0{sample_id}".encode("utf-8")
        ).hexdigest()
        records.append((index, sample_id, route, key))
    # One percent is large enough to track generalization on the 630k corpus,
    # while the fixed cap keeps validation cost bounded. Tiny smoke caches must
    # not accidentally reserve almost every row for validation.
    proportional = math.ceil(len(records) * 0.01)
    count = min(int(validation_samples), proportional, max(0, len(records) - 1))
    heldout_ids = {
        sample_id
        for _, sample_id, _, _ in sorted(records, key=lambda item: (item[3], item[1]))[:count]
    }
    train = [record for record in records if record[1] not in heldout_ids]
    validation = [record for record in records if record[1] in heldout_ids]
    train_ids = tuple(sorted(record[1] for record in train))
    validation_ids = tuple(sorted(record[1] for record in validation))
    train_routes = dict(sorted(Counter(record[2] for record in train).items()))
    validation_routes = dict(
        sorted(Counter(record[2] for record in validation).items())
    )
    identity_payload = {
        "schema": "glm53-heldout-split-v1",
        "seed": int(seed),
        "train_sample_ids": train_ids,
        "validation_sample_ids": validation_ids,
        "train_route_counts": train_routes,
        "validation_route_counts": validation_routes,
    }
    return DatasetSplit(
        train_indices=tuple(record[0] for record in train),
        validation_indices=tuple(record[0] for record in validation),
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        train_route_counts=train_routes,
        validation_route_counts=validation_routes,
        identity=semantic_fingerprint(identity_payload),
        seed=int(seed),
        requested_validation_samples=int(validation_samples),
    )


def _is_optimizer_boundary(
    accumulated_micro_steps: int,
    *,
    epoch: int,
    cursor: int,
    samples_per_epoch: int,
    epochs: int,
    accumulation: int,
) -> bool:
    if accumulated_micro_steps >= accumulation:
        return True
    return epoch + 1 == epochs and cursor + 1 == samples_per_epoch


def _remaining_epochs(
    progress: TrainingProgress, *, total_steps: int, epochs: int
) -> range:
    if progress.global_step >= total_steps:
        return range(0)
    return range(progress.epoch, epochs)


def training_semantics(
    *,
    cache_identity: str,
    target_io_weights_sha256: str,
    target_io_artifact_identity: str,
    target_tokenizer_fingerprint: str,
    config: DraftModelConfig,
    recipe: TrainingRecipe,
    mask_token_id: int,
    mask_token: str,
    vocab_chunk_size: int,
    max_samples: int | None,
    max_steps: int | None,
    total_steps: int,
    production_eligible_input: bool,
    data_split: Mapping[str, Any] | None = None,
    training_window_tokens: int = 4096,
) -> dict[str, Any]:
    return _json_value(
        {
            "schema": "glm53-offline-training-semantics-v4",
            "cache_identity": cache_identity,
            "target_io": {
                "artifact_identity": target_io_artifact_identity,
                "weights_sha256": target_io_weights_sha256,
                "tokenizer_fingerprint": target_tokenizer_fingerprint,
            },
            "model": asdict(config),
            "recipe": asdict(recipe),
            "method_layout": {
                "external_physical_block_size": recipe.block_size,
                "internal_query_tokens": (
                    recipe.block_size - 1
                    if recipe.method == "dspark"
                    else recipe.block_size
                ),
                "proposal_tokens": recipe.block_size - 1,
            },
            "mask_token": {"token": str(mask_token), "token_id": int(mask_token_id)},
            "vocab_chunk_size": int(vocab_chunk_size),
            "max_samples": max_samples,
            "max_steps": max_steps,
            "total_steps": int(total_steps),
            "data_split": dict(data_split or {"schema": "unspecified-test-split"}),
            "training_window": {
                "tokens": int(training_window_tokens),
                "selection": "stable-cycling-assistant-containing-contiguous",
                "anchor_selection": "stable-permutation-cycling-buckets",
                "absolute_position_ids": True,
            },
            "production_eligible_input": bool(production_eligible_input),
        }
    )


def validate_training_mask_identity(
    target_io_manifest: Mapping[str, Any], requested_token_id: int
) -> dict[str, Any]:
    """Bind the runtime mask argument to the frozen tokenizer identity."""

    if target_io_manifest.get("schema") != TARGET_IO_SCHEMA:
        raise ValueError("target-I/O mask identity requires frozen schema v3")
    mask = target_io_manifest.get("mask_token")
    if (
        not isinstance(mask, Mapping)
        or mask.get("token") != "[MASK]"
        or mask.get("special") is not True
    ):
        raise ValueError("target-I/O manifest has no exact special [MASK] identity")
    token_id = mask.get("token_id")
    if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
        raise ValueError("target-I/O [MASK] token ID is invalid")
    if requested_token_id != token_id:
        raise ValueError(
            f"requested mask token ID {requested_token_id} differs from frozen {token_id}"
        )
    tokenizer_identity = target_io_manifest.get("tokenizer_fingerprint")
    if not isinstance(tokenizer_identity, str) or not tokenizer_identity:
        raise ValueError("target-I/O tokenizer fingerprint is missing")
    return {
        "token": "[MASK]",
        "token_id": token_id,
        "tokenizer_fingerprint": tokenizer_identity,
    }


def _training_batch(
    row: Mapping[str, Any],
    *,
    epoch: int,
    recipe: TrainingRecipe,
    mask_token_id: int,
    device: torch.device,
    include: bool = True,
) -> TrainingBatch:
    window = select_training_window(
        row["loss_mask"],
        sample_id=str(row["sample_id"]),
        epoch=epoch,
        block_size=recipe.block_size,
        max_tokens=4096,
        seed=recipe.seed,
    )
    input_ids = row["input_ids"][window.start:window.end]
    loss_mask = row["loss_mask"][window.start:window.end]
    auxiliary_context = row["aux_hidden_states"][window.start:window.end]
    final_context = row["target_final_hidden"][window.start:window.end]
    anchors = sample_anchor_positions(
        loss_mask,
        sample_id=str(row["sample_id"]),
        epoch=epoch,
        block_size=recipe.block_size,
        count=recipe.anchors_per_sample,
        seed=recipe.seed,
    )
    blocks = build_physical_blocks(
        input_ids,
        auxiliary_context,
        anchors,
        block_size=recipe.block_size,
        mask_token_id=mask_token_id,
        target_final_hidden=final_context,
        absolute_position_offset=window.start,
    )
    assert blocks.target_final_hidden is not None
    assert blocks.full_position_ids is not None
    assert blocks.attention_mask is not None
    return TrainingBatch(
        input_ids=blocks.input_ids.unsqueeze(0).to(device),
        target_ids=blocks.target_ids.unsqueeze(0).to(device),
        position_ids=blocks.full_position_ids.unsqueeze(0).to(device),
        auxiliary_hidden=blocks.auxiliary_hidden.unsqueeze(0).to(device),
        target_final_hidden=blocks.target_final_hidden.unsqueeze(0).to(device),
        keep_mask=(
            blocks.keep_mask if include else torch.zeros_like(blocks.keep_mask)
        ).unsqueeze(0).to(device),
        attention_mask=blocks.attention_mask.unsqueeze(0).to(device),
    )


def _accumulate_additive_metrics(
    accumulator: dict[str, tuple[torch.Tensor, torch.Tensor]],
    metrics: Mapping[str, Any],
) -> None:
    for name, scalar in metrics.items():
        numerator = scalar.numerator.detach().float()
        denominator = scalar.denominator.detach().float()
        if name in accumulator:
            old_numerator, old_denominator = accumulator[name]
            accumulator[name] = (
                old_numerator + numerator,
                old_denominator + denominator,
            )
        else:
            accumulator[name] = (numerator.clone(), denominator.clone())


def _reduce_additive_accumulator(
    accumulator: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    flattened: dict[str, torch.Tensor] = {}
    for name, (numerator, denominator) in accumulator.items():
        flattened[f"{name}.numerator"] = numerator
        flattened[f"{name}.denominator"] = denominator
    reduced = reduce_additive_metrics(flattened)
    result: dict[str, dict[str, float]] = {}
    for name in sorted(accumulator):
        numerator = float(reduced[f"{name}.numerator"].cpu())
        denominator = float(reduced[f"{name}.denominator"].cpu())
        result[name] = {
            "mean": numerator / denominator if denominator > 0 else 0.0,
            "numerator": numerator,
            "denominator": denominator,
        }
    return result


def _evaluate_holdout(
    *,
    trainer: OfflineMethodTrainer,
    cache: PackedHiddenDataset,
    assignments: Sequence[tuple[int, bool]],
    recipe: TrainingRecipe,
    mask_token_id: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    if not assignments:
        return {}
    was_training = trainer.model.training
    trainer.model.eval()
    accumulator: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    try:
        with torch.no_grad():
            for index, include in assignments:
                batch = _training_batch(
                    cache[index],
                    epoch=0,
                    recipe=recipe,
                    mask_token_id=mask_token_id,
                    device=device,
                    include=include,
                )
                result = trainer.compute_loss(batch)
                _accumulate_additive_metrics(accumulator, result.metrics)
        return _reduce_additive_accumulator(accumulator)
    finally:
        trainer.model.train(was_training)


def _tree_identity(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_semantics(path: Path, semantics: Mapping[str, Any]) -> None:
    encoded = _canonical_json(semantics) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != semantics:
            raise ValueError("output semantic configuration differs from this run")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _seed_everything(seed: int, rank: int) -> torch.Generator:
    actual = int(seed) + int(rank)
    random.seed(actual)
    np.random.seed(actual)
    torch.manual_seed(actual)
    return torch.Generator(device="cpu").manual_seed(actual)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    recipe = _validate_args(args)
    context = initialize_distributed(args.device)
    try:
        anchor_generator = _seed_everything(recipe.seed, context.rank)
        cache = PackedHiddenDataset(
            args.hidden_cache,
            allow_smoke_unverified=args.allow_smoke_unverified,
            verify_checksums=True,
        )
        target_io = load_frozen_target_io(args.target_io)
        _validate_cache_and_io(
            cache,
            target_io,
            allow_smoke_unverified=args.allow_smoke_unverified,
        )
        sample_count = len(cache)
        if args.max_samples is not None:
            sample_count = min(sample_count, args.max_samples)
        if sample_count < 1:
            raise ValueError("hidden cache has no samples to train")
        split = _dataset_split(
            cache.rows[:sample_count],
            validation_samples=recipe.validation_samples,
            seed=recipe.seed,
        )

        config = DraftModelConfig.production()
        dtype = torch.bfloat16
        model = _build_model(config, recipe.method).to(context.device, dtype=dtype)
        model = apply_fsdp2(model, enabled=True, dtype=dtype)
        target_io.embed_tokens.to(context.device)
        target_io.lm_head.to(context.device)
        trainer = OfflineMethodTrainer(
            method=recipe.method,
            block_size=recipe.block_size,
            model=model,
            target_embedding=target_io.embed_tokens,
            target_lm_head=target_io.lm_head,
            vocab_chunk_size=args.vocab_chunk_size,
        )
        mask_identity = validate_training_mask_identity(
            target_io.manifest, args.mask_token_id
        )
        optimizer = torch.optim.AdamW(
            trainer.trainable_parameters(),
            lr=recipe.learning_rate,
            betas=recipe.betas,
            weight_decay=recipe.weight_decay,
        )
        local_samples = math.ceil(len(split.train_indices) / context.world_size)
        total_steps = math.ceil(
            recipe.epochs * local_samples / recipe.gradient_accumulation
        )
        if args.max_steps is not None:
            total_steps = min(total_steps, args.max_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: cosine_warmup_multiplier(
                step,
                total_steps=total_steps,
                warmup_steps=recipe.warmup_steps,
                min_ratio=recipe.min_lr_ratio,
            ),
        )
        semantics = training_semantics(
            cache_identity=cache.cache_identity,
            target_io_weights_sha256=target_io.manifest["weights_sha256"],
            target_io_artifact_identity=target_io.manifest["artifact_identity"],
            target_tokenizer_fingerprint=mask_identity["tokenizer_fingerprint"],
            config=config,
            recipe=recipe,
            mask_token_id=args.mask_token_id,
            mask_token=mask_identity["token"],
            vocab_chunk_size=args.vocab_chunk_size,
            max_samples=args.max_samples,
            max_steps=args.max_steps,
            total_steps=total_steps,
            production_eligible_input=bool(
                cache.manifest.get("production_eligible")
            ),
            data_split=split.manifest(),
        )
        if context.is_main:
            _atomic_semantics(args.output_dir / "semantic_config.json", semantics)
        if dist.is_initialized():
            dist.barrier()

        progress = TrainingProgress(0, 0, 0, 0)
        if args.resume is not None:
            progress = load_training_checkpoint(
                args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                anchor_generator=anchor_generator,
                expected_semantic_config=semantics,
            )
        metrics_logger = (
            StepJsonlLogger(args.output_dir / "optimizer_metrics.jsonl")
            if context.is_main
            else None
        )
        if metrics_logger is not None:
            metrics_logger.reconcile(progress.global_step)
        capability_path = args.output_dir / "candidate-capability.json"
        if progress.global_step >= total_steps and capability_path.is_file():
            return validate_candidate_capability(
                json.loads(capability_path.read_text(encoding="utf-8"))
            )

        optimizer.zero_grad(set_to_none=True)
        accumulated_denominator = torch.zeros(
            (), device=context.device, dtype=torch.float32
        )
        accumulated_micro_steps = 0
        accumulated_metrics: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        global_step = progress.global_step
        validation_assignments = _distributed_assignments(
            split.validation_indices,
            rank=context.rank,
            world_size=context.world_size,
        )
        for epoch in _remaining_epochs(
            progress, total_steps=total_steps, epochs=recipe.epochs
        ):
            assignments = _epoch_assignments(
                split.train_indices,
                epoch=epoch,
                rank=context.rank,
                world_size=context.world_size,
                seed=recipe.seed,
            )
            cursor_start = progress.sampler_cursor if epoch == progress.epoch else 0
            for cursor in range(cursor_start, len(assignments)):
                index, include = assignments[cursor]
                row = cache[index]
                batch = _training_batch(
                    row,
                    epoch=epoch,
                    recipe=recipe,
                    mask_token_id=args.mask_token_id,
                    device=context.device,
                    include=include,
                )
                is_epoch_end = cursor + 1 == len(assignments)
                boundary = _is_optimizer_boundary(
                    accumulated_micro_steps + 1,
                    epoch=epoch,
                    cursor=cursor,
                    samples_per_epoch=len(assignments),
                    epochs=recipe.epochs,
                    accumulation=recipe.gradient_accumulation,
                )
                configure_accumulation(model, synchronize=boundary)
                result = trainer.compute_loss(batch)
                total = result.metrics["total"]
                scaled_numerator, global_denominator = (
                    scale_additive_loss_for_accumulation(
                        total.numerator, total.denominator
                    )
                )
                scaled_numerator.backward()
                accumulated_denominator += global_denominator
                _accumulate_additive_metrics(accumulated_metrics, result.metrics)
                accumulated_micro_steps += 1
                if not boundary:
                    continue

                next_progress = TrainingProgress(
                    global_step=global_step,
                    micro_step=0,
                    epoch=epoch + 1 if is_epoch_end else epoch,
                    sampler_cursor=0 if is_epoch_end else cursor + 1,
                )
                did_step = bool(accumulated_denominator > 0)
                if did_step:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated_denominator)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), recipe.gradient_clip
                    )
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                reduced_training_metrics = _reduce_additive_accumulator(
                    accumulated_metrics
                )
                optimizer.zero_grad(set_to_none=True)
                accumulated_denominator.zero_()
                accumulated_micro_steps = 0
                accumulated_metrics = {}
                next_progress = TrainingProgress(
                    global_step=global_step,
                    micro_step=0,
                    epoch=epoch + 1 if is_epoch_end else epoch,
                    sampler_cursor=0 if is_epoch_end else cursor + 1,
                )
                if did_step and metrics_logger is not None:
                    metrics_logger.append(
                        {
                            "schema": "glm53-optimizer-metrics-v1",
                            "split": "train",
                            "step": global_step,
                            "epoch": next_progress.epoch,
                            "sampler_cursor": next_progress.sampler_cursor,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "metrics": reduced_training_metrics,
                        }
                    )
                if did_step and global_step % args.checkpoint_every == 0:
                    save_training_checkpoint(
                        args.output_dir / "checkpoints" / f"step-{global_step:08d}",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        progress=next_progress,
                        anchor_generator=anchor_generator,
                        at_optimizer_boundary=True,
                        semantic_config=semantics,
                    )
                if (
                    did_step
                    and validation_assignments
                    and global_step % recipe.validation_every == 0
                ):
                    validation_metrics = _evaluate_holdout(
                        trainer=trainer,
                        cache=cache,
                        assignments=validation_assignments,
                        recipe=recipe,
                        mask_token_id=args.mask_token_id,
                        device=context.device,
                    )
                    if metrics_logger is not None:
                        metrics_logger.append(
                            {
                                "schema": "glm53-optimizer-metrics-v1",
                                "split": "validation",
                                "step": global_step,
                                "fixed_window_epoch": 0,
                                "split_identity": split.identity,
                                "metrics": validation_metrics,
                            }
                        )
                if global_step >= total_steps:
                    progress = next_progress
                    break
                progress = next_progress
            if global_step >= total_steps:
                break

        final_checkpoint = (
            args.output_dir / "checkpoints" / f"final-step-{global_step:08d}"
        )
        if not (final_checkpoint / "COMPLETE").is_file():
            save_training_checkpoint(
                final_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                progress=progress,
                anchor_generator=anchor_generator,
                at_optimizer_boundary=True,
                semantic_config=semantics,
            )
        if validation_assignments and global_step % recipe.validation_every != 0:
            validation_metrics = _evaluate_holdout(
                trainer=trainer,
                cache=cache,
                assignments=validation_assignments,
                recipe=recipe,
                mask_token_id=args.mask_token_id,
                device=context.device,
            )
            if metrics_logger is not None:
                metrics_logger.append(
                    {
                        "schema": "glm53-optimizer-metrics-v1",
                        "split": "validation",
                        "step": global_step,
                        "fixed_window_epoch": 0,
                        "split_identity": split.identity,
                        "metrics": validation_metrics,
                    }
                )
        if dist.is_initialized():
            dist.barrier()
        record: dict[str, Any] = {}
        if context.is_main:
            record = write_candidate_capability(
                capability_path,
                artifact_identity=_tree_identity(final_checkpoint),
            )
        if dist.is_initialized():
            dist.barrier()
        return record
    finally:
        shutdown_distributed()


def run_tiny_smoke(*, method: str, block_size: int) -> dict[str, Any]:
    recipe_for(method, block_size)
    torch.manual_seed(123)
    config = DraftModelConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_aux_layers=2,
        vocab_size=17,
    )
    if method == "dflash2":
        model: nn.Module = DFlash2Model(
            config,
            convolution_group_size=4,
            selector_rank=4,
            selector_top_k=4,
        )
    elif method == "dspark":
        model = DSparkModel(config, markov_rank=4)
    else:
        model = DFlashModel(config)
    embedding = nn.Embedding(config.vocab_size, config.hidden_size)
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    trainer = OfflineMethodTrainer(
        method=method,
        block_size=block_size,
        model=model,
        target_embedding=embedding,
        target_lm_head=lm_head,
        vocab_chunk_size=5,
    )
    optimizer = torch.optim.AdamW(trainer.trainable_parameters(), lr=1e-3)
    anchors, depth, context = 1, block_size, 4
    queries = anchors * depth
    batch = TrainingBatch(
        input_ids=torch.randint(0, config.vocab_size, (1, anchors, depth)),
        target_ids=torch.randint(0, config.vocab_size, (1, anchors, depth)),
        position_ids=torch.arange(context + queries).view(1, -1),
        auxiliary_hidden=torch.randn(
            1, context, config.num_aux_layers, config.hidden_size
        ),
        target_final_hidden=torch.randn(1, anchors, depth, config.hidden_size),
        keep_mask=torch.ones(1, anchors, dtype=torch.bool),
        attention_mask=torch.ones(
            1, 1, queries, context + queries, dtype=torch.bool
        ),
    )
    result = trainer.compute_loss(batch)
    result.loss.backward()
    optimizer.step()
    return {
        "method": method,
        "block_size": block_size,
        "finite_loss": bool(torch.isfinite(result.loss)),
        "optimizer_steps": 1,
        "runtime_attested": False,
        "deployable_export": False,
    }


def main() -> int:
    args = parse_args()
    record = run_training(args)
    if record:
        print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
