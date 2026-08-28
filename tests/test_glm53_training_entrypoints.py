from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch
from torch import nn
import tools.train_drafter_offline as training_entrypoint

from glm53_drafters.contracts import METHOD_BLOCK_SIZES
from tools.train_drafter_offline import (
    TrainingRecipe,
    StepJsonlLogger,
    _dataset_split,
    _distributed_assignments,
    _epoch_assignments,
    _epoch_indices,
    _is_optimizer_boundary,
    _remaining_epochs,
    cosine_warmup_multiplier,
    parse_args,
    recipe_for,
    run_tiny_smoke,
    semantic_fingerprint,
    training_semantics,
)
from glm53_drafters.checkpointing import TrainingProgress
from glm53_drafters.chunked_lm_head import AdditiveScalar, additive_mean
from glm53_drafters.modeling_common import DraftModelConfig
from glm53_drafters.offline_trainer import TrainingResult


ROOT = Path(__file__).resolve().parents[1]


def test_training_mask_must_match_frozen_target_io_identity() -> None:
    manifest = {
        "schema": "glm53-target-io-v3",
        "artifact_identity": "target-io-a",
        "tokenizer_fingerprint": "tokenizer-a",
        "mask_token": {"token": "[MASK]", "token_id": 154821, "special": True},
    }
    assert training_entrypoint.validate_training_mask_identity(manifest, 154821) == {
        "token": "[MASK]",
        "token_id": 154821,
        "tokenizer_fingerprint": "tokenizer-a",
    }
    with pytest.raises(ValueError, match="154821|differs"):
        training_entrypoint.validate_training_mask_identity(manifest, 154822)
    with pytest.raises(ValueError, match="mask|schema"):
        training_entrypoint.validate_training_mask_identity(
            {**manifest, "schema": "glm53-target-io-v2"}, 154821
        )


def test_exact_production_recipes_and_method_block_matrix() -> None:
    assert dict(METHOD_BLOCK_SIZES) == {
        "dflash": (8, 16),
        "dflash2": (8, 16),
        "dspark": (8,),
    }
    for method, blocks in METHOD_BLOCK_SIZES.items():
        for block_size in blocks:
            recipe = recipe_for(method, block_size)
            assert recipe == TrainingRecipe(
                method=method,
                block_size=block_size,
                gamma=7.0 if block_size == 16 else 4.0,
                anchors_per_sample=64,
                epochs=3,
                learning_rate=6e-4,
                betas=(0.9, 0.95),
                weight_decay=0.0,
                warmup_steps=1000,
                min_lr_ratio=0.1,
                per_rank_batch=1,
                gradient_accumulation=8,
                gradient_clip=1.0,
                seed=42,
                validation_samples=128,
                validation_every=1000,
            )
    with pytest.raises(ValueError, match="DSpark.*8"):
        recipe_for("dspark", 16)


def test_cli_defaults_are_npu_hccl_fsdp2_bf16() -> None:
    args = parse_args(
        [
            "--method",
            "dflash2",
            "--block-size",
            "16",
            "--hidden-cache",
            "/cache",
            "--target-io",
            "/io",
            "--output-dir",
            "/output",
        ]
    )
    assert args.device == "npu"
    assert args.backend == "hccl"
    assert args.strategy == "fsdp2"
    assert args.dtype == "bfloat16"
    assert args.per_rank_batch == 1
    assert args.gradient_accumulation == 8


def test_scheduler_and_semantic_fingerprint_are_resume_stable() -> None:
    assert cosine_warmup_multiplier(0, total_steps=2000, warmup_steps=1000) == 0.0
    assert cosine_warmup_multiplier(1000, total_steps=2000, warmup_steps=1000) == 1.0
    assert cosine_warmup_multiplier(2000, total_steps=2000, warmup_steps=1000) == pytest.approx(0.1)
    first = semantic_fingerprint({"method": "dflash", "block_size": 8})
    second = semantic_fingerprint({"block_size": 8, "method": "dflash"})
    changed = semantic_fingerprint({"method": "dflash", "block_size": 16})
    assert first == second
    assert first != changed


def test_gradient_accumulation_does_not_reset_at_epoch_boundaries() -> None:
    boundaries = []
    accumulated = 0
    for epoch in range(3):
        for cursor in range(3):
            accumulated += 1
            if _is_optimizer_boundary(
                accumulated,
                epoch=epoch,
                cursor=cursor,
                samples_per_epoch=3,
                epochs=3,
                accumulation=8,
            ):
                boundaries.append((epoch, cursor, accumulated))
                accumulated = 0
    assert boundaries == [(2, 1, 8), (2, 2, 1)]


def test_distributed_epoch_padding_gives_every_rank_equal_work() -> None:
    one_sample = [
        _epoch_indices(1, epoch=0, rank=rank, world_size=8, seed=42)
        for rank in range(8)
    ]
    assert one_sample == [[0]] * 8
    three_samples = [
        _epoch_indices(3, epoch=0, rank=rank, world_size=8, seed=42)
        for rank in range(8)
    ]
    assert all(len(indices) == 1 for indices in three_samples)
    assert {indices[0] for indices in three_samples} == {0, 1, 2}


def test_distributed_assignments_mark_padding_without_biasing_samples() -> None:
    source = [10, 11, 12]
    assignments = [
        _epoch_assignments(source, epoch=0, rank=rank, world_size=8, seed=42)
        for rank in range(8)
    ]
    assert all(len(items) == 1 for items in assignments)
    real = [index for items in assignments for index, include in items if include]
    assert sorted(real) == source
    assert sum(include for items in assignments for _, include in items) == 3
    fixed = [
        _distributed_assignments(source, rank=rank, world_size=8)
        for rank in range(8)
    ]
    assert sorted(index for items in fixed for index, include in items if include) == source


def test_heldout_split_is_disjoint_deterministic_route_auditable_and_tiny_safe() -> None:
    rows = [
        {
            "sample_id": f"sample-{index}",
            "metadata": {
                "generation_route": "original_trajectory" if index % 3 == 0 else "workspace_task"
            },
        }
        for index in range(500)
    ]
    first = _dataset_split(rows, validation_samples=128, seed=42)
    second = _dataset_split(list(reversed(rows)), validation_samples=128, seed=42)
    assert len(first.validation_indices) == 5
    assert set(first.train_sample_ids).isdisjoint(first.validation_sample_ids)
    assert set(first.train_sample_ids) | set(first.validation_sample_ids) == {
        row["sample_id"] for row in rows
    }
    assert first.identity == second.identity
    assert first.validation_sample_ids == second.validation_sample_ids
    assert sum(first.validation_route_counts.values()) == 5
    assert set(first.validation_route_counts) == {
        "original_trajectory",
        "workspace_task",
    }
    singleton = _dataset_split(rows[:1], validation_samples=128, seed=42)
    assert singleton.train_indices == (0,)
    assert singleton.validation_indices == ()
    pair = _dataset_split(rows[:2], validation_samples=128, seed=42)
    assert len(pair.train_indices) == len(pair.validation_indices) == 1


def test_step_jsonl_logger_reconciles_resume_and_rejects_conflicting_duplicate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = StepJsonlLogger(path)
    row1 = {"split": "train", "step": 1, "metrics": {"total": 2.0}}
    row2 = {"split": "train", "step": 2, "metrics": {"total": 1.5}}
    validation = {"split": "validation", "step": 2, "metrics": {"total": 1.7}}
    assert logger.append(row1) is True
    assert logger.append(row1) is False
    assert logger.append(row2) is True
    assert logger.append(validation) is True
    logger.reconcile(1)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [row1]
    resumed = StepJsonlLogger(path)
    assert resumed.append(row1) is False
    with pytest.raises(ValueError, match="conflicting"):
        resumed.append({**row1, "metrics": {"total": 999.0}})


def test_fixed_holdout_uses_no_grad_ignores_padding_and_restores_train_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1)
    model.train()

    class FakeTrainer:
        def __init__(self) -> None:
            self.model = model

        def compute_loss(self, batch):
            assert torch.is_grad_enabled() is False
            denominator = batch["denominator"]
            numerator = denominator * 2.0
            scalar = AdditiveScalar(
                numerator, denominator, additive_mean(numerator, denominator)
            )
            return TrainingResult(scalar.mean, {"total": scalar})

    monkeypatch.setattr(
        training_entrypoint,
        "_training_batch",
        lambda row, **kwargs: {
            "denominator": torch.tensor(1.0 if kwargs["include"] else 0.0)
        },
    )
    metrics = training_entrypoint._evaluate_holdout(
        trainer=FakeTrainer(),
        cache=[{"sample_id": "a"}],
        assignments=[(0, True), (0, False)],
        recipe=recipe_for("dflash", 8),
        mask_token_id=154821,
        device=torch.device("cpu"),
    )
    assert metrics["total"] == {
        "mean": 2.0,
        "numerator": 2.0,
        "denominator": 1.0,
    }
    assert model.training is True


def test_resume_contract_binds_mask_chunks_and_schedule_and_completed_is_empty() -> None:
    recipe = recipe_for("dflash", 8)
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
    kwargs = {
        "cache_identity": "cache-a",
        "target_io_weights_sha256": "weights-a",
        "target_io_artifact_identity": "target-io-a",
        "target_tokenizer_fingerprint": "tokenizer-a",
        "config": config,
        "recipe": recipe,
        "mask_token_id": 7,
        "mask_token": "[MASK]",
        "vocab_chunk_size": 5,
        "max_samples": None,
        "max_steps": 9,
        "total_steps": 9,
        "production_eligible_input": True,
    }
    baseline = training_semantics(**kwargs)
    assert baseline["schema"] == "glm53-offline-training-semantics-v4"
    assert baseline["target_io"] == {
        "artifact_identity": "target-io-a",
        "weights_sha256": "weights-a",
        "tokenizer_fingerprint": "tokenizer-a",
    }
    assert baseline["mask_token"] == {"token": "[MASK]", "token_id": 7}
    assert baseline["method_layout"] == {
        "external_physical_block_size": 8,
        "internal_query_tokens": 8,
        "proposal_tokens": 7,
    }
    dspark_kwargs = dict(kwargs)
    dspark_kwargs["recipe"] = recipe_for("dspark", 8)
    assert training_semantics(**dspark_kwargs)["method_layout"] == {
        "external_physical_block_size": 8,
        "internal_query_tokens": 7,
        "proposal_tokens": 7,
    }
    for key, changed in (
        ("mask_token_id", 8),
        ("vocab_chunk_size", 7),
        ("max_steps", 8),
        ("total_steps", 8),
        ("target_io_artifact_identity", "target-io-b"),
        ("target_tokenizer_fingerprint", "tokenizer-b"),
        ("mask_token", "[gMASK]"),
    ):
        modified = dict(kwargs)
        modified[key] = changed
        assert training_semantics(**modified) != baseline
    complete = TrainingProgress(global_step=9, micro_step=0, epoch=1, sampler_cursor=2)
    assert list(_remaining_epochs(complete, total_steps=9, epochs=3)) == []


@pytest.mark.parametrize(
    ("method", "block_size"),
    [("dflash", 8), ("dflash", 16), ("dflash2", 8), ("dflash2", 16), ("dspark", 8)],
)
def test_cpu_tiny_smoke_takes_one_finite_optimizer_step(
    method: str, block_size: int
) -> None:
    result = run_tiny_smoke(method=method, block_size=block_size)
    assert result["finite_loss"] is True
    assert result["optimizer_steps"] == 1
    assert result["runtime_attested"] is False
    assert result["deployable_export"] is False


def test_training_launchers_are_valid_and_production_is_ascend_only() -> None:
    for relative in ("scripts/train_drafter.sh", "scripts/smoke_stage_b_training.sh"):
        path = ROOT / relative
        subprocess.run(["bash", "-n", str(path)], check=True)
    production = (ROOT / "scripts/train_drafter.sh").read_text(encoding="utf-8")
    assert "--device npu" in production
    assert "--backend hccl" in production
    assert "--strategy fsdp2" in production
    assert "--dtype bfloat16" in production
    smoke = (ROOT / "scripts/smoke_stage_b_training.sh").read_text(encoding="utf-8")
    assert '${ROOT_DIR}/src:${ROOT_DIR}' in smoke
    combined = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8").lower()
        for relative in (
            "tools/train_drafter_offline.py",
            "scripts/train_drafter.sh",
            "scripts/smoke_stage_b_training.sh",
        )
    )
    for forbidden in ("cuda", "nccl", "flash_attn", "flashattention"):
        assert forbidden not in combined
