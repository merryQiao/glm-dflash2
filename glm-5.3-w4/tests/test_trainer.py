from __future__ import annotations

import pytest
import torch

from glm53_w4.blocks import build_sliding_blocks
from glm53_w4.dflash2 import DFlash2Model
from glm53_w4.dspark import DSparkModel
from glm53_w4.modeling import DraftModelConfig
from glm53_w4.target_io import FrozenTargetIO
from glm53_w4.trainer import (
    OfflineDrafterTrainer,
    compute_dflash2_objective,
    compute_dspark_objective,
    accumulation_real_count,
    rank_loss_scale,
    recipe_for,
)


def test_partial_accumulation_uses_real_examples_not_padding() -> None:
    assert rank_loss_scale(include=True, world_size=4) == 4.0
    assert rank_loss_scale(include=False, world_size=4) == 0.0
    assert accumulation_real_count([4, 4, 1, 0]) == 9
    with pytest.raises(ValueError, match="at least one real"):
        accumulation_real_count([0, 0])


def test_method_recipes_are_fixed() -> None:
    dspark = recipe_for("dspark", 8)
    assert (dspark.learning_rate, dspark.epochs, dspark.gamma) == (6e-4, 3, 4.0)
    assert (dspark.ce_weight, dspark.tv_weight, dspark.confidence_weight) == (
        0.1,
        0.9,
        1.0,
    )
    assert recipe_for("dflash2", 8).gamma == 4.0
    assert recipe_for("dflash2", 16).gamma == 7.0


def test_dflash2_selector_loss_is_hit_only() -> None:
    base_nll = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    candidates = torch.tensor([[[[2, 3], [4, 5]]]])
    scores = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], requires_grad=True)
    targets = torch.tensor([[[1, 3, 9]]])
    mask = torch.ones((1, 1, 2), dtype=torch.bool)
    loss = compute_dflash2_objective(
        base_nll=base_nll,
        candidate_ids=candidates,
        selector_scores=scores,
        target_ids=targets,
        prediction_mask=mask,
        gamma=4.0,
    )
    assert loss.candidate_hits.item() == 1
    loss.total.backward()
    assert scores.grad is not None
    assert torch.equal(scores.grad[0, 0, 1], torch.zeros(2))


def test_dflash2_base_and_hit_only_selector_are_independently_normalized() -> None:
    base_nll = torch.tensor([[[2.0, 4.0]]])
    candidates = torch.tensor([[[[3, 2], [4, 5]]]])
    scores = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    targets = torch.tensor([[[1, 3, 9]]])
    mask = torch.ones((1, 1, 2), dtype=torch.bool)
    terms = compute_dflash2_objective(
        base_nll=base_nll,
        candidate_ids=candidates,
        selector_scores=scores,
        target_ids=targets,
        prediction_mask=mask,
        gamma=4.0,
    )
    # Only depth 0 hits. Its selector CE is log(2), independent of the two-token
    # denominator used by the full-vocabulary base CE.
    assert torch.allclose(terms.total, terms.base + torch.log(torch.tensor(2.0)))


def test_dspark_exact_tv_and_confidence_are_finite() -> None:
    draft = torch.randn(1, 1, 2, 4, requires_grad=True)
    teacher = torch.randn(1, 1, 2, 4)
    ids = torch.tensor([[[1, 2]]])
    predecessors = torch.tensor([[[0, 1]]])
    confidence = torch.randn(1, 1, 2, requires_grad=True)
    lm_head = torch.randn(7, 4)
    markov_w1 = torch.randn(7, 3, requires_grad=True)
    markov_w2 = torch.randn(7, 3, requires_grad=True)
    terms = compute_dspark_objective(
        draft_hidden=draft,
        target_hidden=teacher,
        target_ids=ids,
        predecessor_ids=predecessors,
        confidence_logits=confidence,
        lm_head_weight=lm_head,
        markov_w1=markov_w1,
        markov_w2=markov_w2,
        prediction_mask=torch.ones_like(ids, dtype=torch.bool),
        gamma=4.0,
        vocab_chunk_size=3,
    )
    assert torch.isfinite(terms.total)
    assert bool(((terms.confidence_target >= 0) & (terms.confidence_target <= 1)).all())
    terms.total.backward()
    assert draft.grad is not None and confidence.grad is not None


def _config(method: str) -> DraftModelConfig:
    return DraftModelConfig(
        vocab_size=24,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        target_layer_ids=(1,),
        block_size=8,
        mask_token_id=23,
        sliding_window=4,
        selector_top_k=4,
        selector_rank=4,
        markov_rank=4,
        conv_group_size=4,
        anchor_chunk_size=2,
    )


def _io() -> FrozenTargetIO:
    embedding = torch.nn.Embedding(24, 16)
    head = torch.nn.Linear(16, 24, bias=False)
    embedding.weight.requires_grad_(False)
    head.weight.requires_grad_(False)
    return FrozenTargetIO(embedding, head, {"source_model_fingerprint": "x"})


def test_end_to_end_tiny_dflash2_step() -> None:
    config = _config("dflash2")
    model = DFlash2Model(config)
    trainer = OfflineDrafterTrainer(model, _io(), method="dflash2", gamma=4.0)
    ids = torch.randint(0, 20, (1, 16))
    batch = {
        "input_ids": ids,
        "loss_mask": torch.ones_like(ids, dtype=torch.bool),
        "aux_hidden_states": torch.randn(1, 16, 1, 16),
        "target_final_hidden": torch.randn(1, 16, 16),
        "sample_ids": ["a"],
    }
    output = trainer(
        batch,
        epoch=0,
        anchor_positions=torch.tensor([[2, 7]]),
        block_keep_mask=torch.tensor([[True, True]]),
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_dspark_tail_anchor_masks_padding_without_out_of_bounds_gather() -> None:
    config = _config("dspark")
    model = DSparkModel(config)
    trainer = OfflineDrafterTrainer(model, _io(), method="dspark", gamma=4.0)
    ids = torch.randint(0, 20, (1, 10))
    batch = {
        "input_ids": ids,
        "loss_mask": torch.ones_like(ids, dtype=torch.bool),
        "aux_hidden_states": torch.randn(1, 10, 1, 16),
        "target_final_hidden": torch.randn(1, 10, 16),
        "sample_ids": ["tail"],
    }
    output = trainer(
        batch,
        epoch=0,
        anchor_positions=torch.tensor([[8]]),
        block_keep_mask=torch.tensor([[True]]),
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()
