from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from glm53_drafters.dflash2_model import DFlash2Model
from glm53_drafters.dflash_model import DFlashModel
from glm53_drafters.dspark_model import (
    DSparkModel,
    LowRankMarkovHead,
    MarkovConfidenceHead,
    align_predecessors,
    teacher_forced_predecessor_ids,
)
from glm53_drafters.modeling_common import DraftModelConfig
from glm53_drafters.offline_trainer import OfflineMethodTrainer, TrainingBatch
import glm53_drafters.offline_trainer as offline_trainer_module
from glm53_drafters.target_io import FrozenLinear


def config() -> DraftModelConfig:
    return DraftModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        num_aux_layers=3,
        vocab_size=23,
    )


def batch(*, valid: bool = True) -> TrainingBatch:
    torch.manual_seed(7)
    context, queries = 5, 2 * 8
    return TrainingBatch(
        input_ids=torch.randint(0, 23, (1, 2, 8)),
        target_ids=torch.randint(0, 23, (1, 2, 8)),
        position_ids=torch.arange(context + queries).view(1, -1),
        auxiliary_hidden=torch.randn(1, context, 3, 16),
        target_final_hidden=torch.randn(1, 2, 8, 16),
        keep_mask=torch.full((1, 2), valid, dtype=torch.bool),
        attention_mask=torch.ones(
            1, 1, queries, context + queries, dtype=torch.bool
        ),
    )


def target_io() -> tuple[nn.Embedding, nn.Linear]:
    embedding = nn.Embedding(23, 16)
    lm_head = nn.Linear(16, 23, bias=False)
    return embedding, lm_head


def test_dspark_rank_interfaces_and_predecessor_alignment() -> None:
    hidden = torch.arange(1 * 2 * 4 * 16, dtype=torch.float32).view(1, 2, 4, 16)
    predecessor = align_predecessors(hidden)
    assert torch.equal(predecessor[:, :, 0], torch.zeros_like(predecessor[:, :, 0]))
    assert torch.equal(predecessor[:, :, 1:], hidden[:, :, :-1])
    markov = LowRankMarkovHead(vocab_size=23, rank=8)
    confidence = MarkovConfidenceHead(hidden_size=16, rank=8)
    predecessor_ids = torch.randint(0, 23, (1, 2, 4))
    markov_features = markov.features(predecessor_ids)
    assert markov(predecessor_ids).shape == (1, 2, 4, 23)
    assert markov_features.shape == (1, 2, 4, 8)
    assert confidence(hidden, markov_features).shape == (1, 2, 4)
    assert markov.markov_w1.weight.std().item() == pytest.approx(0.02, abs=0.005)
    assert markov.markov_w2.weight.std().item() == pytest.approx(0.02, abs=0.005)
    assert torch.count_nonzero(confidence.output.bias) == 0


def test_dspark_markov_predecessor_is_teacher_forced_token_id() -> None:
    torch.manual_seed(19)
    model = DSparkModel(config(), markov_rank=8)
    embeddings = torch.randn(1, 16, 16)
    auxiliary = torch.randn(1, 5, 3, 16)
    targets = torch.randint(0, 23, (1, 2, 8))
    aligned = teacher_forced_predecessor_ids(targets)
    assert torch.equal(aligned[..., 0], targets[..., 0])
    assert torch.equal(aligned[..., 1:], targets[..., :-1])
    predecessors = aligned.flatten(1, 2)
    changed = predecessors.clone()
    changed[:, 1] = (changed[:, 1] + 1) % 23
    positions = torch.arange(21).view(1, -1)
    attention = torch.ones(1, 1, 16, 21, dtype=torch.bool)

    hidden, first_markov, first_confidence = model(
        embeddings, auxiliary, predecessor_token_ids=predecessors,
        position_ids=positions, attention_mask=attention, block_size=8,
    )
    changed_hidden, changed_markov, changed_confidence = model(
        embeddings, auxiliary, predecessor_token_ids=changed,
        position_ids=positions, attention_mask=attention, block_size=8,
    )
    assert torch.equal(hidden, changed_hidden)
    assert torch.equal(first_markov[:, 0], changed_markov[:, 0])
    assert not torch.equal(first_markov[:, 1], changed_markov[:, 1])
    assert not torch.equal(first_confidence[:, 1], changed_confidence[:, 1])


def build_trainer(method: str) -> OfflineMethodTrainer:
    cfg = config()
    if method == "dflash":
        model = DFlashModel(cfg)
    elif method == "dflash2":
        model = DFlash2Model(
            cfg,
            convolution_group_size=4,
            selector_rank=8,
            selector_top_k=5,
        )
    else:
        model = DSparkModel(cfg, markov_rank=8)
    embedding, lm_head = target_io()
    return OfflineMethodTrainer(
        method=method,
        block_size=8,
        model=model,
        target_embedding=embedding,
        target_lm_head=lm_head,
        vocab_chunk_size=7,
    )


@pytest.mark.parametrize("method", ["dflash", "dflash2", "dspark"])
def test_each_method_has_additive_metrics_and_one_finite_cpu_step(method: str) -> None:
    trainer = build_trainer(method)
    target_parameter_ids = {
        id(parameter)
        for module in (trainer.target_embedding, trainer.target_lm_head)
        for parameter in module.parameters()
    }
    trainable = list(trainer.trainable_parameters())
    assert target_parameter_ids.isdisjoint(map(id, trainable))
    assert all(not parameter.requires_grad for module in (
        trainer.target_embedding,
        trainer.target_lm_head,
    ) for parameter in module.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    result = trainer.compute_loss(batch())
    assert torch.isfinite(result.loss)
    assert result.metrics["total"].denominator.item() > 0
    if method == "dflash2":
        assert {"selector", "unary_recall"} <= set(result.metrics)
        assert result.metrics["selector"].denominator <= result.metrics["base"].denominator
        expected = result.metrics["base"].mean + result.metrics["selector"].mean
        torch.testing.assert_close(result.loss, expected)
        torch.testing.assert_close(result.metrics["total"].mean, expected)
        assert result.metrics["total"].denominator.item() == 1.0
    if method == "dspark":
        assert {"ce", "tv", "confidence"} <= set(result.metrics)
    result.loss.backward()
    optimizer.step()


def test_dspark_total_formula_and_denominators_are_locked() -> None:
    trainer = build_trainer("dspark")
    result = trainer.compute_loss(batch())
    metrics = result.metrics
    denominator = metrics["total"].denominator
    assert metrics["ce"].denominator == denominator
    assert metrics["tv"].denominator == denominator
    assert metrics["confidence"].denominator == denominator
    expected = (
        0.1 * metrics["ce"].numerator
        + 0.9 * metrics["tv"].numerator
        + 1.0 * metrics["confidence"].numerator
    )
    torch.testing.assert_close(metrics["total"].numerator, expected)
    torch.testing.assert_close(result.loss, expected / denominator)


def test_zero_valid_tokens_is_differentiable_for_every_method() -> None:
    for method in ("dflash", "dflash2", "dspark"):
        trainer = build_trainer(method)
        result = trainer.compute_loss(batch(valid=False))
        assert result.loss.item() == 0.0
        result.loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in trainer.trainable_parameters()
        )


def test_method_block_matrix_is_enforced() -> None:
    cfg = config()
    embedding, lm_head = target_io()
    with pytest.raises(ValueError, match="DSpark.*8"):
        OfflineMethodTrainer(
            method="dspark",
            block_size=16,
            model=DSparkModel(cfg, markov_rank=8),
            target_embedding=embedding,
            target_lm_head=lm_head,
        )


def test_real_frozen_target_io_head_without_bias_attribute_is_supported() -> None:
    cfg = config()
    embedding = nn.Embedding(23, 16)
    lm_head = FrozenLinear(torch.randn(23, 16))
    trainer = OfflineMethodTrainer(
        method="dflash",
        block_size=8,
        model=DFlashModel(cfg),
        target_embedding=embedding,
        target_lm_head=lm_head,
        vocab_chunk_size=7,
    )
    result = trainer.compute_loss(batch())
    assert torch.isfinite(result.loss)


def test_dspark_teacher_logits_use_causal_predecessor_hidden_sentinel(
    monkeypatch,
) -> None:
    trainer = build_trainer("dspark")
    value = batch()
    sentinel = torch.zeros_like(value.target_final_hidden)
    assert sentinel is not None
    for position in range(sentinel.shape[-2]):
        sentinel[..., position, 0] = 100 + position
    value = TrainingBatch(
        input_ids=value.input_ids,
        target_ids=value.target_ids,
        position_ids=value.position_ids,
        auxiliary_hidden=value.auxiliary_hidden,
        target_final_hidden=sentinel,
        keep_mask=value.keep_mask,
        attention_mask=value.attention_mask,
    )
    captured = {}
    original = offline_trainer_module.exact_total_variation

    def capture(student_logits, target_logits, *, weights=None):
        captured["target_logits"] = target_logits.detach().clone()
        return original(student_logits, target_logits, weights=weights)

    monkeypatch.setattr(offline_trainer_module, "exact_total_variation", capture)
    trainer.compute_loss(value)
    expected = F.linear(
        sentinel[..., :-1, :], trainer.target_lm_head.weight
    ).float()
    torch.testing.assert_close(captured["target_logits"], expected)


def test_external_b8_uses_official_method_specific_query_lengths() -> None:
    observed = {}
    for method, expected in (("dflash", 16), ("dspark", 14)):
        trainer = build_trainer(method)

        def capture(_module, args, _kwargs, *, name=method):
            observed[name] = args[0].shape[1]

        handle = trainer.model.register_forward_pre_hook(capture, with_kwargs=True)
        trainer.compute_loss(batch())
        handle.remove()
    # Two anchors: DFlash runs physical 8 and drops output zero; DSpark runs
    # official internal 7 and supervises every output. Both propose 7 tokens.
    assert observed == {"dflash": 2 * 8, "dspark": 2 * 7}
