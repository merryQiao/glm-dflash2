from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from glm53_drafters.blocks import (
    build_physical_blocks,
    sample_anchor_positions,
    select_training_window,
)
from glm53_drafters.chunked_lm_head import chunked_cross_entropy
from glm53_drafters.dflash2_model import CandidateSelector
from glm53_drafters.objectives import (
    confidence_bce,
    depth_weights,
    exact_total_variation,
    selector_cross_entropy,
)


def test_anchor_sampling_is_stable_mask_bounded_and_padded_without_duplicates() -> None:
    mask = torch.tensor([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=torch.bool)
    first = sample_anchor_positions(
        mask,
        sample_id="sample-a",
        epoch=2,
        block_size=4,
        count=4,
        seed=42,
    )
    second = sample_anchor_positions(
        mask,
        sample_id="sample-a",
        epoch=2,
        block_size=4,
        count=4,
        seed=42,
    )
    assert torch.equal(first.positions, second.positions)
    assert torch.equal(first.keep_mask, second.keep_mask)
    kept = first.positions[first.keep_mask]
    assert set(kept.tolist()) == {1, 6}
    assert len(set(kept.tolist())) == kept.numel()
    assert torch.equal(first.positions[~first.keep_mask], torch.full((2,), -1))


def test_physical_blocks_have_one_clean_anchor_masks_and_absolute_positions() -> None:
    input_ids = torch.arange(12)
    auxiliary = torch.randn(12, 3, 4)
    final_hidden = torch.randn(12, 4)
    anchors = sample_anchor_positions(
        torch.ones(12, dtype=torch.bool),
        sample_id="x",
        epoch=0,
        block_size=4,
        count=2,
    )
    blocks = build_physical_blocks(
        input_ids,
        auxiliary,
        anchors,
        block_size=4,
        mask_token_id=99,
        target_final_hidden=final_hidden,
    )
    assert blocks.input_ids.shape == blocks.target_ids.shape == (2, 4)
    assert torch.equal(blocks.input_ids[:, 0], blocks.target_ids[:, 0])
    assert torch.equal(blocks.input_ids[:, 1:], torch.full((2, 3), 99))
    assert torch.equal(
        blocks.position_ids,
        anchors.positions[:, None] + torch.arange(4)[None, :],
    )
    assert blocks.auxiliary_hidden.shape == (12, 3, 4)
    assert blocks.target_final_hidden is not None
    assert torch.equal(blocks.target_final_hidden, final_hidden[blocks.position_ids])
    assert blocks.attention_mask is not None
    first, second = anchors.positions.tolist()
    # Each query sees only its own draft block plus context strictly before anchor.
    assert blocks.attention_mask[0, 0, :first].all()
    assert not blocks.attention_mask[0, 0, first:12].any()
    assert blocks.attention_mask[0, 0, 12:16].all()
    assert not blocks.attention_mask[0, 0, 16:20].any()
    assert blocks.attention_mask[0, 4, :second].all()
    assert blocks.attention_mask[0, 4, 16:20].all()


def test_long_training_window_is_bounded_stable_and_epoch_variant() -> None:
    mask = torch.ones(10_000, dtype=torch.bool)
    first = select_training_window(
        mask, sample_id="long", epoch=0, block_size=16, max_tokens=128
    )
    repeated = select_training_window(
        mask, sample_id="long", epoch=0, block_size=16, max_tokens=128
    )
    later = select_training_window(
        mask, sample_id="long", epoch=1, block_size=16, max_tokens=128
    )
    assert first == repeated
    assert first.end - first.start == later.end - later.start == 128
    assert first != later


def test_long_training_windows_and_anchor_buckets_cycle_without_resume_drift() -> None:
    mask = torch.ones(1_000, dtype=torch.bool)
    block_size, max_tokens = 16, 128
    stride = max_tokens - block_size + 1
    cycle = math.ceil((mask.numel() - max_tokens) / stride) + 1
    windows = [
        select_training_window(
            mask,
            sample_id="long-cycle",
            epoch=epoch,
            block_size=block_size,
            max_tokens=max_tokens,
        )
        for epoch in range(cycle)
    ]
    assert len({(window.start, window.end) for window in windows}) == cycle
    assert select_training_window(
        mask,
        sample_id="long-cycle",
        epoch=cycle,
        block_size=block_size,
        max_tokens=max_tokens,
    ) == windows[0]

    valid_count = mask.numel() - 4 + 1
    anchor_cycle = math.ceil(valid_count / 10)
    covered: set[int] = set()
    for epoch in range(anchor_cycle):
        anchors = sample_anchor_positions(
            mask,
            sample_id="anchor-cycle",
            epoch=epoch,
            block_size=4,
            count=10,
        )
        covered.update(anchors.positions[anchors.keep_mask].tolist())
    assert covered == set(range(valid_count))


def test_short_sequence_with_no_valid_anchor_pads_without_indexing_past_end() -> None:
    input_ids = torch.tensor([4, 5, 6])
    auxiliary = torch.randn(3, 2, 4)
    final_hidden = torch.randn(3, 4)
    anchors = sample_anchor_positions(
        torch.ones(3, dtype=torch.bool),
        sample_id="short",
        epoch=0,
        block_size=8,
        count=2,
    )
    blocks = build_physical_blocks(
        input_ids,
        auxiliary,
        anchors,
        block_size=8,
        mask_token_id=9,
        target_final_hidden=final_hidden,
    )
    assert not blocks.keep_mask.any()
    assert blocks.input_ids.shape == blocks.target_ids.shape == (2, 8)
    assert torch.count_nonzero(blocks.input_ids) == 0
    assert blocks.target_final_hidden is not None
    assert torch.count_nonzero(blocks.target_final_hidden) == 0


def test_chunked_cross_entropy_matches_dense_full_vocabulary_ce_and_gradients() -> None:
    torch.manual_seed(4)
    hidden_dense = torch.randn(7, 5, requires_grad=True)
    hidden_chunked = hidden_dense.detach().clone().requires_grad_(True)
    weight = torch.randn(13, 5)
    targets = torch.randint(0, 13, (7,))
    sample_weights = torch.rand(7)

    dense = (
        F.cross_entropy(F.linear(hidden_dense.float(), weight.float()), targets, reduction="none")
        * sample_weights
    ).sum() / sample_weights.sum()
    chunked = chunked_cross_entropy(
        hidden_chunked,
        weight,
        targets,
        weights=sample_weights,
        vocab_chunk_size=4,
    ).mean
    assert torch.allclose(chunked, dense, atol=1e-6, rtol=1e-6)
    dense.backward()
    chunked.backward()
    assert torch.allclose(hidden_chunked.grad, hidden_dense.grad, atol=2e-6, rtol=2e-6)


def test_depth_weights_and_selector_loss_rejects_positive_weight_candidate_miss() -> None:
    weights = depth_weights(block_size=8, gamma=4)
    assert weights.shape == (7,)
    assert torch.allclose(weights[0], torch.tensor(1.0))
    assert torch.allclose(weights[1], torch.tensor(math.exp(-1 / 4)))

    candidate_ids = torch.tensor([[4, 2, 1], [5, 6, 7]])
    candidate_logits = torch.tensor([[0.2, 1.1, -0.3], [9.0, 8.0, 7.0]])
    targets = torch.tensor([2, 3])
    with pytest.raises(ValueError, match="target.*candidate"):
        selector_cross_entropy(candidate_ids, candidate_logits, targets)
    loss = selector_cross_entropy(
        candidate_ids,
        candidate_logits,
        targets,
        weights=torch.tensor([0.25, 0.0]),
    )
    expected = F.cross_entropy(candidate_logits[:1], torch.tensor([1]))
    assert loss.denominator.item() == pytest.approx(0.25)
    assert torch.allclose(loss.mean, expected)


def test_dflash2_training_candidates_inject_misses_and_preserve_inference() -> None:
    torch.manual_seed(91)
    selector = CandidateSelector(hidden_size=8, vocab_size=11, rank=4, top_k=3)
    hidden = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(11, 8)
    predecessors = torch.tensor([1, 2, 3, 4])
    inference_ids, inference_logits = selector(hidden, weight, predecessors)
    targets = inference_ids[:, 0].clone()
    targets[1] = next(
        token
        for token in range(11)
        if token not in set(inference_ids[1].tolist())
    )
    training_ids, training_logits, unary_hits = selector.training_forward(
        hidden, weight, predecessors, targets
    )
    repeated_ids, repeated_logits = selector(hidden, weight, predecessors)
    assert torch.equal(repeated_ids, inference_ids)
    torch.testing.assert_close(repeated_logits, inference_logits)
    assert unary_hits.tolist() == [True, False, True, True]
    assert training_ids.eq(targets.unsqueeze(-1)).any(-1).all()
    loss = selector_cross_entropy(training_ids, training_logits, targets)
    loss.mean.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_exact_tv_and_confidence_target_are_full_vocabulary_one_minus_tv() -> None:
    student = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
    target = torch.tensor([[1.0, 1.0, -1.0], [0.0, 2.0, 0.0]])
    tv = exact_total_variation(student, target)
    expected = 0.5 * (
        student.softmax(-1) - target.softmax(-1)
    ).abs().sum(-1)
    assert torch.allclose(tv.per_token, expected)
    confidence_logits = torch.tensor([0.3, -0.4])
    confidence = confidence_bce(confidence_logits, tv.per_token)
    expected_bce = F.binary_cross_entropy_with_logits(
        confidence_logits, 1.0 - expected
    )
    assert torch.allclose(confidence.mean, expected_bce)


def test_confidence_target_is_detached_from_student_distribution() -> None:
    student = torch.randn(2, 5, requires_grad=True)
    target = torch.randn(2, 5)
    confidence_logits = torch.randn(2, requires_grad=True)
    tv = exact_total_variation(student, target)
    confidence_bce(confidence_logits, tv.per_token).mean.backward()
    assert student.grad is None or torch.count_nonzero(student.grad) == 0
    assert confidence_logits.grad is not None


def test_vocabulary_matmuls_remain_bf16_while_reductions_are_fp32(
    monkeypatch,
) -> None:
    calls: list[tuple[torch.dtype, torch.dtype]] = []
    original = F.linear

    def recording_linear(value, weight, bias=None):
        calls.append((value.dtype, weight.dtype))
        return original(value, weight, bias)

    monkeypatch.setattr(F, "linear", recording_linear)
    hidden = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(11, 8, dtype=torch.bfloat16)
    objective = chunked_cross_entropy(
        hidden,
        weight,
        torch.tensor([1, 5, 9]),
        vocab_chunk_size=4,
    )
    assert calls and set(calls) == {(torch.bfloat16, torch.bfloat16)}
    assert objective.numerator.dtype == objective.denominator.dtype == torch.float32

    calls.clear()
    selector = CandidateSelector(
        hidden_size=8, vocab_size=11, rank=4, top_k=3
    ).bfloat16()
    selector(hidden, weight, torch.tensor([1, 2, 3]))
    assert calls and set(calls) == {(torch.bfloat16, torch.bfloat16)}
