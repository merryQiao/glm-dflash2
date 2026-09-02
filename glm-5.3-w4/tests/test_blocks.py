from __future__ import annotations

import torch

from glm53_w4.blocks import build_sliding_blocks, sample_anchor_positions
from glm53_w4.data import expand_training_window, TrainingWindow


def test_physical_halo_never_contains_anchor_or_future() -> None:
    ids = torch.arange(20).reshape(1, -1)
    loss = torch.ones_like(ids, dtype=torch.bool)
    anchors = torch.tensor([[3, 12]])
    keep = torch.tensor([[True, True]])
    blocks = build_sliding_blocks(
        ids,
        loss,
        anchors,
        keep,
        block_size=4,
        mask_token_id=99,
        sliding_window=8,
    )
    assert blocks.context_indices[0, 0].tolist() == [0, 0, 0, 0, 0, 0, 1, 2]
    assert blocks.context_mask[0, 0].tolist() == [False] * 5 + [True] * 3
    assert blocks.context_indices[0, 1].tolist() == list(range(4, 12))
    assert blocks.context_mask[0, 1].all()
    assert not bool((blocks.context_indices[0, 1] >= 12).any())


def test_local_block_has_real_anchor_and_parallel_masks() -> None:
    ids = torch.arange(20).reshape(1, -1)
    loss = torch.ones_like(ids, dtype=torch.bool)
    blocks = build_sliding_blocks(
        ids,
        loss,
        torch.tensor([[5]]),
        torch.tensor([[True]]),
        block_size=4,
        mask_token_id=99,
        sliding_window=8,
    )
    assert blocks.target_ids.tolist() == [[[5, 6, 7, 8]]]
    assert blocks.noise_ids.tolist() == [[[5, 99, 99, 99]]]
    assert blocks.local_visibility.shape == (1, 1, 4, 4)
    assert blocks.local_visibility.all()
    assert blocks.prediction_mask.tolist() == [[[True, True, True]]]


def test_cropped_window_preserves_absolute_positions() -> None:
    ids = torch.arange(20).reshape(1, -1)
    loss = torch.ones_like(ids, dtype=torch.bool)
    blocks = build_sliding_blocks(
        ids,
        loss,
        torch.tensor([[5]]),
        torch.tensor([[True]]),
        block_size=4,
        mask_token_id=99,
        sliding_window=8,
        position_offset=torch.tensor([100]),
    )
    assert blocks.draft_position_ids.tolist() == [[[105, 106, 107, 108]]]
    assert blocks.context_position_ids[0, 0, -5:].tolist() == [100, 101, 102, 103, 104]


def test_tail_mask_is_prefix_closed() -> None:
    ids = torch.arange(10).reshape(1, -1)
    loss = torch.ones_like(ids, dtype=torch.bool)
    loss[0, 8] = False
    blocks = build_sliding_blocks(
        ids,
        loss,
        torch.tensor([[6]]),
        torch.tensor([[True]]),
        block_size=4,
        mask_token_id=99,
        sliding_window=8,
    )
    assert blocks.target_mask.tolist() == [[[True, True, False, False]]]
    assert blocks.prediction_mask.tolist() == [[[True, False, False]]]


def test_anchor_sampling_is_deterministic_and_allows_many_anchors() -> None:
    loss = torch.ones((1, 700), dtype=torch.bool)
    args = dict(
        sample_ids=["sample-a"],
        global_seed=9,
        epoch=2,
        block_size=16,
        num_anchors=512,
    )
    first = sample_anchor_positions(loss, **args)
    second = sample_anchor_positions(loss, **args)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[0].shape == (1, 512)


def test_cropped_supervision_window_keeps_full_left_halo_and_block_tail() -> None:
    read = expand_training_window(
        TrainingWindow(5000, 9096),
        total_tokens=20_000,
        sliding_window=2048,
        block_size=16,
    )
    assert (read.start, read.end) == (2952, 9111)
    assert (read.anchor_start, read.anchor_end) == (2048, 6144)
