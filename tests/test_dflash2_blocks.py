from __future__ import annotations

import unittest

import torch

from glm_dflash2.dflash2_blocks import (
    NoValidAnchorsError,
    build_dflash_blocks,
    sample_anchor_positions,
)


class DFlash2BlocksTest(unittest.TestCase):
    def test_uniform_without_replacement_sampling_is_sorted_and_padded(self):
        mask = torch.zeros(2, 12, dtype=torch.bool)
        mask[0, :8] = True
        mask[1, :2] = True
        generator = torch.Generator().manual_seed(123)
        anchors, keep = sample_anchor_positions(
            mask, block_size=4, num_anchors=3, generator=generator
        )
        self.assertEqual(tuple(anchors.shape), (2, 3))
        self.assertEqual(keep.tolist(), [[True, True, True], [True, False, False]])
        self.assertEqual(anchors[0].tolist(), sorted(anchors[0].tolist()))
        self.assertEqual(anchors[1, 2].item(), 0)
        for row in range(2):
            selected = anchors[row][keep[row]].tolist()
            self.assertEqual(len(selected), len(set(selected)))

    def test_sampling_is_reproducible_from_generator_state(self):
        mask = torch.ones(1, 20, dtype=torch.bool)
        first = sample_anchor_positions(
            mask, block_size=4, num_anchors=4, generator=torch.Generator().manual_seed(77)
        )[0]
        second = sample_anchor_positions(
            mask, block_size=4, num_anchors=4, generator=torch.Generator().manual_seed(77)
        )[0]
        self.assertTrue(torch.equal(first, second))

    def test_sampling_has_hard_anchor_cap(self):
        with self.assertRaisesRegex(ValueError, r"\[1, 64\]"):
            sample_anchor_positions(
                torch.ones(1, 100, dtype=torch.bool),
                block_size=4,
                num_anchors=65,
                generator=torch.Generator(),
            )

    def test_no_valid_two_token_anchor_is_explicit(self):
        mask = torch.tensor([[False, False, False, True]])
        with self.assertRaises(NoValidAnchorsError):
            sample_anchor_positions(
                mask, block_size=4, num_anchors=2, generator=torch.Generator()
            )

    def test_anchor_requires_at_least_one_supervised_successor(self):
        mask = torch.tensor([[True, False, False, False]])
        with self.assertRaises(NoValidAnchorsError):
            sample_anchor_positions(
                mask, block_size=4, num_anchors=1, generator=torch.Generator()
            )

    def test_sampling_uses_each_rows_real_length_not_batch_padding(self):
        loss_mask = torch.zeros(2, 8, dtype=torch.bool)
        loss_mask[0, :8] = True
        loss_mask[1, :5] = True
        attention_mask = torch.tensor(
            [[True] * 8, [True] * 5 + [False] * 3], dtype=torch.bool
        )
        anchors, keep = sample_anchor_positions(
            loss_mask,
            attention_mask=attention_mask,
            block_size=4,
            num_anchors=8,
            generator=torch.Generator().manual_seed(1),
        )
        self.assertEqual(anchors[1][keep[1]].tolist(), [0, 1, 2, 3])

    def test_builder_rejects_anchor_without_real_successor(self):
        ids = torch.arange(16).reshape(2, 8)
        mask = torch.ones_like(ids, dtype=torch.bool)
        attention_mask = torch.tensor(
            [[True] * 8, [True] * 5 + [False] * 3], dtype=torch.bool
        )
        with self.assertRaisesRegex(ValueError, "in-range successor"):
            build_dflash_blocks(
                ids,
                mask,
                torch.tensor([[0], [4]]),
                torch.tensor([[True], [True]]),
                attention_mask=attention_mask,
                block_size=4,
                mask_token_id=99,
                sliding_window=None,
            )

    def test_block_targets_are_unshifted_and_mask_positions_are_clean(self):
        ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
        loss_mask = torch.tensor([[False, True, True, True, True, True, False, False]])
        anchors = torch.tensor([[1, 3]])
        keep = torch.tensor([[True, True]])
        blocks = build_dflash_blocks(
            ids,
            loss_mask,
            anchors,
            keep,
            block_size=4,
            mask_token_id=99,
            sliding_window=None,
        )
        self.assertEqual(blocks.noise_ids.tolist(), [[11, 99, 99, 99, 13, 99, 99, 99]])
        self.assertEqual(blocks.target_ids.tolist(), [[[11, 12, 13, 14], [13, 14, 15, 16]]])
        self.assertEqual(
            blocks.target_mask.tolist(),
            [[[True, True, True, True], [True, True, True, False]]],
        )
        self.assertEqual(blocks.draft_position_ids.tolist(), [[1, 2, 3, 4, 3, 4, 5, 6]])

    def test_attention_sees_strict_full_prefix_and_only_own_block(self):
        ids = torch.arange(10).unsqueeze(0)
        loss_mask = torch.ones_like(ids, dtype=torch.bool)
        blocks = build_dflash_blocks(
            ids,
            loss_mask,
            torch.tensor([[5, 6]]),
            torch.tensor([[True, True]]),
            block_size=2,
            mask_token_id=99,
            sliding_window=None,
        )
        visible = torch.isfinite(blocks.attention_mask[0, 0])
        # Every local query sees the same strict prefix and both own slots.
        self.assertEqual(visible[0].nonzero().flatten().tolist(), [0, 1, 2, 3, 4, 10, 11])
        self.assertEqual(visible[1].nonzero().flatten().tolist(), [0, 1, 2, 3, 4, 10, 11])
        # Second block never sees the first block's noise slots.
        self.assertEqual(visible[2].nonzero().flatten().tolist(), [0, 1, 2, 3, 4, 5, 12, 13])


if __name__ == "__main__":
    unittest.main()
