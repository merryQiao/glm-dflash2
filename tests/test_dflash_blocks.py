from __future__ import annotations

import unittest

import torch

from glm_dflash2.dflash_blocks import (
    NoValidAnchorsError,
    build_dflash_blocks,
    sample_anchor_positions,
)


class DFlashBlocksTest(unittest.TestCase):
    def test_sampling_is_sorted_unique_and_respects_real_row_lengths(self):
        loss_mask = torch.zeros(2, 12, dtype=torch.bool)
        loss_mask[0, :8] = True
        loss_mask[1, :5] = True
        attention_mask = torch.tensor(
            [[True] * 12, [True] * 5 + [False] * 7], dtype=torch.bool
        )
        anchors, keep = sample_anchor_positions(
            loss_mask,
            sample_ids=("long", "short"),
            global_seed=123,
            epoch=0,
            attention_mask=attention_mask,
            block_size=4,
            num_anchors=8,
        )
        for row in range(2):
            selected = anchors[row][keep[row]].tolist()
            self.assertEqual(selected, sorted(set(selected)))
        self.assertEqual(anchors[1][keep[1]].tolist(), [0, 1, 2, 3])

    def test_sampling_rejects_invalid_anchor_requests(self):
        with self.assertRaisesRegex(ValueError, r"\[1, 64\]"):
            sample_anchor_positions(
                torch.ones(1, 100, dtype=torch.bool),
                sample_ids=("cap",),
                global_seed=1,
                epoch=0,
                block_size=4,
                num_anchors=65,
            )
        with self.assertRaises(NoValidAnchorsError):
            sample_anchor_positions(
                torch.tensor([[False, False, False, True]]),
                sample_ids=("no-successor",),
                global_seed=1,
                epoch=0,
                block_size=4,
                num_anchors=2,
            )

    def test_sample_ids_make_anchor_sampling_batch_order_independent(self):
        mask = torch.ones(2, 20, dtype=torch.bool)
        first, first_keep = sample_anchor_positions(
            mask,
            sample_ids=("alpha", "beta"),
            global_seed=7,
            epoch=3,
            block_size=16,
            num_anchors=4,
        )
        second, second_keep = sample_anchor_positions(
            mask.flip(0),
            sample_ids=("beta", "alpha"),
            global_seed=7,
            epoch=3,
            block_size=16,
            num_anchors=4,
        )
        torch.testing.assert_close(first[0][first_keep[0]], second[1][second_keep[1]])
        torch.testing.assert_close(first[1][first_keep[1]], second[0][second_keep[0]])

    def test_anchor_only_requires_two_supervised_in_range_tokens(self):
        mask = torch.tensor([[False, False, True, True, False]])
        anchors, keep = sample_anchor_positions(
            mask,
            sample_ids=("tail",),
            global_seed=1,
            epoch=0,
            block_size=16,
            num_anchors=8,
        )
        self.assertEqual(anchors[0][keep[0]].tolist(), [2])

    def test_validity_is_cumulative_and_partial_tail_is_masked(self):
        ids = torch.tensor([[10, 11, 12, 13, 14]])
        loss_mask = torch.tensor([[True, True, True, False, True]])
        blocks = build_dflash_blocks(
            ids,
            loss_mask,
            torch.tensor([[1]]),
            torch.tensor([[True]]),
            block_size=4,
            mask_token_id=99,
        )
        self.assertEqual(blocks.target_ids.tolist(), [[[11, 12, 13, 14]]])
        self.assertEqual(blocks.target_mask.tolist(), [[[True, True, False, False]]])

        tail = build_dflash_blocks(
            ids,
            torch.ones_like(ids, dtype=torch.bool),
            torch.tensor([[3]]),
            torch.tensor([[True]]),
            block_size=4,
            mask_token_id=99,
        )
        self.assertEqual(tail.target_mask.tolist(), [[[True, True, False, False]]])
        self.assertEqual(tail.draft_position_ids.tolist(), [[3, 4, 5, 6]])

    def test_every_query_sees_strict_context_prefix_and_all_local_slots(self):
        ids = torch.arange(8).unsqueeze(0)
        blocks = build_dflash_blocks(
            ids,
            torch.ones_like(ids, dtype=torch.bool),
            torch.tensor([[4]]),
            torch.tensor([[True]]),
            block_size=4,
            mask_token_id=99,
        )
        visible = torch.isfinite(blocks.attention_mask[0, 0])
        for query in range(4):
            self.assertEqual(visible[query].nonzero().flatten().tolist(), list(range(4)) + list(range(8, 12)))
        self.assertEqual(blocks.context_position_ids.tolist(), [[0, 1, 2, 3, 4, 5, 6, 7]])
        self.assertEqual(blocks.draft_position_ids.tolist(), [[4, 5, 6, 7]])

    def test_multiple_blocks_never_attend_to_each_others_noise_slots(self):
        ids = torch.arange(10).unsqueeze(0)
        blocks = build_dflash_blocks(
            ids,
            torch.ones_like(ids, dtype=torch.bool),
            torch.tensor([[5, 6]]),
            torch.tensor([[True, True]]),
            block_size=2,
            mask_token_id=99,
        )
        visible = torch.isfinite(blocks.attention_mask[0, 0])
        self.assertEqual(
            visible[0].nonzero().flatten().tolist(), [0, 1, 2, 3, 4, 10, 11]
        )
        self.assertEqual(
            visible[2].nonzero().flatten().tolist(), [0, 1, 2, 3, 4, 5, 12, 13]
        )


if __name__ == "__main__":
    unittest.main()
