from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from glm_dflash2.dflash2_objective import (
    compute_acceptance_stats,
    compute_dflash2_loss,
    selector_supervision,
)


class DFlash2ObjectiveTest(unittest.TestCase):
    def test_exact_base_and_selector_denominators(self):
        base_nll = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
        target_ids = torch.tensor([[[10, 20, 30, 40]]])
        pred_mask = torch.tensor([[[1.0, 1.0, 0.0]]])
        candidate_ids = torch.tensor([[[[20, 21], [31, 30], [40, 41]]]])
        selector_scores = torch.tensor(
            [[[[2.0, 0.0], [3.0, 1.0], [0.0, 0.0]]]], requires_grad=True
        )
        out = compute_dflash2_loss(
            base_nll=base_nll,
            candidate_ids=candidate_ids,
            selector_scores=selector_scores,
            target_ids=target_ids,
            pred_mask=pred_mask,
            gamma=2.0,
            selector_loss_weight=1.0,
        )
        w0, w1 = 1.0, math.exp(-0.5)
        expected_base = (1.0 * w0 + 2.0 * w1) / (w0 + w1)
        # Both valid targets hit top-k; target slots are 0 and 1.
        nll0 = F.cross_entropy(torch.tensor([[2.0, 0.0]]), torch.tensor([0])).item()
        nll1 = F.cross_entropy(torch.tensor([[3.0, 1.0]]), torch.tensor([1])).item()
        expected_selector = (nll0 * w0 + nll1 * w1) / (w0 + w1 + 1e-6)
        expected_total = (
            1.0 * w0 + 2.0 * w1 + nll0 * w0 + nll1 * w1
        ) / (w0 + w1)
        self.assertAlmostEqual(out.base_loss.item(), expected_base, places=6)
        self.assertAlmostEqual(out.selector_loss.item(), expected_selector, places=6)
        self.assertAlmostEqual(out.loss.item(), expected_total, places=6)

    def test_missing_true_candidates_have_zero_selector_signal(self):
        out = compute_dflash2_loss(
            base_nll=torch.ones(1, 1, 2, requires_grad=True),
            candidate_ids=torch.tensor([[[[7, 8], [9, 10]]]]),
            selector_scores=torch.randn(1, 1, 2, 2, requires_grad=True),
            target_ids=torch.tensor([[[1, 2, 3]]]),
            pred_mask=torch.ones(1, 1, 2),
            gamma=7.0,
        )
        self.assertEqual(out.selector_loss.item(), 0.0)
        self.assertEqual(out.candidate_hits.item(), 0)
        self.assertEqual(out.candidate_total.item(), 2)

    def test_total_loss_uses_one_common_token_denominator(self):
        base_nll = torch.tensor([[[1.0, 3.0]]], requires_grad=True)
        candidate_ids = torch.tensor([[[[2, 9], [7, 8]]]])
        selector_scores = torch.tensor(
            [[[[2.0, 0.0], [1.0, -1.0]]]], requires_grad=True
        )
        out = compute_dflash2_loss(
            base_nll=base_nll,
            candidate_ids=candidate_ids,
            selector_scores=selector_scores,
            target_ids=torch.tensor([[[1, 2, 3]]]),
            pred_mask=torch.ones(1, 1, 2),
            gamma=2.0,
        )
        w0, w1 = 1.0, math.exp(-0.5)
        base_numerator = 1.0 * w0 + 3.0 * w1
        selector_numerator = F.cross_entropy(
            torch.tensor([[2.0, 0.0]]), torch.tensor([0])
        ).item() * w0
        expected = (base_numerator + selector_numerator) / (w0 + w1)
        self.assertAlmostEqual(out.loss.item(), expected, places=6)

    def test_masked_negative_infinity_candidates_do_not_create_nan(self):
        selector_scores = torch.tensor(
            [[[[2.0, 1.0], [float("-inf"), float("-inf")]]]],
            requires_grad=True,
        )
        out = compute_dflash2_loss(
            base_nll=torch.tensor([[[1.0, 0.0]]], requires_grad=True),
            candidate_ids=torch.tensor([[[[2, 3], [0, 0]]]]),
            selector_scores=selector_scores,
            target_ids=torch.tensor([[[1, 2, 4]]]),
            pred_mask=torch.tensor([[[True, False]]]),
            gamma=7.0,
        )
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertTrue(torch.isfinite(selector_scores.grad[0, 0, 0]).all())

    def test_selector_supervision_uses_successor_targets(self):
        candidate_ids = torch.tensor([[[[5, 6], [7, 8]]]])
        target_ids = torch.tensor([[[4, 6, 7]]])
        slots, hit = selector_supervision(candidate_ids, target_ids)
        self.assertEqual(slots.tolist(), [[[1, 0]]])
        self.assertEqual(hit.tolist(), [[[True, True]]])

    def test_bonus_inclusive_acceptance_stops_at_first_error(self):
        pred = torch.tensor([[[1, 2, 9], [4, 8, 6]]])
        target = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
        mask = torch.ones_like(pred, dtype=torch.bool)
        mean, total, blocks = compute_acceptance_stats(pred, target, mask)
        # accepted drafts are 2 and 1; metric adds one target bonus to each.
        self.assertEqual(total.item(), 5.0)
        self.assertEqual(blocks.item(), 2)
        self.assertEqual(mean.item(), 2.5)

    def test_invalid_positions_do_not_truncate_or_count(self):
        pred = torch.tensor([[[1, 99, 3]]])
        target = torch.tensor([[[1, 2, 3]]])
        mask = torch.tensor([[[True, False, True]]])
        mean, _, _ = compute_acceptance_stats(pred, target, mask)
        self.assertEqual(mean.item(), 3.0)


if __name__ == "__main__":
    unittest.main()
