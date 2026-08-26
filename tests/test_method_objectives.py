from __future__ import annotations

import math
import unittest

import torch

from glm_dflash2.method_objectives import depth_weighted_objective


class MethodObjectivesTest(unittest.TestCase):
    def test_dflash_ce_uses_exp_negative_depth_over_gamma(self):
        losses = torch.tensor([[[1.0, 2.0, 9.0]]], requires_grad=True)
        mask = torch.tensor([[[True, True, False]]])
        terms = depth_weighted_objective(losses, mask, gamma=7.0)
        weights = torch.tensor([1.0, math.exp(-1.0 / 7.0)])
        expected_numerator = losses[0, 0, :2] @ weights
        self.assertAlmostEqual(terms.numerator.item(), expected_numerator.item(), places=6)
        self.assertAlmostEqual(terms.denominator.item(), weights.sum().item(), places=6)
        self.assertAlmostEqual(
            terms.mean.item(), (expected_numerator / weights.sum()).item(), places=6
        )

    def test_empty_mask_returns_differentiable_zero_and_additive_counts(self):
        losses = torch.randn(1, 2, 3, requires_grad=True)
        terms = depth_weighted_objective(
            losses, torch.zeros_like(losses, dtype=torch.bool), gamma=7.0
        )
        self.assertEqual(terms.numerator.item(), 0.0)
        self.assertEqual(terms.denominator.item(), 0.0)
        self.assertEqual(terms.mean.item(), 0.0)
        terms.mean.backward()
        self.assertIsNotNone(losses.grad)
        self.assertTrue((losses.grad == 0).all())


if __name__ == "__main__":
    unittest.main()
