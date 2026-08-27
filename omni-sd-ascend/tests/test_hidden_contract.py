from __future__ import annotations

import unittest

import torch

from omni_sd.vllm_ascend_hidden import HiddenContractError, response_loss_mask, validate_hidden_tensors


class HiddenContractTests(unittest.TestCase):
    def test_response_mask_excludes_prompt(self):
        self.assertEqual(response_loss_mask(5, 2).tolist(), [False, False, True, True, True])

    def test_non_finite_hidden_is_rejected(self):
        hidden = torch.zeros(3, 2, 4)
        hidden[1, 1, 1] = float("nan")
        with self.assertRaisesRegex(HiddenContractError, "finite"):
            validate_hidden_tensors(hidden, torch.zeros(3, 4), tokens=3, layers=2, hidden_size=4)


if __name__ == "__main__":
    unittest.main()
