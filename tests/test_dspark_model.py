from __future__ import annotations

import unittest

import torch

from glm_dflash2.dflash2_model import build_dflash2_config
from glm_dflash2.dspark_model import DSparkDraftModel, LowRankMarkovHead


class DSparkModelTest(unittest.TestCase):
    def _config(self):
        return build_dflash2_config(
            vocab_size=11,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            target_layer_ids=[0, 1],
            num_target_layers=2,
            block_size=4,
            mask_token_id=10,
            conv_group_size=4,
            selector_rank=4,
            selector_top_k=4,
            sliding_window=None,
        )

    def test_dspark_adds_only_markov_and_confidence_to_plain_backbone(self):
        model = DSparkDraftModel(self._config(), markov_rank=4)
        self.assertFalse(any("conv" in name for name, _ in model.named_modules()))
        self.assertEqual(tuple(model.markov_head.markov_w1.weight.shape), (11, 4))
        self.assertEqual(tuple(model.markov_head.markov_w2.weight.shape), (11, 4))
        self.assertEqual(model.confidence_head.in_features, 12)
        hidden = torch.randn(2, 3, 8)
        predecessors = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(
            tuple(model.confidence_logits(hidden, predecessors).shape), (2, 3)
        )

    def test_vanilla_markov_bias_depends_on_predecessor_but_not_hidden(self):
        head = LowRankMarkovHead(vocab_size=5, hidden_size=3, rank=2)
        with torch.no_grad():
            head.markov_w1.weight.zero_()
            head.markov_w2.weight.zero_()
            head.markov_w1.weight[2] = torch.tensor([1.0, 2.0])
            head.markov_w2.weight[3] = torch.tensor([4.0, 5.0])
        predecessor = torch.tensor([2])
        first = head.score_chunk(torch.tensor([[2.0, 3.0, 0.0]]), predecessor, 0, 5)
        second = head.score_chunk(torch.tensor([[-7.0, 9.0, 4.0]]), predecessor, 0, 5)
        self.assertEqual(tuple(first.shape), (1, 5))
        self.assertAlmostEqual(first[0, 3].item(), 14.0)
        torch.testing.assert_close(first, second)

    def test_confidence_is_markov_aware(self):
        model = DSparkDraftModel(self._config(), markov_rank=4)
        with torch.no_grad():
            model.markov_head.markov_w1.weight.zero_()
            model.markov_head.markov_w1.weight[2, 0] = 1.0
            model.confidence_head.weight.zero_()
            model.confidence_head.bias.zero_()
            model.confidence_head.weight[0, 8] = 2.0
        hidden = torch.zeros(1, 1, 8)
        low = model.confidence_logits(hidden, torch.tensor([[1]]))
        high = model.confidence_logits(hidden, torch.tensor([[2]]))
        self.assertAlmostEqual(low.item(), 0.0)
        self.assertAlmostEqual(high.item(), 2.0)

    def test_head_initialization_uses_model_initializer_range(self):
        torch.manual_seed(123)
        head = LowRankMarkovHead(vocab_size=4096, hidden_size=8, rank=64)
        self.assertAlmostEqual(head.markov_w1.weight.std().item(), 0.02, delta=0.001)
        self.assertAlmostEqual(head.markov_w2.weight.std().item(), 0.02, delta=0.001)
        model = DSparkDraftModel(self._config(), markov_rank=4)
        self.assertAlmostEqual(
            model.confidence_head.weight.std().item(),
            model.config.initializer_range,
            delta=0.01,
        )
        torch.testing.assert_close(
            model.confidence_head.bias, torch.zeros_like(model.confidence_head.bias)
        )


if __name__ == "__main__":
    unittest.main()
