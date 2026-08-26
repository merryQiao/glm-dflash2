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
        self.assertEqual(tuple(model.markov_head.predecessor_codebook.shape), (11, 4))
        self.assertEqual(tuple(model.markov_head.successor_codebook.shape), (11, 4))
        hidden = torch.randn(2, 3, 8)
        self.assertEqual(tuple(model.confidence_logits(hidden).shape), (2, 3))

    def test_markov_head_scores_predecessor_hidden_and_successor_before_softmax(self):
        head = LowRankMarkovHead(vocab_size=5, hidden_size=3, rank=2)
        with torch.no_grad():
            head.predecessor_codebook.zero_()
            head.successor_codebook.zero_()
            head.hidden_projection.weight.zero_()
            head.predecessor_codebook[2] = torch.tensor([1.0, 2.0])
            head.successor_codebook[3] = torch.tensor([4.0, 5.0])
            head.hidden_projection.weight[:, :2] = torch.eye(2)
        hidden = torch.tensor([[2.0, 3.0, 0.0]])
        bias = head.score_chunk(hidden, torch.tensor([2]), 0, 5)
        self.assertEqual(tuple(bias.shape), (1, 5))
        self.assertAlmostEqual(bias[0, 3].item(), 38.0)


if __name__ == "__main__":
    unittest.main()
