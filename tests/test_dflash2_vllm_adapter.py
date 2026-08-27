from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch

from glm_dflash2.vllm_ascend.export_dflash2 import export_dflash2
from integrations.vllm_ascend.dflash2_model_loader import load_dflash2_candidate
from integrations.vllm_ascend.dflash2_proposer import rerank_topk_candidates
from tests.export_test_utils import tiny_config, tiny_target_io
from tools.train_drafter_offline import build_method_model


class DFlash2AdapterTest(unittest.TestCase):
    def test_reranking_returns_one_token_per_position_without_full_vocab_reprojection(self):
        base_logits = torch.tensor([[[0.1, 2.0, 1.0, 0.0], [1.0, 0.0, 2.0, 0.5]]])
        top_scores, top_ids = base_logits.topk(3, dim=-1)
        residual = torch.zeros_like(top_scores)
        residual[..., 1] = 5.0
        chosen = rerank_topk_candidates(top_ids, top_scores, residual)
        self.assertEqual(tuple(chosen.shape), (1, 2))
        self.assertTrue(torch.equal(chosen, top_ids[..., 1]))

    def test_reranking_rejects_shape_mismatch_or_empty_candidates(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            rerank_topk_candidates(torch.zeros(1, 2, 3, dtype=torch.long), torch.zeros(1, 2, 2), torch.zeros(1, 2, 3))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            rerank_topk_candidates(torch.zeros(1, 2, 0, dtype=torch.long), torch.zeros(1, 2, 0), torch.zeros(1, 2, 0))

    def test_loader_accepts_only_a_declared_dflash2_candidate(self):
        config = tiny_config(block_size=8)
        model = build_method_model("dflash2", config, markov_rank=4)
        with tempfile.TemporaryDirectory() as tmp:
            export_dflash2(
                Path(tmp), config=config, state_dict=model.state_dict(), target_io=tiny_target_io()
            )
            loaded = load_dflash2_candidate(tmp)
            self.assertEqual(loaded.method, "dflash2")
            self.assertEqual(loaded.manifest["runtime_adapter"], "custom_class:dflash2")


if __name__ == "__main__":
    unittest.main()
