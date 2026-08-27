from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from glm_dflash2.vllm_ascend.export_common import load_candidate_export
from glm_dflash2.vllm_ascend.export_dspark import export_dspark
from export_test_utils import tiny_config, tiny_target_io
from tools.train_drafter_offline import build_method_model


ROOT = Path(__file__).resolve().parents[1]


class DSparkExportTest(unittest.TestCase):
    def test_b8_matches_frozen_runtime_abi_except_target_layer_ids(self):
        fixture = json.loads((ROOT / "tests/fixtures/vllm_speculators_dspark_abi.json").read_text())
        config = tiny_config(block_size=8)
        model = build_method_model("dspark", config, markov_rank=4)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = export_dspark(
                Path(tmp), config=config, state_dict=model.state_dict(), target_io=tiny_target_io()
            )
            loaded = load_candidate_export(tmp)
            self.assertEqual(loaded.config["config_class"], fixture["config_class"])
            self.assertEqual(loaded.config["aux_hidden_state_layer_ids"], [1, 20, 38, 56, 75])
            self.assertEqual(loaded.config["markov_head_type"], "vanilla")
            self.assertEqual(loaded.config["markov_rank"], 4)
            self.assertTrue(loaded.config["enable_confidence_head"])
            self.assertFalse(loaded.config["sample_from_anchor"])
            self.assertEqual(manifest["num_speculative_tokens"], 7)
            for key in fixture["required_state_keys"]:
                self.assertIn(key, loaded.weights)

    def test_exported_markov_and_confidence_outputs_match_training_model(self):
        torch.manual_seed(4)
        config = tiny_config(block_size=8)
        model = build_method_model("dspark", config, markov_rank=4).eval()
        predecessor = torch.tensor([[1, 3, 5]])
        hidden = torch.randn(1, 3, 8)
        with torch.no_grad():
            expected_markov = model.markov_head(predecessor, 0, 17)
            expected_confidence = model.confidence_logits(hidden, predecessor)
        with tempfile.TemporaryDirectory() as tmp:
            export_dspark(
                Path(tmp), config=config, state_dict=model.state_dict(), target_io=tiny_target_io()
            )
            weights = load_candidate_export(tmp).weights
            markov_embedding = F.embedding(
                predecessor, weights["markov_head.markov_w1.weight"]
            )
            actual_markov = torch.einsum(
                "...r,vr->...v", markov_embedding, weights["markov_head.markov_w2.weight"]
            )
            features = torch.cat((hidden, markov_embedding), dim=-1)
            actual_confidence = F.linear(
                features,
                weights["confidence_head.proj.weight"],
                weights["confidence_head.proj.bias"],
            ).squeeze(-1)
            torch.testing.assert_close(actual_markov, expected_markov)
            torch.testing.assert_close(actual_confidence, expected_confidence)

    def test_b16_is_rejected(self):
        config = tiny_config(block_size=16)
        model = build_method_model("dspark", config, markov_rank=4)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "B8"):
            export_dspark(Path(tmp), config=config, state_dict=model.state_dict(), target_io=tiny_target_io())


if __name__ == "__main__":
    unittest.main()
