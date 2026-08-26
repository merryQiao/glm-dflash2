from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from glm_dflash2.dflash2_model import build_dflash2_config
from glm_dflash2.speculator_export import (
    export_speculator,
    load_exported_speculator,
)
from glm_dflash2.target_io import FrozenTargetIO
from tools.train_drafter_offline import build_method_model


class SpeculatorExportTest(unittest.TestCase):
    def _config(self, *, block_size: int = 8):
        return build_dflash2_config(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=5,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            target_layer_ids=[1, 20, 38, 56, 75],
            num_target_layers=78,
            block_size=block_size,
            mask_token_id=16,
            conv_group_size=4,
            selector_rank=4,
            selector_top_k=4,
            sliding_window=None,
        )

    def _target_io(self) -> FrozenTargetIO:
        embed = nn.Embedding(17, 8, dtype=torch.bfloat16)
        head = nn.Linear(8, 17, bias=False, dtype=torch.bfloat16)
        embed.weight.requires_grad_(False)
        head.weight.requires_grad_(False)
        manifest = {
            "schema": "glm-drafter-target-io-v2",
            "source_model_dir": "/models/GLM-5.2-BF16",
            "source_model_fingerprint": "model-sha",
            "model_revision": "revision-sha",
            "tokenizer_fingerprint": "tokenizer-sha",
            "model_type": "glm_moe_dsa",
            "vocab_size": 17,
            "hidden_size": 8,
            "tensors": {},
        }
        return FrozenTargetIO(embed, head, manifest)

    def test_standard_dflash_and_dspark_export_contains_complete_runtime_weights(self):
        for method in ("dflash", "dspark"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                config = self._config()
                model = build_method_model(method, config, markov_rank=4)
                target_io = self._target_io()
                export_speculator(
                    Path(tmp), method=method, config=config, model=model,
                    target_io=target_io,
                )

                exported_config = json.loads((Path(tmp) / "config.json").read_text())
                self.assertEqual(exported_config["aux_hidden_state_layer_ids"], [1, 20, 38, 56, 75])
                self.assertEqual(exported_config["block_size"], 8)
                self.assertFalse(exported_config["sample_from_anchor"])
                proposal = exported_config["speculators_config"]["proposal_methods"][0]
                self.assertEqual(proposal["speculative_tokens"], 7)
                self.assertEqual(exported_config["transformer_layer_config"]["layer_types"], ["full_attention"] * 5)
                self.assertIsNone(exported_config["transformer_layer_config"]["sliding_window"])

                weights = load_file(Path(tmp) / "model.safetensors")
                self.assertIn("embed_tokens.weight", weights)
                self.assertIn("lm_head.weight", weights)
                self.assertTrue(torch.equal(weights["embed_tokens.weight"], target_io.embed_tokens.weight))
                self.assertTrue(torch.equal(weights["lm_head.weight"], target_io.lm_head.weight))
                if method == "dspark":
                    self.assertIn("confidence_head.proj.weight", weights)
                    self.assertNotIn("confidence_head.weight", weights)

                manifest = json.loads((Path(tmp) / "export_manifest.json").read_text())
                self.assertEqual(manifest["runtime_compatibility"], "vllm-ascend-stock")
                self.assertEqual(manifest["target_model_fingerprint"], "model-sha")
                self.assertTrue((Path(tmp) / "config.py").is_file())

    def test_round_trip_is_exact_for_every_method(self):
        for method in ("dflash", "dflash2", "dspark"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                config = self._config(block_size=16 if method != "dspark" else 8)
                model = build_method_model(method, config, markov_rank=4)
                target_io = self._target_io()
                expected = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                export_speculator(
                    Path(tmp), method=method, config=config, model=model,
                    target_io=target_io,
                )
                loaded = load_exported_speculator(Path(tmp))
                self.assertEqual(loaded.method, method)
                actual = loaded.model.state_dict()
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    self.assertTrue(torch.equal(actual[key].cpu(), expected[key]), key)
                self.assertTrue(torch.equal(loaded.embed_tokens_weight, target_io.embed_tokens.weight))
                self.assertTrue(torch.equal(loaded.lm_head_weight, target_io.lm_head.weight))
                if method == "dflash2":
                    self.assertEqual(
                        loaded.manifest["runtime_compatibility"],
                        "custom-vllm-ascend-adapter-required",
                    )


if __name__ == "__main__":
    unittest.main()
