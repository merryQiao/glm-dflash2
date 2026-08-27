from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from glm_dflash2.vllm_ascend.export_dflash import export_dflash
from glm_dflash2.vllm_ascend.export_common import load_candidate_export
from export_test_utils import tiny_config, tiny_target_io
from tools.train_drafter_offline import build_method_model


class DFlashExportTest(unittest.TestCase):
    def test_b8_and_b16_have_exact_anchor_and_proposal_contract(self):
        for block_size in (8, 16):
            with self.subTest(block_size=block_size), tempfile.TemporaryDirectory() as tmp:
                config = tiny_config(block_size=block_size)
                model = build_method_model("dflash", config, markov_rank=4)
                manifest = export_dflash(
                    Path(tmp), config=config, state_dict=model.state_dict(), target_io=tiny_target_io()
                )
                loaded = load_candidate_export(tmp)
                self.assertEqual(loaded.config["architectures"], ["DFlashDraftModel"])
                self.assertEqual(loaded.config["aux_hidden_state_layer_ids"], [1, 20, 38, 56, 75])
                self.assertEqual(loaded.config["transformer_layer_config"]["layer_types"], ["full_attention"] * 5)
                self.assertFalse(loaded.config["sample_from_anchor"])
                self.assertEqual(manifest["num_speculative_tokens"], block_size - 1)
                self.assertFalse(any("candidate_selector" in key for key in loaded.weights))
                self.assertFalse(any("markov_head" in key for key in loaded.weights))

    def test_other_block_sizes_are_rejected(self):
        config = tiny_config(block_size=4)
        model = build_method_model("dflash", config, markov_rank=4)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "block size"):
            export_dflash(Path(tmp), config=config, state_dict=model.state_dict(), target_io=tiny_target_io())


if __name__ == "__main__":
    unittest.main()
