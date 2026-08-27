from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from glm_dflash2.vllm_ascend.export_common import load_candidate_export
from glm_dflash2.vllm_ascend.export_dflash2 import export_dflash2
from tests.export_test_utils import tiny_config, tiny_target_io
from tools.train_drafter_offline import build_method_model


class DFlash2ExportTest(unittest.TestCase):
    def test_export_is_distinct_and_preserves_selector_and_convolution(self):
        config = tiny_config(block_size=16)
        model = build_method_model("dflash2", config, markov_rank=4)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = export_dflash2(
                Path(tmp), config=config, state_dict=model.state_dict(), target_io=tiny_target_io()
            )
            loaded = load_candidate_export(tmp)
            self.assertEqual(loaded.config["speculators_model_type"], "dflash2")
            self.assertEqual(loaded.config["dflash2_config"]["selector_rank"], 4)
            self.assertEqual(loaded.config["dflash2_config"]["selector_top_k"], 4)
            self.assertEqual(loaded.config["dflash2_config"]["conv_kernel_size"], 2)
            self.assertEqual(manifest["runtime_adapter"], "custom_class:dflash2")
            self.assertTrue(any("attention_conv" in key for key in loaded.weights))
            self.assertIn("candidate_selector.predecessor_codebook", loaded.weights)
            self.assertFalse(any("markov_head" in key for key in loaded.weights))


if __name__ == "__main__":
    unittest.main()
