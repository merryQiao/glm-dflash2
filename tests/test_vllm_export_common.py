from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from glm_dflash2.vllm_ascend.export_common import (
    load_candidate_export,
    write_candidate_export,
)
from export_test_utils import flip_one_byte, tiny_target_io


class CommonExportTest(unittest.TestCase):
    def _write(self, root: Path):
        return write_candidate_export(
            root,
            method="dflash",
            config={"model_type": "dflash", "block_size": 8},
            weights={"fc.weight": torch.zeros(8, 40, dtype=torch.bfloat16)},
            target_io=tiny_target_io(),
            method_metadata={
                "aux_hidden_state_layer_ids": [1, 20, 38, 56, 75],
                "block_size": 8,
                "num_speculative_tokens": 7,
                "sample_from_anchor": False,
                "method_parameters": {},
            },
            config_source="# test config\n",
        )

    def test_candidate_is_atomic_hashed_and_not_deployable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            manifest = self._write(root)
            self.assertEqual(manifest["schema"], "glm-drafter-speculator-export-v2")
            self.assertEqual(manifest["status"], "candidate-not-deployable")
            self.assertEqual(manifest["target_model_fingerprint"], "model-sha")
            self.assertEqual(manifest["target_io_sha256"], "target-io-sha")
            self.assertFalse((root / "deploy_attestation.json").exists())
            loaded = load_candidate_export(root)
            self.assertEqual(loaded.method, "dflash")
            self.assertIn("embed_tokens.weight", loaded.weights)
            self.assertIn("lm_head.weight", loaded.weights)
            self.assertFalse(any(root.parent.glob(f".{root.name}.tmp-*")))

    def test_checksum_corruption_fails_closed(self):
        for filename in ("config.json", "model.safetensors"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "export"
                self._write(root)
                flip_one_byte(root / filename)
                with self.assertRaisesRegex(ValueError, "checksum"):
                    load_candidate_export(root)

    def test_duplicate_target_keys_and_missing_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "embed_tokens"):
                write_candidate_export(
                    Path(tmp) / "export",
                    method="dflash",
                    config={"block_size": 8},
                    weights={"embed_tokens.weight": torch.zeros(17, 8)},
                    target_io=tiny_target_io(),
                    method_metadata={
                        "aux_hidden_state_layer_ids": [1, 20, 38, 56, 75],
                        "block_size": 8,
                        "num_speculative_tokens": 7,
                        "sample_from_anchor": False,
                        "method_parameters": {},
                    },
                )

        broken = tiny_target_io()
        broken.manifest.pop("weights_sha256")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "target I/O identity"):
            write_candidate_export(
                Path(tmp) / "export",
                method="dflash",
                config={"block_size": 8},
                weights={"fc.weight": torch.zeros(8, 40)},
                target_io=broken,
                method_metadata={
                    "aux_hidden_state_layer_ids": [1, 20, 38, 56, 75],
                    "block_size": 8,
                    "num_speculative_tokens": 7,
                    "sample_from_anchor": False,
                    "method_parameters": {},
                },
            )


if __name__ == "__main__":
    unittest.main()
