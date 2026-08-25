from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from glm_dflash2.target_io import (
    extract_target_io,
    load_frozen_target_io,
    validate_cache_io_compatibility,
)


class TargetIOTest(unittest.TestCase):
    def _source(self, root: Path, *, sharded: bool = False, bad_dtype: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        root.mkdir(parents=True)
        config = {
            "model_type": "glm_moe_dsa",
            "hidden_size": 4,
            "vocab_size": 7,
            "tie_word_embeddings": False,
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "GlmTokenizer", "vocab_size": 7}),
            encoding="utf-8",
        )
        embed = torch.arange(28, dtype=torch.float32).reshape(7, 4).to(torch.bfloat16)
        head = (100 + torch.arange(28, dtype=torch.float32)).reshape(7, 4).to(torch.bfloat16)
        if bad_dtype:
            embed = torch.arange(28, dtype=torch.int64).reshape(7, 4)
        if sharded:
            save_file({"model.embed_tokens.weight": embed}, root / "model-00001-of-00002.safetensors")
            save_file({"lm_head.weight": head}, root / "model-00002-of-00002.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                            "lm_head.weight": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
        else:
            save_file(
                {"model.embed_tokens.weight": embed, "lm_head.weight": head},
                root / "model.safetensors",
            )
        return embed, head

    def test_extracts_only_sharded_dense_io_and_records_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embed, head = self._source(root / "source", sharded=True)
            manifest = extract_target_io(root / "source", root / "io")

            self.assertEqual(manifest["schema"], "glm-dflash2-target-io-v1")
            self.assertEqual(manifest["hidden_size"], 4)
            self.assertEqual(manifest["vocab_size"], 7)
            self.assertEqual(manifest["source_keys"]["embed_tokens"], "model.embed_tokens.weight")
            self.assertEqual(manifest["source_keys"]["lm_head"], "lm_head.weight")
            self.assertEqual(
                manifest["config_sha256"],
                hashlib.sha256((root / "source" / "config.json").read_bytes()).hexdigest(),
            )
            loaded = load_frozen_target_io(root / "io", dtype=torch.bfloat16)
            self.assertTrue(torch.equal(loaded.embed_tokens.weight, embed))
            self.assertTrue(torch.equal(loaded.lm_head.weight, head))
            self.assertFalse(loaded.embed_tokens.weight.requires_grad)
            self.assertFalse(loaded.lm_head.weight.requires_grad)

    def test_rejects_quantized_or_integer_io_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source", bad_dtype=True)
            with self.assertRaisesRegex(ValueError, "floating-point"):
                extract_target_io(root / "source", root / "io")

    def test_rejects_shape_that_disagrees_with_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source")
            config_path = root / "source" / "config.json"
            config = json.loads(config_path.read_text())
            config["hidden_size"] = 5
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "shape"):
                extract_target_io(root / "source", root / "io")

    def test_cache_provenance_must_match_io_and_logical_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source")
            io_manifest = extract_target_io(root / "source", root / "io")
            cache_manifest = {
                "spec": {
                    "layer_ids": [1, 20, 38, 56, 75],
                    "hidden_size": 4,
                    "dtype": "bfloat16",
                    "input_dtype": "int64",
                    "mask_semantics": "dflash_target_token",
                    "schema_version": 1,
                },
                "provenance": {
                    "model_fingerprint": io_manifest["source_model_fingerprint"],
                    "logical_layer_ids": [1, 20, 38, 56, 75],
                    "physical_layer_ids": [2, 21, 39, 57, 76],
                },
            }
            validate_cache_io_compatibility(cache_manifest, io_manifest)
            cache_manifest["provenance"]["model_fingerprint"] = "wrong"
            with self.assertRaisesRegex(ValueError, "model fingerprint"):
                validate_cache_io_compatibility(cache_manifest, io_manifest)

    def test_physical_capture_ids_are_never_accepted_as_logical_ids(self):
        cache_manifest = {
            "spec": {
                "layer_ids": [2, 21, 39, 57, 76],
                "hidden_size": 4,
                "dtype": "bfloat16",
                "mask_semantics": "dflash_target_token",
            },
            "provenance": {
                "model_fingerprint": "same",
                "logical_layer_ids": [1, 20, 38, 56, 75],
                "physical_layer_ids": [2, 21, 39, 57, 76],
            },
        }
        io_manifest = {
            "source_model_fingerprint": "same",
            "hidden_size": 4,
            "vocab_size": 7,
        }
        with self.assertRaisesRegex(ValueError, "logical layer order"):
            validate_cache_io_compatibility(cache_manifest, io_manifest)


if __name__ == "__main__":
    unittest.main()
