from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from glm_dflash2.target_io import (
    extract_target_io,
    load_frozen_target_io,
    validate_cache_io_compatibility,
)


class TargetIOTest(unittest.TestCase):
    def _source(
        self,
        root: Path,
        *,
        sharded: bool = False,
        bad_dtype: bool = False,
        config_updates: dict | None = None,
        head_bias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        root.mkdir(parents=True)
        config = {
            "model_type": "glm_moe_dsa",
            "hidden_size": 4,
            "vocab_size": 7,
            "tie_word_embeddings": False,
            "_commit_hash": "revision-test",
        }
        config.update(config_updates or {})
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
            head_tensors = {"lm_head.weight": head}
            if head_bias:
                head_tensors["lm_head.bias"] = torch.zeros(7, dtype=torch.bfloat16)
            save_file(head_tensors, root / "model-00002-of-00002.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                            "lm_head.weight": "model-00002-of-00002.safetensors",
                            **(
                                {"lm_head.bias": "model-00002-of-00002.safetensors"}
                                if head_bias
                                else {}
                            ),
                        }
                    }
                ),
                encoding="utf-8",
            )
        else:
            tensors = {"model.embed_tokens.weight": embed, "lm_head.weight": head}
            if head_bias:
                tensors["lm_head.bias"] = torch.zeros(7, dtype=torch.bfloat16)
            save_file(tensors, root / "model.safetensors")
        return embed, head

    def test_extracts_only_sharded_dense_io_and_records_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embed, head = self._source(root / "source", sharded=True)
            manifest = extract_target_io(
                root / "source", root / "io", expected_hidden_size=4
            )

            self.assertEqual(manifest["schema"], "glm-drafter-target-io-v2")
            self.assertEqual(manifest["hidden_size"], 4)
            self.assertEqual(manifest["vocab_size"], 7)
            self.assertEqual(manifest["model_revision"], "revision-test")
            self.assertEqual(manifest["logit_transform"], "identity")
            self.assertFalse(manifest["lm_head_bias"])
            self.assertEqual(manifest["source_keys"]["embed_tokens"], "model.embed_tokens.weight")
            self.assertEqual(manifest["source_keys"]["lm_head"], "lm_head.weight")
            self.assertEqual(manifest["tensors"]["embed_tokens"]["shape"], [7, 4])
            self.assertEqual(manifest["tensors"]["lm_head"]["shape"], [7, 4])
            self.assertRegex(manifest["tensors"]["embed_tokens"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(manifest["tensors"]["lm_head"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertNotEqual(
                manifest["tensors"]["embed_tokens"]["sha256"],
                manifest["tensors"]["lm_head"]["sha256"],
            )
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
                extract_target_io(root / "source", root / "io", expected_hidden_size=4)

    def test_rejects_shape_that_disagrees_with_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source")
            config_path = root / "source" / "config.json"
            config = json.loads(config_path.read_text())
            config["hidden_size"] = 5
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "shape"):
                extract_target_io(root / "source", root / "io", expected_hidden_size=4)

    def test_production_extraction_requires_hidden_size_6144(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source")
            with self.assertRaisesRegex(ValueError, "6144"):
                extract_target_io(root / "source", root / "io")

    def test_rejects_bias_scaling_and_soft_cap_instead_of_silently_dropping_them(self):
        cases = (
            ({}, True, "bias"),
            ({"logit_scale": 0.5}, False, "logit_scale"),
            ({"final_logit_softcapping": 30.0}, False, "softcap"),
        )
        for config_updates, head_bias, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._source(
                    root / "source",
                    config_updates=config_updates,
                    head_bias=head_bias,
                )
                with self.assertRaisesRegex(ValueError, message):
                    extract_target_io(
                        root / "source", root / "io", expected_hidden_size=4
                    )

    def test_loader_verifies_each_tensor_content_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source")
            extract_target_io(root / "source", root / "io", expected_hidden_size=4)
            path = root / "io" / "model.safetensors"
            tensors = load_file(path)
            tensors["embed_tokens.weight"][0, 0] += 1
            save_file(tensors, path)
            manifest_path = root / "io" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["weights_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "embed_tokens.*checksum"):
                load_frozen_target_io(root / "io", dtype=torch.bfloat16)

    def test_cache_provenance_must_match_io_and_logical_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source")
            io_manifest = extract_target_io(
                root / "source", root / "io", expected_hidden_size=4
            )
            cache_manifest = {
                "spec": {
                    "layer_ids": [1, 20, 38, 56, 75],
                    "hidden_size": 4,
                    "dtype": "bfloat16",
                    "input_dtype": "int64",
                    "mask_semantics": "dflash_target_token",
                    "schema_version": 2,
                    "final_hidden_semantics": "post_final_norm_lm_head_input",
                },
                "provenance": {
                    "model_fingerprint": io_manifest["source_model_fingerprint"],
                    "model_revision": io_manifest["model_revision"],
                    "tokenizer_fingerprint": io_manifest["tokenizer_fingerprint"],
                    "vocab_size": io_manifest["vocab_size"],
                    "target_hidden_dtype": "bfloat16",
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
            "schema": "glm-drafter-target-io-v2",
            "source_model_fingerprint": "same",
            "model_revision": "revision-test",
            "tokenizer_fingerprint": "tokenizer-test",
            "hidden_size": 4,
            "vocab_size": 7,
            "source_dtypes": {"embed_tokens": "torch.bfloat16", "lm_head": "torch.bfloat16"},
            "logit_transform": "identity",
            "lm_head_bias": False,
        }
        with self.assertRaisesRegex(ValueError, "logical layer order"):
            validate_cache_io_compatibility(cache_manifest, io_manifest)


if __name__ == "__main__":
    unittest.main()
