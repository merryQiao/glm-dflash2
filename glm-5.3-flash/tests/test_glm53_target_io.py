from __future__ import annotations

import hashlib
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from glm53_drafters.target_io import (
    TargetCheckpointContract,
    _extract_target_io,
    _load_frozen_target_io,
    _local_model_identity,
    _validate_cache_io_compatibility,
    extract_target_io,
    load_frozen_target_io,
    validate_cache_io_compatibility,
    validate_target_checkpoint_config,
)


ROOT = Path(__file__).resolve().parents[1]
TINY_CONTRACT = TargetCheckpointContract(
    model_type="glm5_next",
    num_hidden_layers=3,
    hidden_size=4,
    vocab_size=7,
    rms_norm_eps=1e-5,
    dtype="bfloat16",
)


class GLM53TargetIOTest(unittest.TestCase):
    def _source(
        self,
        root: Path,
        *,
        sharded: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        head_bias: bool = False,
        tied: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        root.mkdir(parents=True)
        config = {
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "num_hidden_layers": 3,
                "hidden_size": 4,
                "vocab_size": 7,
                "rms_norm_eps": 1e-5,
                "dtype": "bfloat16",
                "tie_word_embeddings": tied,
            },
            "tie_word_embeddings": tied,
            "_commit_hash": "revision-test",
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "GlmTokenizer", "vocab_size": 7}),
            encoding="utf-8",
        )
        (root / "tokenizer.json").write_text(
            json.dumps(
                {
                    "added_tokens": [
                        {"id": 5, "content": "[MASK]", "special": True}
                    ]
                }
            ),
            encoding="utf-8",
        )
        embed = torch.arange(28, dtype=torch.float32).reshape(7, 4).to(dtype)
        head = (100 + torch.arange(28, dtype=torch.float32)).reshape(7, 4).to(dtype)
        tensors = {
            "model.language_model.embed_tokens.weight": embed,
            "lm_head.weight": head,
        }
        if head_bias:
            tensors["lm_head.bias"] = torch.zeros(7, dtype=dtype)
        if sharded:
            save_file(
                {"model.language_model.embed_tokens.weight": embed},
                root / "model-00001-of-00002.safetensors",
            )
            save_file(
                {key: value for key, value in tensors.items() if "embed_tokens" not in key},
                root / "model-00002-of-00002.safetensors",
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            key: (
                                "model-00001-of-00002.safetensors"
                                if "embed_tokens" in key
                                else "model-00002-of-00002.safetensors"
                            )
                            for key in tensors
                        }
                    }
                ),
                encoding="utf-8",
            )
        else:
            save_file(tensors, root / "model.safetensors")
        return embed, head

    def _cache_manifest(self, identity: dict[str, object]) -> dict[str, object]:
        return {
            "status": "frozen",
            "production_eligible": True,
            "cache_identity": "cache-sha",
            "spec": {
                "schema_version": 2,
                "layer_ids": [1, 3],
                "hidden_size": 4,
                "target_num_hidden_layers": 3,
                "vocab_size": 7,
                "dtype": "bfloat16",
                "mask_semantics": "dflash_target_token",
                "final_hidden_semantics": "post_final_norm_lm_head_input",
            },
            "provenance": {
                **identity,
                "logical_layer_ids": [1, 3],
                "physical_layer_ids": [2, 4],
                "target_hidden_dtype": "bfloat16",
            },
        }

    def test_nested_text_config_extracts_untied_dense_bf16_and_binds_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embed, head = self._source(root / "source", sharded=True)
            identity = _local_model_identity(root / "source", TINY_CONTRACT)
            cache = self._cache_manifest(identity)
            manifest = _extract_target_io(
                root / "source",
                root / "io",
                contract=TINY_CONTRACT,
                cache_manifest=cache,
                expected_layer_ids=(1, 3),
            )
            self.assertEqual(manifest["schema"], "glm53-target-io-v3")
            self.assertEqual(
                manifest["mask_token"],
                {"token": "[MASK]", "token_id": 5, "special": True},
            )
            self.assertEqual(manifest["hidden_size"], 4)
            self.assertEqual(manifest["vocab_size"], 7)
            self.assertEqual(
                manifest["target_checkpoint_contract"],
                {
                    "model_type": "glm5_next",
                    "num_hidden_layers": 3,
                    "hidden_size": 4,
                    "vocab_size": 7,
                    "rms_norm_eps": 1e-5,
                    "dtype": "bfloat16",
                },
            )
            self.assertEqual(manifest["model_revision"], "revision-test")
            self.assertEqual(manifest["hidden_cache_identity"], "cache-sha")
            self.assertEqual(
                manifest["source_keys"]["embed_tokens"],
                "model.language_model.embed_tokens.weight",
            )
            self.assertEqual(manifest["source_keys"]["lm_head"], "lm_head.weight")
            self.assertNotEqual(
                manifest["tensors"]["embed_tokens"]["sha256"],
                manifest["tensors"]["lm_head"]["sha256"],
            )
            with self.assertRaisesRegex(ValueError, "45|4096|154880"):
                load_frozen_target_io(root / "io")
            loaded = _load_frozen_target_io(
                root / "io", contract=TINY_CONTRACT
            )
            self.assertTrue(torch.equal(loaded.embed_tokens.weight, embed))
            self.assertTrue(torch.equal(loaded.lm_head.weight, head))
            self.assertFalse(loaded.embed_tokens.weight.requires_grad)
            self.assertFalse(loaded.lm_head.weight.requires_grad)

    def test_production_defaults_require_154880_by_4096_without_allocating_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source")
            with self.assertRaisesRegex(ValueError, "45|4096|154880"):
                extract_target_io(root / "source", root / "io")

    def test_exact_nested_glm5next_config_rejects_every_off_by_one_or_wrong_field(self):
        base = {
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "num_hidden_layers": 3,
                "hidden_size": 4,
                "vocab_size": 7,
                "rms_norm_eps": 1e-5,
                "dtype": "bfloat16",
                "tie_word_embeddings": False,
            },
        }
        resolved = validate_target_checkpoint_config(base, contract=TINY_CONTRACT)
        self.assertEqual(resolved["model_type"], "glm5_next")
        self.assertEqual(resolved["text_model_type"], "glm5_next_text")
        cases = (
            ("num_hidden_layers", 2, "45|layers|3"),
            ("num_hidden_layers", 4, "45|layers|3"),
            ("hidden_size", 3, "hidden"),
            ("hidden_size", 5, "hidden"),
            ("vocab_size", 6, "vocab"),
            ("vocab_size", 8, "vocab"),
            ("rms_norm_eps", 1e-6, "RMS"),
            ("dtype", "float16", "BF16"),
        )
        for field, wrong, message in cases:
            with self.subTest(field=field, wrong=wrong):
                config = copy.deepcopy(base)
                config["text_config"][field] = wrong
                with self.assertRaisesRegex(ValueError, message):
                    validate_target_checkpoint_config(
                        config, contract=TINY_CONTRACT
                    )
        wrong_root = copy.deepcopy(base)
        wrong_root["model_type"] = "glm5"
        with self.assertRaisesRegex(ValueError, "root.*glm5_next"):
            validate_target_checkpoint_config(wrong_root, contract=TINY_CONTRACT)
        wrong_text = copy.deepcopy(base)
        wrong_text["text_config"]["model_type"] = "glm5_next"
        with self.assertRaisesRegex(ValueError, "text.*glm5_next_text"):
            validate_target_checkpoint_config(wrong_text, contract=TINY_CONTRACT)
        missing_dtype = copy.deepcopy(base)
        missing_dtype["text_config"].pop("dtype")
        missing_dtype["text_config"]["torch_dtype"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate_target_checkpoint_config(missing_dtype, contract=TINY_CONTRACT)
        conflicting_dtype = copy.deepcopy(base)
        conflicting_dtype["text_config"]["torch_dtype"] = "float16"
        with self.assertRaisesRegex(ValueError, "conflict"):
            validate_target_checkpoint_config(conflicting_dtype, contract=TINY_CONTRACT)
        missing_tie = copy.deepcopy(base)
        missing_tie["text_config"].pop("tie_word_embeddings")
        with self.assertRaisesRegex(ValueError, "tie_word_embeddings"):
            validate_target_checkpoint_config(
                missing_tie, contract=TINY_CONTRACT
            )
        outer_only_tie = copy.deepcopy(base)
        outer_only_tie["tie_word_embeddings"] = False
        outer_only_tie["text_config"].pop("tie_word_embeddings")
        with self.assertRaisesRegex(ValueError, "tie_word_embeddings"):
            validate_target_checkpoint_config(
                outer_only_tie, contract=TINY_CONTRACT
            )

    def test_exact_checkpoint_contract_rejects_quantization_declarations(self):
        base = {
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "num_hidden_layers": 3,
                "hidden_size": 4,
                "vocab_size": 7,
                "rms_norm_eps": 1e-5,
                "dtype": "bfloat16",
                "tie_word_embeddings": False,
            },
        }
        declarations = (
            ("quantization_config", {"quant_method": "int8"}),
            ("quantization_config", {}),
            ("quantization_config", None),
            ("quantization", "int8"),
            ("load_in_4bit", True),
            ("load_in_4bit", False),
            ("load_in_8bit", True),
        )
        for field, value in declarations:
            for location in ("root", "text"):
                with self.subTest(field=field, location=location):
                    config = copy.deepcopy(base)
                    target = config if location == "root" else config["text_config"]
                    target[field] = value
                    with self.assertRaisesRegex(ValueError, "quant"):
                        validate_target_checkpoint_config(
                            config, contract=TINY_CONTRACT
                        )

    def test_rejects_non_bf16_tied_or_biased_target_io(self):
        cases = (
            ({"dtype": torch.float32}, "BF16"),
            ({"tied": True}, "untied"),
            ({"head_bias": True}, "bias"),
        )
        for options, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._source(root / "source", **options)
                with self.assertRaisesRegex(ValueError, message):
                    _extract_target_io(
                        root / "source",
                        root / "io",
                        contract=TINY_CONTRACT,
                    )

    def test_loader_verifies_file_and_tensor_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source")
            _extract_target_io(
                root / "source",
                root / "io",
                contract=TINY_CONTRACT,
            )
            path = root / "io/model.safetensors"
            tensors = load_file(path)
            tensors["embed_tokens.weight"][0, 0] += 1
            save_file(tensors, path)
            manifest_path = root / "io/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["weights_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "embed_tokens.*checksum"):
                _load_frozen_target_io(root / "io", contract=TINY_CONTRACT)

    def test_loader_rejects_manifest_checkpoint_contract_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source")
            _extract_target_io(
                root / "source",
                root / "io",
                contract=TINY_CONTRACT,
            )
            manifest_path = root / "io/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["target_checkpoint_contract"]["num_hidden_layers"] = 4
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "checkpoint contract"):
                _load_frozen_target_io(root / "io", contract=TINY_CONTRACT)

    def test_loader_rejects_legacy_maskless_target_io_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source")
            _extract_target_io(
                root / "source",
                root / "io",
                contract=TINY_CONTRACT,
            )
            manifest_path = root / "io/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema"] = "glm53-target-io-v2"
            manifest.pop("mask_token", None)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "schema v3|mask"):
                _load_frozen_target_io(root / "io", contract=TINY_CONTRACT)

    def test_cache_parity_rejects_identity_or_logical_layer_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source")
            identity = _local_model_identity(root / "source", TINY_CONTRACT)
            io_manifest = _extract_target_io(
                root / "source",
                root / "io",
                contract=TINY_CONTRACT,
            )
            cache = self._cache_manifest(identity)
            _validate_cache_io_compatibility(
                cache,
                io_manifest,
                contract=TINY_CONTRACT,
                expected_layer_ids=(1, 3),
            )
            cache["provenance"]["model_fingerprint"] = "wrong"
            with self.assertRaisesRegex(ValueError, "model fingerprint"):
                _validate_cache_io_compatibility(
                    cache,
                    io_manifest,
                    contract=TINY_CONTRACT,
                    expected_layer_ids=(1, 3),
                )
            cache = self._cache_manifest(identity)
            cache["spec"]["layer_ids"] = [2, 4]
            with self.assertRaisesRegex(ValueError, "logical layer order"):
                _validate_cache_io_compatibility(
                    cache,
                    io_manifest,
                    contract=TINY_CONTRACT,
                    expected_layer_ids=(1, 3),
                )

    def test_cache_binder_rejects_non_exact_target_io_artifact_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root / "source")
            identity = _local_model_identity(root / "source", TINY_CONTRACT)
            exact = _extract_target_io(
                root / "source", root / "io", contract=TINY_CONTRACT
            )
            cache = self._cache_manifest(identity)
            mutations = (
                ("status", "building"),
                ("dtype", "float16"),
                ("untied", False),
                ("lm_head_bias", True),
                ("logit_transform", "scaled"),
            )
            for field, value in mutations:
                with self.subTest(field=field):
                    manifest = copy.deepcopy(exact)
                    manifest[field] = value
                    with self.assertRaisesRegex(ValueError, "target-I/O"):
                        _validate_cache_io_compatibility(
                            cache,
                            manifest,
                            contract=TINY_CONTRACT,
                            expected_layer_ids=(1, 3),
                        )

    def test_nested_text_config_non_identity_logits_are_rejected(self):
        for field, value in (
            ("logit_scale", 0.5),
            ("output_logits_scale", 2.0),
            ("final_logit_softcapping", 30.0),
            ("logits_soft_cap", 1.0),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._source(root / "source")
                config_path = root / "source/config.json"
                config = json.loads(config_path.read_text())
                config["text_config"][field] = value
                config_path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "identity|softcap"):
                    _extract_target_io(
                        root / "source", root / "io", contract=TINY_CONTRACT
                    )

    def test_cli_help_and_shell_launcher_are_standalone(self):
        tool = ROOT / "tools/extract_target_io.py"
        launcher = ROOT / "scripts/extract_target_io.sh"
        subprocess.run(
            [sys.executable, str(tool), "--help"],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        self.assertIn("GLM-5.3-Flash-BF16", launcher.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
