import json
import os
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from glm53_stage_a.provenance import (
    dataset_fingerprint,
    load_endpoint_manifest_attestation,
    local_model_fingerprint,
    target_vocab_size,
    tokenizer_fingerprint,
)


class ProvenanceTest(unittest.TestCase):
    def test_external_endpoint_manifest_is_bound_to_local_model_and_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "endpoint.json"
            value = {
                "schema": "glm-sglang-endpoint-v1",
                "served_model_name": "GLM-5.3-Flash-BF16",
                "model_fingerprint": "model-sha",
                "tokenizer_fingerprint": "tokenizer-sha",
                "dtype": "bfloat16",
                "runtime": {
                    "sglang_version": "0.5.16",
                    "cann_version": "8.3.RC1",
                    "image_digest": "sha256:image",
                    "tp_size": 16,
                    "device": "npu",
                    "attention_backend": "ascend",
                    "reasoning_parser": "glm45",
                    "tool_call_parser": "glm47",
                    "context_length": 131072,
                    "max_total_tokens": 131072,
                    "quantization": None,
                    "moe_a2a_backend": None,
                    "deepep_mode": None,
                },
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            loaded = load_endpoint_manifest_attestation(
                path,
                expected_model_fingerprint="model-sha",
                expected_tokenizer_fingerprint="tokenizer-sha",
                expected_served_model_name="GLM-5.3-Flash-BF16",
            )
            self.assertEqual(loaded["manifest"], value)
            self.assertRegex(loaded["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(loaded["weight_identity_status"], "operator_attested")
            self.assertFalse(loaded["weight_identity_verified"])

            value["model_fingerprint"] = "other"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model fingerprint"):
                load_endpoint_manifest_attestation(
                    path,
                    expected_model_fingerprint="model-sha",
                    expected_tokenizer_fingerprint="tokenizer-sha",
                    expected_served_model_name="GLM-5.3-Flash-BF16",
                )

    def test_external_endpoint_manifest_fails_closed_on_missing_runtime_identity(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "endpoint.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "glm-sglang-endpoint-v1",
                        "served_model_name": "GLM-5.3-Flash-BF16",
                        "model_fingerprint": "model-sha",
                        "tokenizer_fingerprint": "tokenizer-sha",
                        "dtype": "bfloat16",
                        "runtime": {"tp_size": 16},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "runtime identity"):
                load_endpoint_manifest_attestation(
                    path,
                    expected_model_fingerprint="model-sha",
                    expected_tokenizer_fingerprint="tokenizer-sha",
                    expected_served_model_name="GLM-5.3-Flash-BF16",
                )

    def test_model_fingerprint_changes_when_weight_artifact_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}")
            (root / "tokenizer_config.json").write_text("{}")
            weight = root / "model-00001-of-00001.safetensors"
            weight.write_bytes(b"first")
            first = local_model_fingerprint(root)
            weight.write_bytes(b"second-weight-version")
            second = local_model_fingerprint(root)
            self.assertNotEqual(first, second)

    def test_model_fingerprint_hashes_weight_middle_not_only_metadata_and_edges(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}")
            (root / "tokenizer_config.json").write_text("{}")
            weight = root / "model.safetensors"
            weight.write_bytes(b"a" * (4 * 1024 * 1024))
            stat = weight.stat()
            first = local_model_fingerprint(root)
            with weight.open("r+b") as handle:
                handle.seek(2 * 1024 * 1024)
                handle.write(b"changed-middle")
            os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            second = local_model_fingerprint(root)
            self.assertNotEqual(first, second)

    def test_model_path_requires_tokenizer_and_weight_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "tokenizer"):
                local_model_fingerprint(root)

    def test_glm53_vocab_size_supports_nested_text_config(self):
        self.assertEqual(target_vocab_size({"vocab_size": 10}), 10)
        self.assertEqual(
            target_vocab_size({"text_config": {"vocab_size": 154880}}), 154880
        )
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            target_vocab_size({"text_config": {}})

    def test_chat_template_and_processor_are_bound_to_both_fingerprints(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}")
            (root / "tokenizer_config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"weights")
            template = root / "chat_template.jinja"
            processor = root / "preprocessor_config.json"
            template.write_text("template-a")
            processor.write_text('{"processor":"a"}')

            model_a = local_model_fingerprint(root)
            tokenizer_a = tokenizer_fingerprint(root)
            template.write_text("template-b")
            model_b = local_model_fingerprint(root)
            tokenizer_b = tokenizer_fingerprint(root)
            self.assertNotEqual(model_a, model_b)
            self.assertNotEqual(tokenizer_a, tokenizer_b)

            processor.write_text('{"processor":"b"}')
            self.assertNotEqual(model_b, local_model_fingerprint(root))
            self.assertNotEqual(tokenizer_b, tokenizer_fingerprint(root))

            model_c = local_model_fingerprint(root)
            tokenizer_c = tokenizer_fingerprint(root)
            (root / "video_preprocessor_config.json").write_text('{"video":"a"}')
            self.assertNotEqual(model_c, local_model_fingerprint(root))
            self.assertNotEqual(tokenizer_c, tokenizer_fingerprint(root))

    def test_dataset_fingerprint_is_bound_to_download_revision_and_parquet_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "processed"
            data.mkdir()
            (root / "DOWNLOAD_REVISION").write_text("repo/name\nrevision-a\n")
            pq.write_table(pa.table({"id": ["a"]}), data / "part.parquet")
            first = dataset_fingerprint(data)
            (root / "DOWNLOAD_REVISION").write_text("repo/name\nrevision-b\n")
            second = dataset_fingerprint(data)
            self.assertNotEqual(first.digest, second.digest)
            self.assertEqual(second.revision, "revision-b")


if __name__ == "__main__":
    unittest.main()
