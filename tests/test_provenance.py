import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from glm_dflash2.provenance import dataset_fingerprint, local_model_fingerprint


class ProvenanceTest(unittest.TestCase):
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

    def test_model_path_requires_tokenizer_and_weight_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "tokenizer"):
                local_model_fingerprint(root)

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
