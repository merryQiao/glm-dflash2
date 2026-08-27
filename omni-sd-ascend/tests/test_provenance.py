from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omni_sd.provenance import artifact_record, sha256_file, verify_artifact_record


class ProvenanceTests(unittest.TestCase):
    def test_artifact_checksum_detects_content_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"first")
            record = artifact_record(path, relative_to=Path(directory))
            self.assertEqual(record["sha256"], sha256_file(path))
            verify_artifact_record(record, root=Path(directory))
            path.write_bytes(b"second")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_artifact_record(record, root=Path(directory))


if __name__ == "__main__":
    unittest.main()
