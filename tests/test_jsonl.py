import tempfile
import unittest
from pathlib import Path

from glm_dflash2.jsonl import OutputShardLock, repair_truncated_jsonl


class JsonlTest(unittest.TestCase):
    def test_repairs_only_truncated_tail(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.jsonl"
            path.write_bytes(b'{"sample_id":"a"}\n{"sample_id":"b"')
            removed = repair_truncated_jsonl(path)
            self.assertGreater(removed, 0)
            self.assertEqual(path.read_text(), '{"sample_id":"a"}\n')
            self.assertEqual(path.read_text(), '{"sample_id":"a"}\n')

    def test_tail_repair_searches_backwards_in_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "large.jsonl"
            complete = b'{"sample_id":"a","payload":"' + (b"x" * 4096) + b'"}\n'
            path.write_bytes(complete + (b"y" * 8192))
            removed = repair_truncated_jsonl(path, chunk_size=128)
            self.assertEqual(removed, 8192)
            self.assertEqual(path.read_bytes(), complete)

    def test_same_output_shard_cannot_be_locked_twice(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "responses.jsonl"
            with OutputShardLock(output):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with OutputShardLock(output):
                        pass

if __name__ == "__main__":
    unittest.main()
