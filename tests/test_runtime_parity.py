from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.compare_sglang_runtime import compare_captures, required_tensors_for_method


class RuntimeParityTest(unittest.TestCase):
    def test_method_dispatch_requires_common_and_method_specific_outputs(self):
        self.assertEqual(
            required_tensors_for_method("dflash"),
            ("backbone_logits", "final_path"),
        )
        self.assertEqual(
            required_tensors_for_method("dflash2"),
            (
                "backbone_logits",
                "candidate_ids",
                "candidate_scores",
                "pair_scores",
                "final_path",
            ),
        )
        self.assertEqual(
            required_tensors_for_method("dspark"),
            (
                "backbone_logits",
                "markov_scores",
                "confidence_logits",
                "final_path",
            ),
        )

    def test_comparison_locks_token_anchor_position_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {
                "input_ids": np.array([1, 2, 3], dtype=np.int64),
                "anchor_positions": np.array([1], dtype=np.int64),
                "position_ids": np.array([1, 2, 3], dtype=np.int64),
                "cache_fingerprint": np.array("cache-sha"),
                "backbone_logits": np.array([[0.1, 0.2]], dtype=np.float32),
                "final_path": np.array([2], dtype=np.int64),
            }
            left = root / "left.npz"
            right = root / "right.npz"
            np.savez(left, **common)
            np.savez(right, **common)
            result = compare_captures(left, right, method="dflash", atol=1e-6, rtol=1e-6)
            self.assertTrue(result["passed"])

            changed = dict(common)
            changed["anchor_positions"] = np.array([0], dtype=np.int64)
            np.savez(right, **changed)
            mismatch = compare_captures(
                left, right, method="dflash", atol=1e-6, rtol=1e-6
            )
            self.assertFalse(mismatch["passed"])
            self.assertFalse(mismatch["results"]["anchor_positions"]["passed"])


if __name__ == "__main__":
    unittest.main()
