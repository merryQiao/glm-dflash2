from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AscendLauncherTest(unittest.TestCase):
    def test_canonical_launchers_are_independent(self):
        stage_a = (ROOT / "scripts/generate_trajectories.sh").read_text()
        stage_b = (ROOT / "scripts/extract_hidden_sglang.sh").read_text()
        train = (ROOT / "scripts/train_drafter.sh").read_text()

        self.assertIn("run_stage_a_trajectories.sh", stage_a)
        self.assertNotIn("extract_hidden_sglang.py", stage_a)
        self.assertNotIn("train_drafter_offline.py", stage_a)

        self.assertIn("run_stage_b_hidden.sh", stage_b)
        self.assertIn("TRAJECTORY_JSONL", stage_b)
        self.assertNotIn("generate_trajectories.py", stage_b)
        self.assertNotIn("train_drafter_offline.py", stage_b)

        self.assertIn("train_glm52_drafter_910b.sh", train)
        self.assertIn("CACHE_DIR", train)
        self.assertNotIn("generate_trajectories.py", train)
        self.assertNotIn("extract_hidden_sglang.py", train)

    def test_stage_a_launcher_passes_explicit_ascend_backend(self):
        script = (ROOT / "scripts/run_stage_a_trajectories.sh").read_text()
        self.assertIn('--device "${DEVICE:-npu}"', script)
        self.assertIn('--attention-backend "${ATTENTION_BACKEND:-ascend}"', script)

    def test_stage_a_defaults_to_official_glm52_sampling(self):
        script = (ROOT / "scripts/run_stage_a_trajectories.sh").read_text()
        self.assertIn('--temperature "${TEMPERATURE:-1.0}"', script)
        self.assertIn('--top-p "${TOP_P:-0.95}"', script)
        self.assertIn('--top-k "${TOP_K:--1}"', script)

    def test_stage_a_uses_bounded_concurrency_defaults(self):
        script = (ROOT / "scripts/run_stage_a_trajectories.sh").read_text()
        self.assertIn('--workers "${WORKERS:-8}"', script)
        self.assertIn('--max-running-requests "${MAX_RUNNING_REQUESTS:-2}"', script)
        self.assertIn('--max-total-tokens "${MAX_TOTAL_TOKENS:-131072}"', script)

    def test_stage_b_launcher_passes_explicit_ascend_backend(self):
        script = (ROOT / "scripts/run_stage_b_hidden.sh").read_text()
        self.assertIn('--device "${DEVICE:-npu}"', script)
        self.assertIn('--attention-backend "${ATTENTION_BACKEND:-ascend}"', script)


if __name__ == "__main__":
    unittest.main()
