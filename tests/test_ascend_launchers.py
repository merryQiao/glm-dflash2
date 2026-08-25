from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AscendLauncherTest(unittest.TestCase):
    def test_stage_a_launcher_passes_explicit_ascend_backend(self):
        script = (ROOT / "scripts/run_stage_a_trajectories.sh").read_text()
        self.assertIn('--device "${DEVICE:-npu}"', script)
        self.assertIn('--attention-backend "${ATTENTION_BACKEND:-ascend}"', script)

    def test_stage_a_defaults_to_official_glm52_sampling(self):
        script = (ROOT / "scripts/run_stage_a_trajectories.sh").read_text()
        self.assertIn('--temperature "${TEMPERATURE:-1.0}"', script)
        self.assertIn('--top-p "${TOP_P:-0.95}"', script)
        self.assertIn('--top-k "${TOP_K:--1}"', script)

    def test_stage_b_launcher_passes_explicit_ascend_backend(self):
        script = (ROOT / "scripts/run_stage_b_hidden.sh").read_text()
        self.assertIn('--device "${DEVICE:-npu}"', script)
        self.assertIn('--attention-backend "${ATTENTION_BACKEND:-ascend}"', script)


if __name__ == "__main__":
    unittest.main()
