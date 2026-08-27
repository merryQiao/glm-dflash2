from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def test_generation_launcher_uses_ascend_visibility_not_cuda(self):
        script = (ROOT / "scripts/generate_thinker_trajectories_ascend.sh").read_text()
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES", script)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", script)
        self.assertNotIn("torchrun", script)

    def test_a3_enables_aiv_but_a2_does_not_force_it(self):
        script = (ROOT / "scripts/common_ascend.sh").read_text()
        self.assertIn('if [[ "${ASCEND_HARDWARE}" == "a3" ]]', script)
        self.assertIn("HCCL_OP_EXPANSION_MODE", script)

    def test_profile_launcher_uses_shared_ascend_contract(self):
        script = (ROOT / "scripts/profile_thinker_ascend.sh").read_text()
        self.assertIn("common_ascend.sh", script)
        self.assertIn("inference_qwen3-omni.py", script)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", script)
        self.assertNotIn("torchrun", script)


if __name__ == "__main__":
    unittest.main()
