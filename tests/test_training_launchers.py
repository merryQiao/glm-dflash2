from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainingLaunchersTest(unittest.TestCase):
    def test_unified_training_launcher_has_fixed_recipe_and_no_target_backbone(self):
        text = (ROOT / "scripts/train_glm52_drafter_910b.sh").read_text()
        for value in (
            "METHOD",
            "CACHE_DIR",
            "TARGET_IO_DIR",
            "MASK_TOKEN_ID",
            "--device npu",
            "--block-size 16",
            "--num-anchors 64",
            "--gamma 7",
            "--selector-rank 256",
            "--selector-top-k 16",
            "--hidden-size 6144",
            "--intermediate-size 12288",
            "--num-draft-layers 5",
            "train_drafter_offline.py",
        ):
            self.assertIn(value, text)
        self.assertNotIn("TARGET_MODEL", text)

    def test_old_dflash2_launcher_is_only_a_unified_compatibility_wrapper(self):
        text = (ROOT / "scripts/train_glm52_dflash2_910b.sh").read_text()
        self.assertIn("METHOD=dflash2", text)
        self.assertIn("train_glm52_drafter_910b.sh", text)
        self.assertNotIn("train_dflash2_offline.py", text)

    def test_extract_and_hardware_gate_are_explicit(self):
        extract = (ROOT / "scripts/extract_glm52_io.sh").read_text()
        self.assertIn("MODEL_PATH", extract)
        self.assertIn("extract_target_io.py", extract)
        gate = (ROOT / "scripts/gate_train_2rank_910b.sh").read_text()
        self.assertIn("--nproc_per_node=2", gate)
        self.assertIn("direct", gate)
        self.assertIn("resumed", gate)
        self.assertIn("compare", gate)
        self.assertIn("gate-result.json", gate)

    def test_training_launcher_supports_multinode_torchrun(self):
        text = (ROOT / "scripts/train_glm52_drafter_910b.sh").read_text()
        for value in ("NNODES", "NODE_RANK", "MASTER_ADDR", "--nnodes", "--node_rank"):
            self.assertIn(value, text)

    def test_requirements_do_not_install_generic_torch(self):
        lines = [
            line.strip()
            for line in (ROOT / "requirements-train.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertFalse(any(line.startswith("torch") for line in lines))
        self.assertIn("transformers==4.57.3", lines)


if __name__ == "__main__":
    unittest.main()
