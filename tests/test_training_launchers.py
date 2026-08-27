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
            "BLOCK_SIZE",
            '--block-size "$BLOCK_SIZE"',
            "--num-anchors 64",
            '--gamma "${GAMMA:-$default_gamma}"',
            "--selector-rank 256",
            "--selector-top-k 16",
            "--hidden-size 6144",
            "--intermediate-size 12288",
            "--num-draft-layers 5",
            "train_drafter_offline.py",
            'DSpark requires BLOCK_SIZE=8',
            'dspark_lr=6e-4',
            'dspark_epochs=3',
            'dspark_gamma=4',
            'b8_gamma=4',
            'b16_gamma=7',
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
        self.assertIn("METHOD", gate)
        self.assertIn("BLOCK_SIZE", gate)
        self.assertIn('--block-size "$BLOCK_SIZE"', gate)
        self.assertIn("train_drafter_offline.py", gate)
        self.assertNotIn("train_dflash2_offline.py", gate)

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

    def test_cpu_training_smoke_covers_all_three_aligned_methods(self):
        text = (ROOT / "tools/smoke_train_tiny.py").read_text()
        for method in ("dflash", "dflash2", "dspark"):
            self.assertIn(f'"{method}"', text)
        launcher = (ROOT / "scripts/smoke_no_model.sh").read_text()
        self.assertIn("smoke_train_no_npu.sh", launcher)

    def test_documentation_describes_unified_schema_v2_and_hardware_gate(self):
        readme = (ROOT / "README.md").read_text()
        for value in (
            "DFlash",
            "DFlash2",
            "DSpark",
            "aux_hidden_states.bin",
            "target_final_hidden.bin",
            "64/64",
            "calibrate_hidden_capture_gate.py",
        ):
            self.assertIn(value, readme)
        runbook = (ROOT / "docs/ASCEND_910B_RUNBOOK.md").read_text()
        self.assertIn("real 910B", runbook)
        self.assertIn("scripts/train_drafter.sh", runbook)
        self.assertIn("candidate-not-deployable", runbook)
        self.assertIn("deploy_attestation.json", runbook)
        self.assertIn("Legacy v1", runbook)


if __name__ == "__main__":
    unittest.main()
