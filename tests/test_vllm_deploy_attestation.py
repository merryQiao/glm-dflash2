from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from glm_dflash2.vllm_ascend.export_dflash import export_dflash
from glm_dflash2.vllm_ascend.parity import (
    attest_candidate,
    candidate_binding,
    validate_deploy_attestation,
)
from tests.export_test_utils import flip_one_byte, tiny_config, tiny_target_io
from tests.test_vllm_capability import runtime_identity
from tools.train_drafter_offline import build_method_model


def parity_results(binding, runtime=None, **gate_overrides):
    gates = {
        "candidate_load": {"passed": True},
        "logits": {"passed": True, "max_abs_error": 0.001},
        "proposals": {"passed": True},
        "token_ids": {"passed": True},
        "rejection_sampling": {"passed": True, "mode": "standard"},
        "speculative_counters": {"passed": True, "draft_tokens": 10},
    }
    gates.update(gate_overrides)
    return {
        "schema": "glm-vllm-ascend-parity-results-v1",
        "candidate_binding": binding,
        "runtime_identity": runtime or runtime_identity(),
        "fixture_id": "fixture-sha",
        "thresholds": {"max_abs_error": 0.01},
        "gates": gates,
    }


class DeployAttestationTest(unittest.TestCase):
    def _candidate(self, root: Path):
        config = tiny_config(block_size=8)
        model = build_method_model("dflash", config, markov_rank=4)
        export_dflash(root, config=config, state_dict=model.state_dict(), target_io=tiny_target_io())

    def test_attestation_binds_candidate_runtime_and_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            self._candidate(root)
            binding = candidate_binding(root)
            attestation = attest_candidate(
                root,
                runtime_identity=runtime_identity(),
                parity_results=parity_results(binding),
            )
            self.assertEqual(attestation["schema"], "glm-vllm-ascend-deploy-attestation-v1")
            self.assertEqual(attestation["fixture_id"], "fixture-sha")
            validated = validate_deploy_attestation(root, active_runtime=runtime_identity())
            self.assertEqual(validated["candidate_binding"]["method"], "dflash")
            manifest = json.loads((root / "export_manifest.json").read_text())
            self.assertEqual(manifest["status"], "runtime-attested")

    def test_failed_gate_cannot_create_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            self._candidate(root)
            binding = candidate_binding(root)
            with self.assertRaisesRegex(ValueError, "logits"):
                attest_candidate(
                    root,
                    runtime_identity=runtime_identity(),
                    parity_results=parity_results(binding, logits={"passed": False}),
                )
            self.assertFalse((root / "deploy_attestation.json").exists())

    def test_candidate_or_runtime_mutation_invalidates_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            self._candidate(root)
            binding = candidate_binding(root)
            attest_candidate(
                root,
                runtime_identity=runtime_identity(),
                parity_results=parity_results(binding),
            )
            with self.assertRaisesRegex(ValueError, "runtime"):
                validate_deploy_attestation(
                    root, active_runtime=runtime_identity(cann_version="8.6.0")
                )
            flip_one_byte(root / "config.json")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_deploy_attestation(root, active_runtime=runtime_identity())


if __name__ == "__main__":
    unittest.main()
