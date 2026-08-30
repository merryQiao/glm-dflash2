from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from glm53_drafters.hidden_cache import PackedHiddenDataset
from glm53_drafters.hidden_capture import (
    CaptureAttestation,
    CaptureTap,
    TargetHiddenCapture,
)
from glm53_drafters.hidden_extraction import (
    estimate_packed_cache_bytes,
    extract_trajectory_cache,
    read_frozen_trajectories,
)


class FakeRunner:
    hidden_size = 3
    physical_layer_ids = (2, 4)
    backend_metadata = {"backend": "fake", "version": "test"}
    capture_mapping = (
        CaptureTap("fake", 1, "hidden_states[2]", "post_decoder_block"),
        CaptureTap("fake", 3, "hidden_states[4]", "post_decoder_block"),
    )

    def extract(self, input_ids):
        tokens = len(input_ids)
        return TargetHiddenCapture(
            aux_hidden_states=torch.arange(tokens * 2 * 3)
            .reshape(tokens, 2, 3)
            .to(torch.bfloat16),
            target_final_hidden=torch.ones(tokens, 3, dtype=torch.bfloat16),
            capture_mapping=self.capture_mapping,
            attestation=CaptureAttestation(
                passed=True,
                token_count=tokens,
                logical_layer_ids=(1, 3),
                physical_layer_ids=(2, 4),
                independent_tap_paths=("layers[1]", "layers[3]"),
                native_logits_path="fake.compute_logits/global",
                aux_max_abs_error=0.0,
                aux_max_rel_error=0.0,
                logits_max_abs_error=0.0,
                logits_max_rel_error=0.0,
                reason="",
            ),
        )


class GLM53HiddenExtractionTest(unittest.TestCase):
    def _write_source(
        self,
        root: Path,
        *,
        status: str = "frozen",
        production_eligible: bool = True,
        samples: int = 1,
    ) -> Path:
        path = root / "trajectories.jsonl"
        rows = []
        for index in range(samples):
            rows.append(
                {
                    "id": f"sample-{index}",
                    "stage_a_complete": True,
                    "generation_route": "workspace_task",
                    "input_ids": [1, 2, 3],
                    "loss_mask": [0, 1, 1],
                    "source_metadata": {"selected_source_index": index},
                    "token_contract": {"mask_semantics": "dflash_target_token"},
                }
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        raw = path.read_bytes()
        manifest = {
            "schema_version": 3,
            "status": status,
            "production_eligible": production_eligible,
            "committed_ids": samples,
            "jsonl_bytes": len(raw),
            "jsonl_sha256": hashlib.sha256(raw).hexdigest(),
            "model_fingerprint": "model-sha",
            "model_revision": "revision-a",
            "tokenizer_fingerprint": "tokenizer-sha",
            "vocab_size": 154880,
        }
        path.with_suffix(path.suffix + ".manifest.json").write_text(
            json.dumps(manifest)
        )
        return path

    def test_storage_estimate_includes_five_aux_final_ids_and_mask(self):
        self.assertEqual(
            estimate_packed_cache_bytes(10, num_layers=5, hidden_size=4096),
            10 * (6 * 4096 * 2 + 9),
        )

    def test_stage_a_reader_requires_frozen_complete_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for status in ("running", "partial", "smoke_failed"):
                path = self._write_source(root, status=status)
                with self.subTest(status=status), self.assertRaisesRegex(
                    ValueError, "not a frozen production artifact|smoke_failed"
                ):
                    list(read_frozen_trajectories(path))

    def test_stage_a_reader_requires_schema_v3_and_canonical_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_source(Path(tmp))
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            manifest = json.loads(manifest_path.read_text())
            manifest["schema_version"] = 2
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "schema"):
                list(read_frozen_trajectories(path))

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_source(Path(tmp))
            row = json.loads(path.read_text())
            row.pop("generation_route")
            path.write_text(json.dumps(row) + "\n")
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            manifest = json.loads(manifest_path.read_text())
            raw = path.read_bytes()
            manifest["jsonl_bytes"] = len(raw)
            manifest["jsonl_sha256"] = hashlib.sha256(raw).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "generation route"):
                list(read_frozen_trajectories(path))

    def test_smoke_unverified_requires_explicit_bounded_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_source(
                root,
                status="smoke_unverified",
                production_eligible=False,
                samples=2,
            )
            with self.assertRaisesRegex(ValueError, "explicit smoke"):
                list(read_frozen_trajectories(path))
            self.assertEqual(
                len(
                    list(
                        read_frozen_trajectories(
                            path, allow_smoke_unverified=True, smoke_max_samples=50
                        )
                    )
                ),
                2,
            )
            path = self._write_source(
                root,
                status="smoke_unverified",
                production_eligible=False,
                samples=51,
            )
            with self.assertRaisesRegex(ValueError, "50"):
                list(
                    read_frozen_trajectories(
                        path, allow_smoke_unverified=True, smoke_max_samples=50
                    )
                )

    def test_cpu_fake_extraction_keeps_exact_ids_but_cannot_freeze_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._write_source(root)
            output = root / "cache"
            count = extract_trajectory_cache(
                trajectory_path=source,
                output_dir=output,
                runner=FakeRunner(),
                logical_layer_ids=(1, 3),
                expected_target_identity={
                    "model_fingerprint": "model-sha",
                    "model_revision": "revision-a",
                    "tokenizer_fingerprint": "tokenizer-sha",
                    "vocab_size": 154880,
                },
            )
            self.assertEqual(count, 1)
            dataset = PackedHiddenDataset(output, require_frozen=False)
            self.assertEqual(dataset[0]["input_ids"].tolist(), [1, 2, 3])
            self.assertEqual(dataset[0]["loss_mask"].tolist(), [False, True, True])
            self.assertRegex(dataset.cache_identity, r"^[0-9a-f]{64}$")
            self.assertEqual(dataset.manifest["provenance"]["logical_layer_ids"], [1, 3])
            self.assertEqual(dataset.manifest["provenance"]["physical_layer_ids"], [2, 4])
            self.assertEqual(dataset.manifest["status"], "incomplete")
            self.assertFalse(dataset.manifest["production_eligible"])

    def test_smoke_extraction_never_produces_a_production_frozen_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._write_source(
                root, status="smoke_unverified", production_eligible=False
            )
            output = root / "cache"
            extract_trajectory_cache(
                trajectory_path=source,
                output_dir=output,
                runner=FakeRunner(),
                logical_layer_ids=(1, 3),
                expected_target_identity={
                    "model_fingerprint": "model-sha",
                    "model_revision": "revision-a",
                    "tokenizer_fingerprint": "tokenizer-sha",
                    "vocab_size": 154880,
                },
                allow_smoke_unverified=True,
                smoke_max_samples=50,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "smoke_unverified")
            self.assertFalse(manifest["production_eligible"])
            with self.assertRaisesRegex(ValueError, "smoke_unverified"):
                PackedHiddenDataset(output)
            self.assertEqual(
                len(PackedHiddenDataset(output, allow_smoke_unverified=True)), 1
            )


if __name__ == "__main__":
    unittest.main()
