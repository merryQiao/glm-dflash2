from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from glm53_drafters.hidden_cache import (
    DFlashHiddenCollator,
    HiddenCacheSpec,
    PackedHiddenDataset,
    PackedHiddenWriter,
    validate_frozen_hidden_cache,
)
from glm53_drafters import hidden_capture
from glm53_drafters.hidden_capture import (
    CaptureAttestation,
    CaptureTap,
    TargetHiddenCapture,
)


class GLM53HiddenCacheTest(unittest.TestCase):
    @staticmethod
    def _spec() -> HiddenCacheSpec:
        return HiddenCacheSpec(
            layer_ids=(1, 3),
            hidden_size=3,
            capture_mapping=(
                ("test", 1, "hidden_states[2]", "post_decoder_block"),
                ("test", 3, "hidden_states[4]", "post_decoder_block"),
            ),
        )

    @staticmethod
    def _provenance(*, production_eligible: bool = False) -> dict[str, object]:
        return {
            "model_fingerprint": "model-sha",
            "model_revision": "revision-a",
            "tokenizer_fingerprint": "tokenizer-sha",
            "trajectory_sha256": "trajectory-sha",
            "production_eligible": production_eligible,
        }

    @staticmethod
    def _attestation(tokens: int = 2, *, passed: bool = True) -> CaptureAttestation:
        return CaptureAttestation(
            passed=passed,
            token_count=tokens,
            logical_layer_ids=(1, 3),
            physical_layer_ids=(2, 4),
            independent_tap_paths=("layers[1]", "layers[3]"),
            native_logits_path="runner.compute_logits/global",
            aux_max_abs_error=0.0 if passed else 1.0,
            aux_max_rel_error=0.0 if passed else 1.0,
            logits_max_abs_error=0.0,
            logits_max_rel_error=0.0,
            reason="" if passed else "aux parity mismatch",
        )

    @staticmethod
    def _append(writer: PackedHiddenWriter, sample_id: str = "sample-a") -> None:
        writer.append(
            sample_id=sample_id,
            source_index=7,
            input_ids=torch.tensor([10, 11]),
            loss_mask=torch.tensor([0, 1], dtype=torch.bool),
            aux_hidden_states=torch.arange(12, dtype=torch.float32)
            .reshape(2, 2, 3)
            .to(torch.bfloat16),
            target_final_hidden=torch.ones(2, 3, dtype=torch.bfloat16),
            attestation=GLM53HiddenCacheTest._attestation(),
        )

    def test_capture_keeps_ordered_aux_and_post_final_norm_streams_together(self):
        taps = (
            CaptureTap("test", 1, "hidden_states[2]", "post_decoder_block"),
            CaptureTap("test", 3, "hidden_states[4]", "post_decoder_block"),
        )
        capture = TargetHiddenCapture(
            aux_hidden_states=torch.randn(4, 2, 3),
            target_final_hidden=torch.randn(4, 3),
            capture_mapping=taps,
        ).cpu_bfloat16()
        self.assertEqual(capture.logical_layer_ids, (1, 3))
        self.assertEqual(capture.aux_hidden_states.dtype, torch.bfloat16)
        self.assertEqual(capture.target_final_hidden.dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "post-final-norm"):
            TargetHiddenCapture(
                aux_hidden_states=torch.zeros(4, 2, 3),
                target_final_hidden=torch.zeros(4, 3),
                capture_mapping=taps,
                final_hidden_semantics="pre_final_norm",
            )

    def test_smoke_cache_roundtrip_seals_nonproduction_and_exposes_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with PackedHiddenWriter(
                root, spec=self._spec(), provenance=self._provenance()
            ) as writer:
                self._append(writer)
                identity = writer.cache_identity
                writer.freeze()
            dataset = PackedHiddenDataset(root, allow_smoke_unverified=True)
            self.assertEqual(dataset.cache_identity, identity)
            self.assertEqual(dataset.manifest["status"], "smoke_unverified")
            self.assertFalse(dataset.manifest["production_eligible"])
            row = dataset[0]
            self.assertEqual(row["input_ids"].tolist(), [10, 11])
            self.assertEqual(row["loss_mask"].tolist(), [False, True])
            self.assertEqual(tuple(row["layer_hidden_states"].shape), (2, 2, 3))
            self.assertEqual(tuple(row["hidden_states"].shape), (2, 6))
            self.assertEqual(tuple(row["target_final_hidden"].shape), (2, 3))

    def test_resume_truncates_uncommitted_tail_and_rejects_identity_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self._spec()
            provenance = self._provenance()
            with PackedHiddenWriter(root, spec=spec, provenance=provenance) as writer:
                self._append(writer)
            stream = root / "segment-00000" / "aux_hidden_states.bin"
            committed_size = stream.stat().st_size
            with stream.open("ab") as handle:
                handle.write(b"uncommitted")
            with PackedHiddenWriter(root, spec=spec, provenance=provenance) as writer:
                self.assertEqual(stream.stat().st_size, committed_size)
                self._append(writer, "sample-b")
                writer.freeze()
            self.assertEqual(
                len(PackedHiddenDataset(root, allow_smoke_unverified=True)), 2
            )
            changed = dict(provenance, model_fingerprint="other")
            with self.assertRaisesRegex(ValueError, "cache identity"):
                with PackedHiddenWriter(root, spec=spec, provenance=changed):
                    pass

    def test_checksums_detect_committed_stream_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with PackedHiddenWriter(
                root, spec=self._spec(), provenance=self._provenance()
            ) as writer:
                self._append(writer)
                writer.freeze()
            path = root / "segment-00000" / "loss_mask.bin"
            raw = bytearray(path.read_bytes())
            raw[0] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "checksum.*sample-a"):
                validate_frozen_hidden_cache(root, allow_smoke_unverified=True)

    def test_production_freeze_is_incomplete_without_every_row_numeric_attestation(self):
        for evidence in (None, self._attestation(passed=False)):
            with self.subTest(evidence=evidence), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with PackedHiddenWriter(
                    root,
                    spec=self._spec(),
                    provenance=self._provenance(production_eligible=True),
                ) as writer:
                    writer.append(
                        sample_id="sample-a",
                        source_index=0,
                        input_ids=[1, 2],
                        loss_mask=[0, 1],
                        aux_hidden_states=torch.zeros(2, 2, 3),
                        target_final_hidden=torch.zeros(2, 3),
                        attestation=evidence,
                    )
                    writer.freeze()
                manifest = json.loads((root / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "incomplete")
                self.assertFalse(manifest["production_eligible"])
                self.assertEqual(manifest["attestation"]["required_rows"], 1)
                with self.assertRaisesRegex(ValueError, "not frozen"):
                    PackedHiddenDataset(root)

    def test_cpu_or_caller_declared_a2_evidence_cannot_mint_production_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provenance = self._provenance(production_eligible=True)
            provenance["ascend_a2_runtime"] = {
                "schema": "glm53-ascend-910b-a2-runtime-v1",
                "passed": True,
                "device_name": "Ascend910B2",
            }
            with PackedHiddenWriter(
                root, spec=self._spec(), provenance=provenance
            ) as writer:
                self._append(writer)
                writer.freeze()
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "incomplete")
            self.assertFalse(manifest["production_eligible"])

    def test_no_caller_importable_a2_capability_factory_exists(self):
        self.assertFalse(
            hasattr(hidden_capture, "_issue_ascend_a2_runtime_attestation"),
            "a caller-importable factory can mint a trusted A2 capability",
        )

    def test_cpu_cannot_freeze_with_a_caller_minted_runtime_capability(self):
        factory = getattr(
            hidden_capture, "_issue_ascend_a2_runtime_attestation", None
        )
        self.assertIsNone(
            factory, "production capability issuance must perform the live probe itself"
        )

    def test_passing_attestation_rejects_errors_above_declared_tolerance(self):
        with self.assertRaisesRegex(ValueError, "tolerance"):
            CaptureAttestation(
                passed=True,
                token_count=2,
                logical_layer_ids=(1, 3),
                physical_layer_ids=(2, 4),
                independent_tap_paths=("layers[1]", "layers[3]"),
                native_logits_path="runner.compute_logits",
                aux_max_abs_error=999.0,
                aux_max_rel_error=999.0,
                logits_max_abs_error=999.0,
                logits_max_rel_error=999.0,
                reason="",
            )

    def test_sealed_manifest_provenance_must_recompute_to_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with PackedHiddenWriter(
                root, spec=self._spec(), provenance=self._provenance()
            ) as writer:
                self._append(writer)
                writer.freeze()
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["provenance"]["model_fingerprint"] = "mutated"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "cache identity"):
                PackedHiddenDataset(root, allow_smoke_unverified=True)

    def test_collator_right_pads_all_schema_v2_streams(self):
        collate = DFlashHiddenCollator(pad_token_id=99)
        batch = collate(
            [
                {
                    "sample_id": "a",
                    "input_ids": torch.tensor([1, 2]),
                    "loss_mask": torch.tensor([True, False]),
                    "hidden_states": torch.ones(2, 6, dtype=torch.bfloat16),
                    "target_final_hidden": torch.ones(2, 3, dtype=torch.bfloat16),
                },
                {
                    "sample_id": "b",
                    "input_ids": torch.tensor([3]),
                    "loss_mask": torch.tensor([True]),
                    "hidden_states": torch.ones(1, 6, dtype=torch.bfloat16),
                    "target_final_hidden": torch.ones(1, 3, dtype=torch.bfloat16),
                },
            ]
        )
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [3, 99]])
        self.assertEqual(batch["attention_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(tuple(batch["hidden_states"].shape), (2, 2, 6))
        self.assertEqual(tuple(batch["target_final_hidden"].shape), (2, 2, 3))


if __name__ == "__main__":
    unittest.main()
