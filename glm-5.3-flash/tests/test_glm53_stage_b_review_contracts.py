from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from glm53_drafters.hidden_cache import HiddenCacheSpec, PackedHiddenWriter
from glm53_drafters.hidden_extraction import read_frozen_trajectories


TARGET_IDENTITY = {
    "model_fingerprint": "model-sha",
    "model_revision": "revision-a",
    "tokenizer_fingerprint": "tokenizer-sha",
    "vocab_size": 154880,
}


class StrictStageBInputContractTest(unittest.TestCase):
    @staticmethod
    def _source(
        root: Path,
        *,
        input_ids: list[object] | None = None,
        loss_mask: list[object] | None = None,
        status: str = "frozen",
        production_eligible: bool = True,
        manifest_updates: dict[str, object] | None = None,
    ) -> Path:
        path = root / "trajectory.jsonl"
        row = {
            "id": "sample-a",
            "stage_a_complete": True,
            "generation_route": "workspace_task",
            "input_ids": [1, 2] if input_ids is None else input_ids,
            "loss_mask": [0, 1] if loss_mask is None else loss_mask,
            "token_contract": {"mask_semantics": "dflash_target_token"},
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        raw = path.read_bytes()
        manifest: dict[str, object] = {
            "schema_version": 3,
            "status": status,
            "production_eligible": production_eligible,
            "committed_ids": 1,
            "jsonl_bytes": len(raw),
            "jsonl_sha256": hashlib.sha256(raw).hexdigest(),
            "unresolved_errors": 0,
            **TARGET_IDENTITY,
        }
        manifest.update(manifest_updates or {})
        path.with_suffix(path.suffix + ".manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return path

    def test_production_manifest_requires_every_exact_identity_field_and_true_eligibility(self):
        for field in (*TARGET_IDENTITY, "production_eligible"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = self._source(root)
                manifest_path = path.with_suffix(path.suffix + ".manifest.json")
                manifest = json.loads(manifest_path.read_text())
                manifest.pop(field)
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, field):
                    list(
                        read_frozen_trajectories(
                            path, expected_target_identity=TARGET_IDENTITY
                        )
                    )

        for field, wrong in (
            ("model_fingerprint", "wrong"),
            ("model_revision", "wrong"),
            ("tokenizer_fingerprint", "wrong"),
            ("vocab_size", 154879),
            ("production_eligible", False),
        ):
            with self.subTest(field=field, wrong=wrong), tempfile.TemporaryDirectory() as tmp:
                path = self._source(Path(tmp), manifest_updates={field: wrong})
                with self.assertRaisesRegex(ValueError, field.replace("_", " ")):
                    list(
                        read_frozen_trajectories(
                            path, expected_target_identity=TARGET_IDENTITY
                        )
                    )

    def test_smoke_cannot_launder_a_missing_or_different_target_identity(self):
        for field, wrong in (
            ("model_fingerprint", "wrong"),
            ("model_revision", "wrong"),
            ("tokenizer_fingerprint", "wrong"),
            ("vocab_size", 154879),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                path = self._source(
                    Path(tmp),
                    status="smoke_unverified",
                    production_eligible=False,
                    manifest_updates={field: wrong},
                )
                with self.assertRaisesRegex(ValueError, field.replace("_", " ")):
                    list(
                        read_frozen_trajectories(
                            path,
                            allow_smoke_unverified=True,
                            smoke_max_samples=50,
                            expected_target_identity=TARGET_IDENTITY,
                        )
                    )

    def test_reader_rejects_non_integer_bool_negative_and_oob_token_ids(self):
        invalid_ids = ([True], [1.0], ["1"], [-1], [154880])
        for input_ids in invalid_ids:
            with self.subTest(input_ids=input_ids), tempfile.TemporaryDirectory() as tmp:
                path = self._source(
                    Path(tmp), input_ids=list(input_ids), loss_mask=[1]
                )
                with self.assertRaisesRegex(ValueError, "token ID"):
                    list(
                        read_frozen_trajectories(
                            path, expected_target_identity=TARGET_IDENTITY
                        )
                    )

    def test_reader_rejects_coercible_or_non_binary_loss_masks(self):
        invalid_masks = ([0.0, 1.0], ["0", "1"], [-1, 1], [0, 2])
        for loss_mask in invalid_masks:
            with self.subTest(loss_mask=loss_mask), tempfile.TemporaryDirectory() as tmp:
                path = self._source(Path(tmp), loss_mask=list(loss_mask))
                with self.assertRaisesRegex(ValueError, "loss mask"):
                    list(
                        read_frozen_trajectories(
                            path, expected_target_identity=TARGET_IDENTITY
                        )
                    )

    @staticmethod
    def _spec() -> HiddenCacheSpec:
        return HiddenCacheSpec(
            layer_ids=(1,),
            hidden_size=2,
            capture_mapping=(("test", 1, "layer[1]", "post_decoder_block"),),
        )

    def test_writer_does_not_coerce_invalid_ids_or_masks(self):
        cases = (
            ([True], [1], "token ID"),
            ([1.0], [1], "token ID"),
            ([-1], [1], "token ID"),
            ([154880], [1], "token ID"),
            ([1], [1.0], "loss mask"),
            ([1], [2], "loss mask"),
        )
        for ids, mask, message in cases:
            with self.subTest(ids=ids, mask=mask), tempfile.TemporaryDirectory() as tmp:
                with PackedHiddenWriter(
                    tmp,
                    spec=self._spec(),
                    provenance={"production_eligible": False},
                ) as writer:
                    with self.assertRaisesRegex(ValueError, message):
                        writer.append(
                            sample_id="sample",
                            source_index=0,
                            input_ids=ids,
                            loss_mask=mask,
                            aux_hidden_states=torch.zeros(1, 1, 2),
                            target_final_hidden=torch.zeros(1, 2),
                        )


if __name__ == "__main__":
    unittest.main()
