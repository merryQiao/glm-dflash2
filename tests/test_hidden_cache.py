from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from glm_dflash2.hidden_cache import (
    DFlashHiddenCollator,
    HiddenCacheSpec,
    PackedHiddenDataset,
    PackedHiddenWriter,
    rebuild_manifest_from_index,
)


class HiddenCacheTest(unittest.TestCase):
    @staticmethod
    def _spec(layer_ids=(1,), hidden_size=2, *, schema_version=2):
        mapping = tuple(
            ("test", layer, f"hidden_states[{layer + 1}]", "post_decoder_block")
            for layer in layer_ids
        )
        return HiddenCacheSpec(
            layer_ids=tuple(layer_ids),
            hidden_size=hidden_size,
            schema_version=schema_version,
            capture_mapping=mapping if schema_version == 2 else (),
        )

    @staticmethod
    def _append(writer, sample_id="s1", tokens=2):
        values = dict(
            sample_id=sample_id,
            source_index=0,
            input_ids=torch.arange(tokens),
            loss_mask=torch.tensor([1] + [0] * (tokens - 1), dtype=torch.bool),
            hidden_states=torch.zeros(tokens, len(writer.spec.layer_ids), writer.spec.hidden_size, dtype=torch.bfloat16),
        )
        if writer.spec.schema_version == 2:
            values["target_final_hidden"] = torch.ones(
                tokens, writer.spec.final_hidden_size, dtype=torch.bfloat16
            )
        writer.append(**values)

    def test_roundtrip_preserves_bf16_layers_and_dflash_flattening(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self._spec((1, 20, 38, 56, 75), 4)
            hidden = torch.arange(3 * 5 * 4, dtype=torch.float32).reshape(3, 5, 4).to(torch.bfloat16)
            with PackedHiddenWriter(root, spec=spec, max_segment_bytes=1 << 20) as writer:
                writer.append(
                    sample_id="s1",
                    source_index=7,
                    input_ids=torch.tensor([11, 12, 13]),
                    loss_mask=torch.tensor([0, 1, 0], dtype=torch.bool),
                    hidden_states=hidden,
                    target_final_hidden=torch.full((3, 4), 2, dtype=torch.bfloat16),
                )
                writer.freeze()

            dataset = PackedHiddenDataset(root)
            row = dataset[0]
            self.assertEqual(row["sample_id"], "s1")
            self.assertEqual(row["input_ids"].tolist(), [11, 12, 13])
            self.assertEqual(row["loss_mask"].tolist(), [False, True, False])
            self.assertEqual(tuple(row["layer_hidden_states"].shape), (3, 5, 4))
            self.assertTrue(torch.equal(row["layer_hidden_states"], hidden))
            self.assertEqual(tuple(row["hidden_states"].shape), (3, 20))
            self.assertTrue(torch.equal(row["hidden_states"], hidden.flatten(1)))
            self.assertEqual(tuple(row["target_final_hidden"].shape), (3, 4))
            self.assertTrue((row["target_final_hidden"] == 2).all())
            self.assertTrue(dataset.aligned_methods_allowed)

    def test_collator_right_pads_and_keeps_target_position_mask(self):
        collate = DFlashHiddenCollator(pad_token_id=99)
        batch = collate(
            [
                {
                    "sample_id": "a",
                    "input_ids": torch.tensor([1, 2]),
                    "loss_mask": torch.tensor([True, False]),
                    "hidden_states": torch.ones(2, 6, dtype=torch.bfloat16),
                    "target_final_hidden": torch.ones(2, 2, dtype=torch.bfloat16),
                },
                {
                    "sample_id": "b",
                    "input_ids": torch.tensor([3]),
                    "loss_mask": torch.tensor([True]),
                    "hidden_states": torch.full((1, 6), 2, dtype=torch.bfloat16),
                    "target_final_hidden": torch.full((1, 2), 2, dtype=torch.bfloat16),
                },
            ]
        )
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [3, 99]])
        self.assertEqual(batch["attention_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch["loss_mask"].tolist(), [[True, False], [True, False]])
        self.assertEqual(tuple(batch["hidden_states"].shape), (2, 2, 6))
        self.assertEqual(tuple(batch["target_final_hidden"].shape), (2, 2, 2))

    def test_rebuild_manifest_recovers_committed_index_ahead_of_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self._spec((1, 2), 2)
            with PackedHiddenWriter(root, spec=spec) as writer:
                writer.append(
                    sample_id="s1",
                    source_index=0,
                    input_ids=torch.tensor([1, 2]),
                    loss_mask=torch.tensor([1, 0], dtype=torch.bool),
                    hidden_states=torch.zeros(2, 2, 2, dtype=torch.bfloat16),
                    target_final_hidden=torch.zeros(2, 2, dtype=torch.bfloat16),
                )
            manifest_path = root / "manifest.json"
            stale = json.loads(manifest_path.read_text())
            stale["samples"] = 0
            manifest_path.write_text(json.dumps(stale))
            rebuilt = rebuild_manifest_from_index(root)
            self.assertEqual(rebuilt["samples"], 1)
            self.assertEqual(rebuilt["total_tokens"], 2)

    def test_dataset_refuses_unfrozen_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with PackedHiddenWriter(root, spec=self._spec()):
                pass
            with self.assertRaisesRegex(ValueError, "not frozen"):
                PackedHiddenDataset(root)

    def test_duplicate_sample_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with PackedHiddenWriter(
                Path(tmp), spec=self._spec()
            ) as writer:
                self._append(writer)
                with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                    self._append(writer)

    def test_non_finite_hidden_is_rejected_before_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with PackedHiddenWriter(
                root, spec=self._spec()
            ) as writer:
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    writer.append(
                        sample_id="bad",
                        source_index=0,
                        input_ids=[1],
                        loss_mask=[0],
                        hidden_states=torch.tensor(
                            [[[float("nan"), 0.0]]], dtype=torch.bfloat16
                        ),
                        target_final_hidden=torch.zeros(1, 2, dtype=torch.bfloat16),
                    )
            self.assertEqual((root / "index.jsonl").read_text(), "")

    def test_rollover_uses_multiple_segments_without_splitting_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with PackedHiddenWriter(
                root,
                spec=self._spec(),
                max_segment_bytes=30,
            ) as writer:
                self._append(writer, "a", 2)
                self._append(writer, "b", 2)
                writer.freeze()
            rows = [json.loads(line) for line in (root / "index.jsonl").read_text().splitlines()]
            self.assertEqual([row["segment"] for row in rows], [0, 1])
            self.assertEqual(len(PackedHiddenDataset(root)), 2)

    def test_read_detects_committed_stream_checksum_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with PackedHiddenWriter(
                root, spec=self._spec()
            ) as writer:
                self._append(writer)
                writer.freeze()
            path = root / "segment-00000" / "loss_mask.bin"
            raw = bytearray(path.read_bytes())
            raw[0] ^= 1
            path.write_bytes(raw)
            dataset = PackedHiddenDataset(root, verify_checksums=True)
            with self.assertRaisesRegex(ValueError, "checksum"):
                _ = dataset[0]

    def test_opening_frozen_dataset_is_read_only_and_checksum_is_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with PackedHiddenWriter(
                root, spec=self._spec()
            ) as writer:
                self._append(writer)
                writer.freeze()
            with mock.patch("glm_dflash2.hidden_cache._atomic_json") as atomic:
                dataset = PackedHiddenDataset(root)
                _ = dataset[0]
            atomic.assert_not_called()

    def test_resume_truncates_only_unindexed_stream_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self._spec()
            with PackedHiddenWriter(root, spec=spec) as writer:
                self._append(writer)
            path = root / "segment-00000" / "aux_hidden_states.bin"
            committed_size = path.stat().st_size
            with path.open("ab") as handle:
                handle.write(b"uncommitted")
            with PackedHiddenWriter(root, spec=spec) as writer:
                self.assertEqual(path.stat().st_size, committed_size)
                self._append(writer, "s2")
                writer.freeze()
            self.assertEqual(len(PackedHiddenDataset(root)), 2)

    def test_schema_v1_requires_explicit_legacy_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self._spec(schema_version=1)
            with PackedHiddenWriter(root, spec=spec) as writer:
                self._append(writer)
                writer.freeze()
            with self.assertRaisesRegex(ValueError, "legacy schema v1"):
                PackedHiddenDataset(root)
            dataset = PackedHiddenDataset(root, allow_legacy_v1=True)
            self.assertFalse(dataset.aligned_methods_allowed)
            self.assertNotIn("target_final_hidden", dataset[0])


if __name__ == "__main__":
    unittest.main()
