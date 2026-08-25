from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from glm_dflash2.hidden_cache import PackedHiddenDataset
from glm_dflash2.hidden_extraction import (
    estimate_packed_cache_bytes,
    extract_trajectory_cache,
    read_frozen_trajectories,
)


class FakeRunner:
    hidden_size = 3
    physical_layer_ids = (2, 21, 39, 57, 76)
    backend_metadata = {"backend": "fake", "version": "test"}

    def extract(self, input_ids):
        tokens = len(input_ids)
        return torch.arange(tokens * 5 * 3).reshape(tokens, 5, 3).to(torch.bfloat16)


class HiddenExtractionTest(unittest.TestCase):
    def test_storage_estimate_includes_hidden_ids_and_mask(self):
        self.assertEqual(
            estimate_packed_cache_bytes(10, num_layers=5, hidden_size=6144),
            10 * (5 * 6144 * 2 + 9),
        )

    def _write_trajectory(self, root: Path, *, status: str = "frozen") -> Path:
        path = root / "trajectories.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "x",
                    "stage_a_complete": True,
                    "input_ids": [1, 2, 3],
                    "loss_mask": [0, 1, 0],
                    "source_metadata": {"selected_source_index": 4},
                    "token_contract": {"mask_semantics": "dflash_target_token"},
                }
            )
            + "\n"
        )
        path.with_suffix(path.suffix + ".manifest.json").write_text(
            json.dumps({"status": status, "committed_ids": 1})
        )
        return path

    def test_requires_frozen_trajectory_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_trajectory(Path(tmp), status="running")
            with self.assertRaisesRegex(ValueError, "not frozen"):
                list(read_frozen_trajectories(path))

    def test_partial_trajectory_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_trajectory(Path(tmp), status="partial")
            with self.assertRaisesRegex(ValueError, "not frozen"):
                list(read_frozen_trajectories(path))
            self.assertEqual(
                len(list(read_frozen_trajectories(path, allow_partial=True))), 1
            )

    def test_extracts_exact_input_ids_and_records_layer_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._write_trajectory(root)
            output = root / "cache"
            count = extract_trajectory_cache(
                trajectory_path=source,
                output_dir=output,
                runner=FakeRunner(),
                logical_layer_ids=(1, 20, 38, 56, 75),
            )
            self.assertEqual(count, 1)
            dataset = PackedHiddenDataset(output)
            self.assertEqual(dataset[0]["input_ids"].tolist(), [1, 2, 3])
            provenance = dataset.manifest["provenance"]
            self.assertEqual(provenance["logical_layer_ids"], [1, 20, 38, 56, 75])
            self.assertEqual(provenance["physical_layer_ids"], [2, 21, 39, 57, 76])

    def test_max_samples_leaves_building_cache_that_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._write_trajectory(root)
            first = json.loads(source.read_text())
            second = dict(first)
            second["id"] = "y"
            second["source_metadata"] = {"selected_source_index": 5}
            source.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n"
            )
            output = root / "cache"
            self.assertEqual(
                extract_trajectory_cache(
                    trajectory_path=source,
                    output_dir=output,
                    runner=FakeRunner(),
                    logical_layer_ids=(1, 20, 38, 56, 75),
                    max_samples=1,
                ),
                1,
            )
            with self.assertRaisesRegex(ValueError, "not frozen"):
                PackedHiddenDataset(output)
            self.assertEqual(
                extract_trajectory_cache(
                    trajectory_path=source,
                    output_dir=output,
                    runner=FakeRunner(),
                    logical_layer_ids=(1, 20, 38, 56, 75),
                ),
                1,
            )
            self.assertEqual(len(PackedHiddenDataset(output)), 2)


if __name__ == "__main__":
    unittest.main()
