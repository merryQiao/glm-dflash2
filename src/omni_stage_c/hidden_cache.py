from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from safetensors import safe_open

from omni_sd.provenance import verify_artifact_record

from .contracts import CACHE_CONTRACT, TARGET_CONTRACT


class PackedThinkerHiddenCache:
    """Strict random-access adapter for immutable Stage B v3 shards."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self._validate_manifest()
        self.rows: list[dict[str, Any]] = []
        for record in self.manifest["files"]:
            data_path = verify_artifact_record(record["data"], root=self.root)
            index_path = verify_artifact_record(record["index"], root=self.root)
            for row in pq.read_table(index_path).to_pylist():
                row["data_path"] = str(data_path)
                self.rows.append(row)

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("status") != "PASS" or manifest.get("schema") != CACHE_CONTRACT.schema:
            raise ValueError("Stage C requires a verified omni-thinker-hidden-cache-v3 cache")
        if int(manifest.get("hidden_size", -1)) != TARGET_CONTRACT.hidden_size:
            raise ValueError("cache hidden width differs from official Thinker")
        if tuple(manifest.get("target_layer_ids", ())) != TARGET_CONTRACT.logical_layer_ids:
            raise ValueError("cache layer IDs differ from Stage C contract")
        if manifest.get("dtype") != "bfloat16":
            raise ValueError("Stage C production cache must use bfloat16 hidden states")
        if manifest.get("position_layout") != "tokens,axes" or manifest.get("position_axes") != [
            "temporal", "height", "width"
        ]:
            raise ValueError("cache lacks exact three-axis Thinker mRoPE positions")
        if manifest.get("position_ids_source") != "official_transformers_get_rope_index":
            raise ValueError("cache positions do not come from official Thinker mRoPE")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        start, end = int(row["start"]), int(row["end"])
        result: dict[str, Any] = {"sample_id": str(row["condition_id"])}
        names = {
            "input_ids": "input_ids",
            "loss_mask": "loss_mask",
            "auxiliary_hidden": "target_hidden_states",
            "target_final_hidden": "target_last_hidden_states",
            "position_ids": "position_ids",
        }
        with safe_open(row["data_path"], framework="pt", device="cpu") as handle:
            for output, stored in names.items():
                tensor = handle.get_slice(stored)[start:end]
                result[output] = torch.as_tensor(tensor)
        tokens = end - start
        if result["position_ids"].shape != (tokens, 3):
            raise ValueError("corrupt mRoPE stream")
        if result["auxiliary_hidden"].shape != (tokens, 5, 2048):
            raise ValueError("corrupt auxiliary hidden stream")
        if result["target_final_hidden"].shape != (tokens, 2048):
            raise ValueError("corrupt final hidden stream")
        return result
