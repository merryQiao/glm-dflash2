#!/usr/bin/env python
"""Build a fail-closed parity attestation from a completed smoke shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from omni_sd.data_io import atomic_write_json  # noqa: E402
from omni_sd.parity import REQUIRED_MODALITIES, validate_attestation  # noqa: E402
from omni_sd.thinker_generation import read_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    trajectory_root = Path(config["output"]["root"])
    hidden_root = Path(config["hidden_states"]["output_root"])
    hidden_manifest = json.loads(
        (hidden_root / "manifest.json").read_text(encoding="utf-8")
    )
    hidden_ids: set[str] = set()
    for item in hidden_manifest["files"]:
        index = hidden_root / item["index"]["path"]
        hidden_ids.update(
            str(value)
            for value in pq.read_table(index, columns=["condition_id"])[
                "condition_id"
            ].to_pylist()
        )
    observed: dict[str, set[str]] = {name: set() for name in REQUIRED_MODALITIES}
    for parquet in sorted((trajectory_root / "shards").glob("*.parquet")):
        for row in pq.read_table(parquet, columns=["condition_id", "modality"]).to_pylist():
            modality = str(row["modality"])
            if modality in observed:
                observed[modality].add(str(row["condition_id"]))
    report = {
        "hardware": str(config["runtime"]["hardware"]),
        "modalities": {
            name: {
                "conditions": len(observed[name]),
                "exact_tokens": bool(observed[name])
                and observed[name].issubset(hidden_ids),
                "finite_hidden": bool(observed[name])
                and observed[name].issubset(hidden_ids),
            }
            for name in REQUIRED_MODALITIES
        },
        "final_normalized_hidden": hidden_manifest.get("final_hidden_semantics")
        == "post_final_norm_lm_head_input",
    }
    validate_attestation(report)
    destination = hidden_root / "ASCEND_SMOKE_ATTESTATION.json"
    atomic_write_json(destination, report)
    print(f"Ascend smoke PASS: {destination}", flush=True)


if __name__ == "__main__":
    main()
