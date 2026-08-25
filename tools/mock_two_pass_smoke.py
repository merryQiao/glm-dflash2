#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm_dflash2.hidden_extraction import extract_trajectory_cache
from glm_dflash2.sglang_stage_a import CommittedJsonlWriter


class MockRunner:
    hidden_size = 4
    physical_layer_ids = (2, 21, 39, 57, 76)
    backend_metadata = {"backend": "mock", "version": "1"}

    def extract(self, input_ids):
        values = torch.arange(len(input_ids) * 5 * self.hidden_size)
        return values.reshape(len(input_ids), 5, self.hidden_size).to(torch.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = args.output_dir / "trajectories.jsonl"
    row = {
        "id": "smoke-0",
        "stage_a_complete": True,
        "input_ids": [11, 12, 13, 14],
        "loss_mask": [0, 1, 1, 0],
        "source_metadata": {"selected_source_index": 0},
        "token_contract": {"mask_semantics": "dflash_target_token"},
    }
    with CommittedJsonlWriter(trajectory, truncate=True) as writer:
        writer.append(row)
    trajectory.with_suffix(trajectory.suffix + ".manifest.json").write_text(
        json.dumps({"status": "frozen", "committed_ids": 1}) + "\n"
    )
    extract_trajectory_cache(
        trajectory_path=trajectory,
        output_dir=args.output_dir / "hidden",
        runner=MockRunner(),
        logical_layer_ids=(1, 20, 38, 56, 75),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
