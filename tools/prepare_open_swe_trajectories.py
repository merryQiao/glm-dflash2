#!/usr/bin/env python3
"""Restore complete Open-SWE traces referenced by vibe_coding_630k."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glm_dflash2.vibe_coding import discover_parquet_files  # noqa: E402


# Usage:
#
# python scripts/prepare_open_swe_trajectories.py \
#   --dataset data/vibe_coding_630k \
#   --output outputs/open_swe_original.sqlite

OPEN_SWE_KINDS = frozenset(
    {"agent_trajectory_prefix", "terminal_trajectory_prefix"}
)
OPEN_SWE_SOURCES = (
    ("minisweagent", "qwen36_27b"),
    ("sweagent", "qwen36_27b"),
    ("openhands", "qwen36_27b"),
    ("sweagent", "qwen35_122b"),
    ("openhands", "qwen35_122b"),
    ("sweagent", "minimax_m25"),
    ("openhands", "minimax_m25"),
    ("openhands", "deepseek_v4_flash"),
)
UNKNOWN_SOURCE = ("*", "*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/vibe_coding_630k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/open_swe_original.sqlite",
    )
    return parser.parse_args()


def normalized_targets(
    dataset: Path,
) -> dict[tuple[str, str], dict[str, tuple[str, str]]]:
    """Group trajectory_id -> (normalized id, source_id) by HF config/split."""

    grouped: dict[tuple[str, str], dict[str, tuple[str, str]]] = defaultdict(dict)
    columns = ["id", "source_id", "input_kind", "metadata_json"]
    for path in discover_parquet_files(dataset):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=8192):
            rows = batch.to_pydict()
            for row_id, source_id, input_kind, metadata_json in zip(
                rows["id"],
                rows["source_id"],
                rows["input_kind"],
                rows["metadata_json"],
            ):
                if input_kind not in OPEN_SWE_KINDS:
                    continue
                parts = str(source_id).split(":", 2)
                if len(parts) == 3 and all(parts):
                    config, split, trajectory_id = parts
                else:
                    metadata = json.loads(metadata_json or "{}")
                    config = str(metadata.get("agent_config") or "")
                    split = str(metadata.get("model_split") or "")
                    trajectory_id = str(source_id)
                if not trajectory_id:
                    raise ValueError(
                        f"invalid Open-SWE source_id for {row_id}: {source_id!r}"
                    )
                if not config or not split:
                    config, split = UNKNOWN_SOURCE
                grouped[(config, split)][trajectory_id] = (row_id, source_id)
    return dict(grouped)


def open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS trajectories (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL UNIQUE,
            instance_id TEXT NOT NULL,
            repo TEXT NOT NULL,
            trajectory_json TEXT NOT NULL,
            tools_json TEXT NOT NULL,
            source_config TEXT NOT NULL,
            source_split TEXT NOT NULL
        )
        """
    )
    return connection


def existing_ids(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT id FROM trajectories")}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def restore(
    connection: sqlite3.Connection,
    targets: dict[tuple[str, str], dict[str, tuple[str, str]]],
) -> tuple[int, int]:
    done = existing_ids(connection)
    restored = 0
    total = sum(len(rows) for rows in targets.values())
    unknown = {
        trajectory_id: identifiers
        for trajectory_id, identifiers in targets.get(UNKNOWN_SOURCE, {}).items()
        if identifiers[0] not in done
    }
    sources = list(
        dict.fromkeys(
            [key for key in targets if key != UNKNOWN_SOURCE]
            + list(OPEN_SWE_SOURCES)
        )
    )
    for config, split in sources:
        rows = targets.get((config, split), {})
        pending = {
            trajectory_id: identifiers
            for trajectory_id, identifiers in rows.items()
            if identifiers[0] not in done
        }
        if not pending and not unknown:
            continue
        print(
            compact_json(
                {
                    "config": config,
                    "split": split,
                    "pending": len(pending),
                    "unknown_source_pending": len(unknown),
                }
            ),
            flush=True,
        )
        source = load_dataset(
            "nvidia/Open-SWE-Traces",
            config,
            split=split,
            streaming=True,
        )
        for raw in source:
            trajectory_id = str(raw.get("trajectory_id") or "")
            identifiers = pending.pop(trajectory_id, None)
            if identifiers is None:
                identifiers = unknown.pop(trajectory_id, None)
            if identifiers is None:
                continue
            row_id, source_id = identifiers
            connection.execute(
                """
                INSERT OR REPLACE INTO trajectories (
                    id, source_id, instance_id, repo, trajectory_json,
                    tools_json, source_config, source_split
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    source_id,
                    str(raw.get("instance_id") or ""),
                    str(raw.get("repo") or ""),
                    compact_json(raw.get("trajectory") or []),
                    compact_json(raw.get("tools") or []),
                    config,
                    split,
                ),
            )
            restored += 1
            done.add(row_id)
            if restored % 100 == 0:
                connection.commit()
                print(
                    compact_json(
                        {
                            "restored_this_run": restored,
                            "remaining_in_source": len(pending),
                            "unknown_source_pending": len(unknown),
                        }
                    ),
                    flush=True,
                )
            if not pending and not unknown:
                break
        connection.commit()
        if pending:
            examples = sorted(pending)[:10]
            raise RuntimeError(
                f"{len(pending)} trajectories missing from {config}/{split}: {examples}"
            )
    if unknown:
        examples = sorted(unknown)[:10]
        raise RuntimeError(
            f"{len(unknown)} trajectories with legacy source metadata were not found: "
            f"{examples}"
        )
    return restored, total


def main() -> int:
    args = parse_args()
    targets = normalized_targets(args.dataset)
    connection = open_database(args.output)
    restored, total = restore(connection, targets)
    stored = connection.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
    connection.close()
    print(
        compact_json(
            {
                "output": str(args.output.resolve()),
                "target_rows": total,
                "restored_this_run": restored,
                "stored_rows": stored,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
