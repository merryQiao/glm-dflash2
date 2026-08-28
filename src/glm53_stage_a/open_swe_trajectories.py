"""Disk-backed access to complete Open-SWE trajectories."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .agent_trajectory import validate_messages
from .vibe_coding import ModelInput, row_to_model_input


class OpenSWETrajectoryStore:
    """Read complete source trajectories without loading them all into RAM."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=True)
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def get(self, item: ModelInput) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT source_id, instance_id, repo, trajectory_json, tools_json,
                   source_config, source_split
            FROM trajectories
            WHERE id = ?
            """,
            (item.id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"complete Open-SWE trajectory is missing: {item.id}")
        raw_messages = json.loads(row["trajectory_json"])
        raw_tools = json.loads(row["tools_json"])
        normalized = row_to_model_input(
            {
                "id": item.id,
                "prompt": "Restore the complete source trajectory.",
                "input_kind": "agent_trajectory_prefix",
                "context_json": json.dumps(
                    {"messages": raw_messages, "tools": raw_tools},
                    ensure_ascii=False,
                ),
            }
        )
        messages = normalized.messages
        validation = validate_messages(messages, require_tool=True)
        return {
            "id": item.id,
            "messages": messages,
            "generation_start_message_index": 0,
            "tools": normalized.tools,
            "tool_events": [],
            "terminal_reason": "original_open_swe_trajectory",
            "validation": validation,
            "response_metadata": [],
            "source_metadata": {
                **item.source_metadata,
                "input_kind": item.input_kind,
                "repo": row["repo"] or item.repo,
                "base_commit": item.base_commit,
                "trajectory_origin": "nvidia/Open-SWE-Traces",
                "source_id": row["source_id"],
                "instance_id": row["instance_id"],
                "source_config": row["source_config"],
                "source_split": row["source_split"],
                "workspace_mode": "original-trajectory",
            },
        }
