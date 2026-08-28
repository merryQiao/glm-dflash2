from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from glm53_stage_a.open_swe_trajectories import OpenSWETrajectoryStore
from glm53_stage_a.vibe_coding import ModelInput, row_to_model_input
from glm53_stage_a.workspaces import AutomaticWorkspaceProvider


class WorkspaceParityTest(unittest.TestCase):
    def test_seed_file_gets_an_isolated_temporary_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = AutomaticWorkspaceProvider(Path(directory) / "cache")
            item = ModelInput(
                id="file-task:1",
                route="workspace_task",
                input_kind="file_before_change",
                messages=[],
                workspace_required=True,
                workspace_seed_files={"src/example.py": "print('before')\n"},
            )
            with provider.acquire(item) as lease:
                self.assertIsNotNone(lease)
                self.assertEqual(lease.mode, "temporary-file")
                workspace = lease.executor.root
                self.assertEqual(
                    (workspace / "src/example.py").read_text(), "print('before')\n"
                )
            self.assertFalse(workspace.exists())

    def test_executable_row_retains_container_contract(self) -> None:
        item = row_to_model_input(
            {
                "id": "exec:1",
                "prompt": "Fix it.",
                "input_kind": "executable_repo_reference",
                "context_json": json.dumps(
                    {
                        "image_name": "registry.example/task:1",
                        "install_config": {"test_cmd": "pytest -q"},
                    }
                ),
                "repo": "owner/project",
                "base_commit": "a" * 40,
            }
        )
        self.assertEqual(item.workspace_image, "registry.example/task:1")
        self.assertEqual(item.workspace_test_command, "pytest -q")

    def test_open_swe_store_restores_complete_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "open-swe.sqlite"
            connection = sqlite3.connect(path)
            connection.execute(
                """CREATE TABLE trajectories (
                    id TEXT PRIMARY KEY, source_id TEXT NOT NULL UNIQUE,
                    instance_id TEXT NOT NULL, repo TEXT NOT NULL,
                    trajectory_json TEXT NOT NULL, tools_json TEXT NOT NULL,
                    source_config TEXT NOT NULL, source_split TEXT NOT NULL
                )"""
            )
            messages = [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Inspect a.py."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "a.py"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "read_file",
                    "tool_call_id": "call-1",
                    "content": "1: value = 1",
                },
                {"role": "assistant", "content": "The value is 1."},
            ]
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {"type": "object"},
                    },
                }
            ]
            connection.execute(
                "INSERT INTO trajectories VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "open-swe:1",
                    "source:1",
                    "instance-1",
                    "owner/project",
                    json.dumps(messages),
                    json.dumps(tools),
                    "sweagent",
                    "train",
                ),
            )
            connection.commit()
            connection.close()
            item = ModelInput(
                id="open-swe:1",
                route="trajectory_prefix",
                input_kind="agent_trajectory_prefix",
                messages=[],
                workspace_required=True,
            )
            store = OpenSWETrajectoryStore(path)
            try:
                trajectory = store.get(item)
            finally:
                store.close()
            self.assertTrue(trajectory["validation"]["valid"])
            self.assertEqual(trajectory["messages"], messages)


if __name__ == "__main__":
    unittest.main()
