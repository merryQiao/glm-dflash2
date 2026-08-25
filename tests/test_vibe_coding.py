from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from glm_dflash2.vibe_coding import load_vibe_coding_table, row_to_model_input


def base_row(**updates):
    row = {
        "id": "row-1",
        "source_id": "source-row-1",
        "category": "greenfield_codegen",
        "subcategory": "example",
        "source": "public/example",
        "prompt": "Implement the requested change.",
        "context_json": "{}",
        "repo": "",
        "base_commit": "",
        "language": "Python",
        "license": "MIT",
        "input_kind": "instruction_only",
    }
    row.update(updates)
    return row


class VibeCodingTest(unittest.TestCase):
    def test_file_before_change_becomes_workspace_seed(self):
        item = row_to_model_input(
            base_row(
                input_kind="file_before_change",
                context_json=json.dumps({"path": "src/a.py", "old_contents": "x = 1\n"}),
            )
        )
        self.assertEqual(item.route, "workspace_task")
        self.assertEqual(item.workspace_seed_files, {"src/a.py": "x = 1\n"})

    def test_loader_reads_nested_parquet_and_ignores_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            part = root / "processed" / "kind" / "subset"
            part.mkdir(parents=True)
            pq.write_table(pa.table({"id": ["a", "b"]}), part / "part-00000.parquet")
            (part / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(load_vibe_coding_table(root).num_rows, 2)

    def test_structured_trajectory_prefix_preserves_messages_and_tools(self):
        messages = [
            {"role": "user", "content": "inspect"},
            {"role": "assistant", "content": "working"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "parameters": {"type": "object"},
                },
            }
        ]
        item = row_to_model_input(
            base_row(
                input_kind="agent_trajectory_prefix",
                context_json=json.dumps({"messages": messages, "tools": tools}),
            )
        )
        self.assertEqual(item.route, "trajectory_prefix")
        self.assertEqual(item.messages, messages)
        self.assertEqual(item.tools, tools)

    def test_invalid_trajectory_prefix_has_explicit_warning(self):
        item = row_to_model_input(
            base_row(
                input_kind="terminal_trajectory_prefix",
                context_json=json.dumps({"messages": "not-a-list"}),
            )
        )
        self.assertEqual(item.route, "workspace_task")
        self.assertTrue(item.warnings)

    def test_conversation_followup_preserves_turn_order(self):
        item = row_to_model_input(
            base_row(
                input_kind="developer_conversation_user_followup",
                prompt="third",
                context_json=json.dumps(
                    {"prior_user_prompts": [{"prompt": "first"}, {"prompt": "second"}]}
                ),
            )
        )
        self.assertEqual(item.route, "conversation_seed")
        self.assertEqual(item.messages[-1]["content"], "first")
        self.assertEqual(item.remaining_user_turns, ["second", "third"])

    def test_pull_request_history_is_injected_as_user_context(self):
        item = row_to_model_input(
            base_row(
                input_kind="pull_request_inline_review",
                context_json=json.dumps(
                    {"review_prefix": [{"reviewer": "alice", "path": "a.py", "body": "fix it"}]}
                ),
            )
        )
        self.assertIn("alice on a.py", item.messages[-1]["content"])
        self.assertIn("fix it", item.messages[-1]["content"])

    def test_terminal_context_preserves_environment(self):
        item = row_to_model_input(
            base_row(
                input_kind="terminal_instruction",
                context_json=json.dumps({"platform": "linux", "shell": "bash"}),
            )
        )
        self.assertIn("linux, bash", item.messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
