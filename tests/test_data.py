import json
import tempfile
import unittest
from pathlib import Path

from glm_dflash2.data import (
    SOURCE_COLUMNS,
    iter_source_records,
    normalize_row,
    source_belongs_to_shard,
)


class NormalizeRowTest(unittest.TestCase):
    def base_row(self):
        return {key: "" for key in SOURCE_COLUMNS} | {
            "id": "sample-1",
            "prompt": "fix the test",
            "category": "debug_test_fix",
            "context_json": "{}",
            "metadata_json": "{}",
        }

    def test_plain_prompt_becomes_single_user_turn(self):
        sample = normalize_row(self.base_row(), "part.parquet", 3, 10)
        self.assertEqual(sample.sample_id, "sample-1")
        self.assertEqual(sample.messages, [{"role": "user", "content": "fix the test"}])
        self.assertEqual(sample.global_index, 10)
        self.assertEqual(sample.source_row, 3)

    def test_valid_context_messages_are_preserved(self):
        row = self.base_row()
        row["context_json"] = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "Use the shell."},
                    {"role": "user", "content": "inspect"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
                    {"role": "tool", "content": "ok"},
                ],
                "tools": [{"type": "function", "function": {"name": "bash"}}],
            }
        )
        sample = normalize_row(row, "part.parquet", 0, 0)
        self.assertEqual([m["role"] for m in sample.messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(sample.tools[0]["function"]["name"], "bash")
        self.assertEqual(sample.conversation_source, "context_json.messages")

    def test_nested_unparsed_messages_are_preserved(self):
        row = self.base_row()
        row["context_json"] = json.dumps(
            {
                "unparsed_text": json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "terminal agent"},
                            {"role": "user", "content": "fix it"},
                        ]
                    }
                )
            }
        )
        sample = normalize_row(row, "part.parquet", 0, 0)
        self.assertEqual(sample.messages[0]["content"], "terminal agent")
        self.assertEqual(sample.conversation_source, "context_json.unparsed_text.messages")

    def test_nested_unparsed_messages_allow_raw_control_characters(self):
        row = self.base_row()
        invalid_strict_json = '{"messages":[{"role":"user","content":"line1\nline2"}]}'
        row["context_json"] = json.dumps({"unparsed_text": invalid_strict_json})
        sample = normalize_row(row, "part.parquet", 0, 0)
        self.assertEqual(sample.messages[0]["content"], "line1\nline2")
        self.assertEqual(sample.conversation_source, "context_json.unparsed_text.messages")

    def test_truncated_nested_messages_fall_back_explicitly(self):
        row = self.base_row()
        row["context_json"] = json.dumps(
            {"unparsed_text": '{"messages":[{"role":"user"\n...[TRUNCATED]...\n,"tools":[]}' }
        )
        sample = normalize_row(row, "part.parquet", 0, 0)
        self.assertEqual(sample.messages, [{"role": "user", "content": "fix the test"}])
        self.assertEqual(sample.conversation_source, "context_json.truncated_messages_fallback_to_prompt")

    def test_truncated_marker_wins_even_when_permissive_json_can_parse_prefix(self):
        row = self.base_row()
        row["context_json"] = json.dumps(
            {
                "unparsed_text": (
                    '{"messages":[{"role":"user","content":"stale history"}],'
                    '"note":"...[TRUNCATED]..."}'
                )
            }
        )
        sample = normalize_row(row, "part.parquet", 0, 0)
        self.assertEqual(sample.messages, [{"role": "user", "content": "fix the test"}])
        self.assertEqual(sample.conversation_source, "context_json.truncated_messages_fallback_to_prompt")

    def test_illegal_message_role_falls_back_instead_of_reaching_chat_template(self):
        row = self.base_row()
        row["context_json"] = json.dumps(
            {"messages": [{"role": "critic", "content": "not a supported chat role"}]}
        )
        sample = normalize_row(row, "part.parquet", 0, 0)
        self.assertEqual(sample.messages, [{"role": "user", "content": "fix the test"}])
        self.assertEqual(sample.conversation_source, "context_json.invalid_messages_fallback_to_prompt")

    def test_string_encoded_tool_schema_is_decoded(self):
        row = self.base_row()
        row["context_json"] = json.dumps(
            {
                "messages": [{"role": "user", "content": "run it"}],
                "tools": [json.dumps({"type": "function", "function": {"name": "bash"}})],
            }
        )
        sample = normalize_row(row, "part.parquet", 0, 0)
        self.assertEqual(sample.tools[0]["function"]["name"], "bash")

    def test_file_before_change_includes_source_contents(self):
        row = self.base_row()
        row["context_json"] = json.dumps({"path": "src/a.py", "old_contents": "print('old')"})
        sample = normalize_row(row, "part.parquet", 0, 0)
        content = sample.messages[0]["content"]
        self.assertIn("src/a.py", content)
        self.assertIn("print('old')", content)
        self.assertIn("fix the test", content)
        self.assertEqual(sample.conversation_source, "context_json.static_context+prompt")

    def test_review_context_keeps_outer_task_pr_metadata_and_thread_replies(self):
        row = self.base_row()
        row["context_json"] = json.dumps(
            {
                "task": "Review this change carefully",
                "pr_url": "https://example.test/pull/7",
                "pr_title": "Fix parser",
                "review_prefix": [
                    {
                        "type": "review_comment",
                        "reviewer": "alice",
                        "reviewer_type": "human",
                        "state": "changes_requested",
                        "body": "Handle empty input",
                        "thread_replies": [
                            {"author": "bob", "body": "Fixed in the latest revision"}
                        ],
                    }
                ],
            }
        )
        sample = normalize_row(row, "part.parquet", 0, 0)
        content = sample.messages[0]["content"]
        self.assertIn("Review this change carefully", content)
        self.assertIn("https://example.test/pull/7", content)
        self.assertIn("Fix parser", content)
        self.assertIn("reviewer_type: human", content)
        self.assertIn("state: changes_requested", content)
        self.assertIn("Fixed in the latest revision", content)

    def test_planning_task_is_not_dropped_when_no_event_prefix_exists(self):
        row = self.base_row()
        row["context_json"] = json.dumps(
            {"task": "Inspect the repository and propose a migration plan", "repo": "org/project"}
        )
        sample = normalize_row(row, "part.parquet", 0, 0)
        self.assertIn("Inspect the repository", sample.messages[0]["content"])
        self.assertIn("org/project", sample.messages[0]["content"])
        self.assertEqual(sample.conversation_source, "context_json.static_context+prompt")

    def test_incomplete_user_history_is_explicitly_marked(self):
        row = self.base_row()
        row["prompt"] = "continue"
        row["context_json"] = json.dumps(
            {
                "artifact": {"type": "issue", "title": "Bug", "body": "details"},
                "prior_user_prompts": [{"prompt": "first request"}],
            }
        )
        sample = normalize_row(row, "part.parquet", 0, 0)
        content = sample.messages[0]["content"]
        self.assertIn("assistant replies are unavailable", content)
        self.assertIn("first request", content)
        self.assertIn("continue", content)
        self.assertEqual(sample.conversation_source, "context_json.incomplete_user_history+prompt")

    def test_malformed_context_falls_back_to_prompt(self):
        row = self.base_row()
        row["context_json"] = "not-json"
        sample = normalize_row(row, "part.parquet", 0, 0)
        self.assertEqual(sample.messages[-1]["content"], "fix the test")
        self.assertEqual(sample.conversation_source, "prompt")

    def test_empty_prompt_is_rejected(self):
        row = self.base_row()
        row["prompt"] = ""
        with self.assertRaisesRegex(ValueError, "empty prompt"):
            normalize_row(row, "part.parquet", 0, 0)

    def test_shards_are_disjoint_and_complete(self):
        assigned = [[i for i in range(50) if source_belongs_to_shard(i, rank, 4)] for rank in range(4)]
        self.assertEqual(sorted(i for shard in assigned for i in shard), list(range(50)))
        self.assertEqual(len(set(i for shard in assigned for i in shard)), 50)

    def test_parquet_iteration_is_stable_and_supports_range(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [self.base_row() | {"id": f"id-{i}", "prompt": f"p-{i}"} for i in range(5)]
            pq.write_table(pa.Table.from_pylist(rows), root / "b.parquet")
            pq.write_table(pa.Table.from_pylist(rows[:2]), root / "a.parquet")
            got = list(iter_source_records(root, start_index=1, end_index=6, shard_index=1, shard_count=2))
            self.assertEqual([x.global_index for x in got], [1, 3, 5])
            self.assertEqual([x.source_path for x in got[:1]], ["a.parquet"])


if __name__ == "__main__":
    unittest.main()
