from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from tools.generate_trajectories import _existing_ids

from glm_dflash2.agent_trajectory import ChatCompletionConfig, OpenAIChatClient
from glm_dflash2.sglang_stage_a import (
    AttemptErrorLedger,
    CommittedJsonlWriter,
    SGLangServerConfig,
    build_server_command,
    owns_source_index,
)


class SGLangStageATest(unittest.TestCase):
    def test_glm52_server_command_uses_model_specific_parsers(self):
        command = build_server_command(
            SGLangServerConfig(
                python="/opt/python",
                model_path=Path("/models/GLM-5.2"),
                served_model_name="GLM-5.2",
                tp_size=16,
                host="127.0.0.1",
                port=30000,
            )
        )
        self.assertEqual(command[:3], ["/opt/python", "-m", "sglang.launch_server"])
        self.assertEqual(command[command.index("--reasoning-parser") + 1], "glm45")
        self.assertEqual(command[command.index("--tool-call-parser") + 1], "glm47")
        self.assertEqual(command[command.index("--tp-size") + 1], "16")
        self.assertEqual(command[command.index("--device") + 1], "npu")
        self.assertEqual(
            command[command.index("--attention-backend") + 1], "ascend"
        )
        self.assertEqual(
            command[command.index("--max-total-tokens") + 1], "131072"
        )
        self.assertIn("--trust-remote-code", command)

    def test_glm52_server_command_exposes_optional_ascend_moe_settings(self):
        command = build_server_command(
            SGLangServerConfig(
                python="python",
                model_path=Path("/models/GLM-5.2-w8a8"),
                quantization="modelslim",
                moe_a2a_backend="deepep",
                deepep_mode="auto",
            )
        )
        self.assertEqual(command[command.index("--quantization") + 1], "modelslim")
        self.assertEqual(command[command.index("--moe-a2a-backend") + 1], "deepep")
        self.assertEqual(command[command.index("--deepep-mode") + 1], "auto")

    def test_chat_payload_preserves_tool_order_and_requests_all_token_ids(self):
        client = OpenAIChatClient(
            ChatCompletionConfig(
                endpoint="http://localhost:30000",
                model="GLM-5.2",
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                reasoning_effort=None,
                chat_template_kwargs={"enable_thinking": True, "custom": "kept"},
                return_token_ids=True,
            )
        )
        response = Mock()
        response.ok = True
        response.json.return_value = {
            "id": "r1",
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
                "prompt_token_ids": [1, 2],
                "response_token_ids": [3, 4],
            }],
        }
        client.session.post = Mock(return_value=response)
        tools = [
            {"type": "function", "function": {"name": "first"}},
            {"type": "function", "function": {"name": "second"}},
        ]
        value = client.complete([{"role": "user", "content": "hi"}], tools)
        payload = client.session.post.call_args.kwargs["json"]
        self.assertEqual([tool["function"]["name"] for tool in payload["tools"]], ["first", "second"])
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True, "custom": "kept"})
        self.assertTrue(payload["return_prompt_token_ids"])
        self.assertTrue(payload["return_token_ids"])
        self.assertEqual(value["_response_metadata"]["prompt_token_ids"], [1, 2])
        self.assertEqual(value["_response_metadata"]["response_token_ids"], [3, 4])

    def test_modulo_sharding_uses_global_source_index(self):
        self.assertTrue(owns_source_index(7, shard_index=1, shard_count=3))
        self.assertFalse(owns_source_index(8, shard_index=1, shard_count=3))

    def test_committed_jsonl_writer_fsyncs_complete_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"
            with CommittedJsonlWriter(path, truncate=True) as writer:
                writer.append({"id": "a", "value": "中文"})
            raw = path.read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            self.assertEqual(json.loads(raw.decode("utf-8")), {"id": "a", "value": "中文"})

    def test_error_ledger_survives_restart_and_tracks_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "errors.jsonl"
            with AttemptErrorLedger(path) as ledger:
                ledger.record_error(
                    sample_id="a", source_index=3, error=RuntimeError("workspace failed")
                )
                self.assertEqual(ledger.unresolved_ids, frozenset({"a"}))
            with AttemptErrorLedger(path) as ledger:
                self.assertEqual(ledger.unresolved_ids, frozenset({"a"}))
                ledger.resolve(sample_id="a", source_index=3)
                self.assertFalse(ledger.unresolved_ids)

    def test_existing_output_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"
            path.write_text('{"id":"a"}\n{"id":"a"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate.*a"):
                _existing_ids(path)


if __name__ == "__main__":
    unittest.main()
