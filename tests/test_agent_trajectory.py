from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from glm_dflash2.agent_trajectory import (
    TOOL_DEFINITIONS,
    WorkspaceToolExecutor,
    render_with_assistant_mask,
    rollout_from_messages,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, messages, tools):
        del messages, tools
        return next(self.responses)


class CharacterChatTokenizer:
    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt, **kwargs):
        del tokenize, kwargs
        rendered = "<TOOLS>" + json.dumps(tools, sort_keys=True)
        for message in messages:
            role = message["role"]
            rendered += f"<{role}>" + str(message.get("reasoning_content") or "")
            rendered += str(message.get("content") or "")
            rendered += json.dumps(message.get("tool_calls") or [], sort_keys=True)
            rendered += f"</{role}>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return [ord(char) for char in rendered]


class AgentTrajectoryTest(unittest.TestCase):
    def test_multiturn_rollout_inserts_real_tool_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "a.py").write_text("value = 1\n", encoding="utf-8")
            client = FakeClient(
                [
                    {
                        "role": "assistant",
                        "content": "inspect",
                        "tool_calls": [{
                            "id": "read-1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": {"path": "a.py"}},
                        }],
                    },
                    {"role": "assistant", "content": "done"},
                ]
            )
            trajectory = rollout_from_messages(
                episode_id="one",
                initial_messages=[{"role": "user", "content": "inspect"}],
                client=client,
                executor=WorkspaceToolExecutor(root),
                tools=TOOL_DEFINITIONS,
                require_tool=True,
            )
            self.assertTrue(trajectory["validation"]["valid"])
            self.assertEqual([m["role"] for m in trajectory["messages"]], ["user", "assistant", "tool", "assistant"])
            self.assertIn("value = 1", json.loads(trajectory["messages"][2]["content"])["content"])

    def test_mask_marks_only_generated_assistant_turns_and_keeps_tools(self):
        tools = TOOL_DEFINITIONS[:1]
        messages = [
            {"role": "assistant", "content": "historical"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "reasoning_content": "think", "content": "first"},
            {"role": "tool", "content": "observation", "tool_call_id": "c1"},
            {"role": "assistant", "content": "final"},
        ]
        ids, mask = render_with_assistant_mask(
            CharacterChatTokenizer(), messages, tools, assistant_start_index=2
        )
        rendered = "".join(map(chr, ids))
        historical = rendered.index("historical")
        observation = rendered.index("observation")
        first = rendered.index("thinkfirst")
        final = rendered.index("final")
        self.assertEqual(mask[historical], 0)
        self.assertEqual(mask[observation], 0)
        self.assertTrue(all(mask[first : first + len("thinkfirst")]))
        self.assertTrue(all(mask[final : final + len("final")]))
        self.assertIn(json.dumps(tools, sort_keys=True), rendered)


if __name__ == "__main__":
    unittest.main()
