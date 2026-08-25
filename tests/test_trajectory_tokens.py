from __future__ import annotations

import json
import unittest

from glm_dflash2.trajectory_tokens import TrajectoryTokenError, freeze_trajectory_tokens


class StableTokenizer:
    name_or_path = "/models/GLM-5.2"
    chat_template = "stable-test-template"

    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt, **kwargs):
        del tokenize
        prefix = "TOOLS=" + json.dumps(tools, sort_keys=True, separators=(",", ":"))
        prefix += ";KW=" + json.dumps(kwargs, sort_keys=True, separators=(",", ":"))
        for message in messages:
            role = message["role"]
            prefix += f"<{role}>"
            prefix += str(message.get("reasoning_content") or "")
            prefix += str(message.get("content") or "")
            prefix += json.dumps(message.get("tool_calls") or [], sort_keys=True)
            prefix += f"</{role}>"
        if add_generation_prompt:
            prefix += "<assistant>"
        return [ord(char) for char in prefix]


def trajectory():
    return {
        "id": "sample-1",
        "messages": [
            {"role": "assistant", "content": "old"},
            {"role": "user", "content": "修复 bug"},
            {
                "role": "assistant",
                "reasoning_content": "分析",
                "content": "call",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "a.py"}},
                }],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "x=1"},
            {"role": "assistant", "content": "done"},
        ],
        "generation_start_message_index": 2,
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
    }


class TrajectoryTokensTest(unittest.TestCase):
    def test_freezes_full_replay_inputs_and_unshifted_assistant_mask(self):
        frozen = freeze_trajectory_tokens(
            StableTokenizer(), trajectory(), chat_template_kwargs={"enable_thinking": True}
        )
        rendered = "".join(map(chr, frozen["input_ids"]))
        self.assertEqual(len(frozen["input_ids"]), len(frozen["loss_mask"]))
        self.assertEqual(frozen["loss_mask"][-1], 0)
        self.assertEqual(frozen["loss_mask"][rendered.index("old")], 0)
        self.assertEqual(frozen["loss_mask"][rendered.index("x=1")], 0)
        self.assertEqual(frozen["loss_mask"][rendered.index("分析call")], 1)
        self.assertEqual(frozen["loss_mask"][rendered.index("done")], 1)
        self.assertIn("enable_thinking", rendered)
        self.assertEqual(frozen["token_contract"]["mask_semantics"], "dflash_target_token")
        self.assertEqual(frozen["token_contract"]["supervised_tokens"], sum(frozen["loss_mask"]))

    def test_rejects_hard_length_bound_instead_of_truncating(self):
        with self.assertRaisesRegex(TrajectoryTokenError, "exceeds max_sequence_tokens"):
            freeze_trajectory_tokens(
                StableTokenizer(), trajectory(), chat_template_kwargs={}, max_sequence_tokens=8
            )

    def test_rejects_trajectory_without_generated_assistant_tokens(self):
        value = trajectory()
        value["generation_start_message_index"] = len(value["messages"])
        with self.assertRaisesRegex(TrajectoryTokenError, "no supervised assistant tokens"):
            freeze_trajectory_tokens(StableTokenizer(), value, chat_template_kwargs={})

    def test_checks_server_round_token_ids_against_frozen_replay(self):
        value = trajectory()
        tokenizer = StableTokenizer()
        tools = value["tools"]
        kwargs = {"enable_thinking": True}
        generated_indices = [2, 4]
        metadata = []
        for index in generated_indices:
            prompt = tokenizer.apply_chat_template(
                value["messages"][:index], tools=tools, tokenize=True,
                add_generation_prompt=True, **kwargs
            )
            through = tokenizer.apply_chat_template(
                value["messages"][: index + 1], tools=tools, tokenize=True,
                add_generation_prompt=False, **kwargs
            )
            metadata.append(
                {"prompt_token_ids": prompt, "response_token_ids": through[len(prompt):]}
            )
        value["response_metadata"] = metadata
        frozen = freeze_trajectory_tokens(tokenizer, value, chat_template_kwargs=kwargs)
        self.assertEqual(frozen["token_contract"]["round_token_checks"], ["matched", "matched"])

        value["response_metadata"][0]["response_token_ids"] = [999]
        with self.assertRaisesRegex(TrajectoryTokenError, "response token IDs differ"):
            freeze_trajectory_tokens(tokenizer, value, chat_template_kwargs=kwargs)

    def test_checks_prompt_ids_when_server_does_not_return_response_ids(self):
        value = trajectory()
        tokenizer = StableTokenizer()
        tools = value["tools"]
        kwargs = {"enable_thinking": True}
        generated_indices = [2, 4]
        value["response_metadata"] = [
            {
                "prompt_token_ids": tokenizer.apply_chat_template(
                    value["messages"][:index],
                    tools=tools,
                    tokenize=True,
                    add_generation_prompt=True,
                    **kwargs,
                ),
                "response_token_ids": None,
            }
            for index in generated_indices
        ]

        frozen = freeze_trajectory_tokens(tokenizer, value, chat_template_kwargs=kwargs)
        self.assertEqual(
            frozen["token_contract"]["round_token_checks"],
            ["prompt_matched", "prompt_matched"],
        )

        value["response_metadata"][0]["prompt_token_ids"] = [999]
        with self.assertRaisesRegex(TrajectoryTokenError, "prompt token IDs differ"):
            freeze_trajectory_tokens(tokenizer, value, chat_template_kwargs=kwargs)


if __name__ == "__main__":
    unittest.main()
