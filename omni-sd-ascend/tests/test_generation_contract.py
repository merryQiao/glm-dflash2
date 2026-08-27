from __future__ import annotations

import unittest

from omni_sd.vllm_ascend_generation import completion_payload


class _Completion:
    token_ids = [10, 11, 151645]
    text = "answer"
    finish_reason = "stop"


class _RequestOutput:
    prompt_token_ids = [1, 2, 3]
    outputs = [_Completion()]


class GenerationContractTests(unittest.TestCase):
    def test_payload_uses_exact_engine_ids(self):
        payload = completion_payload(_RequestOutput(), eos_token_id=151645)
        self.assertEqual(payload["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(payload["response_token_ids"], [10, 11, 151645])
        self.assertEqual(payload["finish_reason"], "eos")

    def test_empty_token_output_is_rejected(self):
        output = _RequestOutput()
        output.outputs = [type("Empty", (), {"token_ids": [], "text": "", "finish_reason": "length"})()]
        with self.assertRaisesRegex(ValueError, "empty"):
            completion_payload(output, eos_token_id=151645)


if __name__ == "__main__":
    unittest.main()
