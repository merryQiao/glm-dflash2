from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from tests.helpers import complete_config
from omni_sd.vllm_ascend_hidden import (
    HiddenContractError,
    extractor_engine_kwargs,
    load_connector_tensors,
)


class HiddenProviderTests(unittest.TestCase):
    def test_engine_uses_native_extract_hidden_states_mode(self):
        kwargs = extractor_engine_kwargs(complete_config())
        spec = kwargs["speculative_config"]
        self.assertEqual(spec["method"], "extract_hidden_states")
        self.assertEqual(spec["num_speculative_tokens"], 1)
        self.assertEqual(
            spec["draft_model_config"]["hf_config"]["eagle_aux_hidden_state_layer_ids"],
            [1, 12, 24, 36, 47, 48],
        )

    def test_raw_final_layer_can_be_normalized_offline(self):
        from omni_sd.vllm_ascend_hidden import FinalRMSNorm

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hidden.safetensors"
            raw = torch.arange(1, 1 + 3 * 2048, dtype=torch.float32).reshape(3, 2048)
            save_file(
                {
                    "token_ids": torch.tensor([1, 2, 3]),
                    "hidden_states": torch.cat(
                        [
                            torch.zeros(3, 5, 2048, dtype=torch.bfloat16),
                            raw.to(torch.bfloat16)[:, None],
                        ],
                        dim=1,
                    ),
                },
                str(path),
            )
            tensors = load_connector_tensors(
                path,
                [1, 2, 3],
                complete_config(),
                normalizer=FinalRMSNorm(torch.ones(2048), 1.0e-6),
            )
            self.assertEqual(tuple(tensors["hidden_states"].shape), (3, 5, 2048))
            self.assertEqual(tuple(tensors["final_hidden_states"].shape), (3, 2048))

    def test_connector_record_requires_exact_token_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hidden.safetensors"
            save_file(
                {
                    "token_ids": torch.tensor([1, 2, 4]),
                    "hidden_states": torch.zeros(3, 5, 2048, dtype=torch.bfloat16),
                    "final_hidden_states": torch.zeros(3, 2048, dtype=torch.bfloat16),
                },
                str(path),
            )
            with self.assertRaisesRegex(HiddenContractError, "token"):
                load_connector_tensors(path, [1, 2, 3], complete_config())

    def test_final_normalized_hidden_fails_closed_when_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hidden.safetensors"
            save_file(
                {
                    "token_ids": torch.tensor([1, 2, 3]),
                    "hidden_states": torch.zeros(3, 5, 2048, dtype=torch.bfloat16),
                },
                str(path),
            )
            with self.assertRaisesRegex(HiddenContractError, "final normalized"):
                load_connector_tensors(path, [1, 2, 3], complete_config())

    def test_valid_record_preserves_archival_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hidden.safetensors"
            save_file(
                {
                    "token_ids": torch.tensor([1, 2, 3]),
                    "hidden_states": torch.zeros(3, 5, 2048, dtype=torch.bfloat16),
                    "final_hidden_states": torch.zeros(3, 2048, dtype=torch.bfloat16),
                },
                str(path),
            )
            tensors = load_connector_tensors(path, [1, 2, 3], complete_config())
            self.assertEqual(tuple(tensors["hidden_states"].shape), (3, 5, 2048))
            self.assertEqual(tuple(tensors["final_hidden_states"].shape), (3, 2048))


if __name__ == "__main__":
    unittest.main()
