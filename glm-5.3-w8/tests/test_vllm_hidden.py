from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch
from safetensors.torch import save_file

from glm53_w8.vllm_hidden import (
    FinalRMSNorm,
    HiddenContractError,
    build_engine_kwargs,
    load_final_normalizer,
    load_connector_tensors,
    trajectory_tokens,
)


def test_engine_uses_vllm_ascend_w8a8_extractor(tmp_path: Path) -> None:
    kwargs = build_engine_kwargs(
        model_path="/models/glm53-w8a8",
        tensor_parallel_size=8,
        scratch_root=tmp_path,
    )
    assert kwargs["quantization"] == "ascend"
    assert kwargs["tensor_parallel_size"] == 8
    assert kwargs["speculative_config"]["method"] == "extract_hidden_states"
    ids = kwargs["speculative_config"]["draft_model_config"]["hf_config"][
        "eagle_aux_hidden_state_layer_ids"
    ]
    assert ids == [1, 20, 38, 56, 75, 78]
    assert kwargs["enable_chunked_prefill"] is False
    assert kwargs["enable_prefix_caching"] is False


def test_trajectory_replay_never_retokenizes() -> None:
    ids, mask, sample_id = trajectory_tokens(
        {
            "sample_id": "x",
            "prompt_token_ids": [1, 2],
            "response_token_ids": [3, 4],
        }
    )
    assert ids == [1, 2, 3, 4]
    assert mask == [False, False, True, True]
    assert sample_id == "x"
    with pytest.raises(HiddenContractError):
        trajectory_tokens({"sample_id": "x", "messages": [{"role": "user"}]})


def test_trajectory_rejects_ids_outside_formal_glm_vocabulary() -> None:
    with pytest.raises(HiddenContractError, match="vocabulary"):
        trajectory_tokens(
            {"sample_id": "x", "input_ids": [1, 154880], "loss_mask": [False, True]}
        )


def test_raw_last_decoder_state_is_normalized_exactly() -> None:
    norm = FinalRMSNorm(torch.tensor([1.0, 2.0]), epsilon=1e-5)
    raw = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
    actual = norm(raw)
    expected = (
        raw.float()
        * torch.rsqrt(raw.float().square().mean(-1, keepdim=True) + 1e-5)
        * torch.tensor([1.0, 2.0], dtype=torch.float32)
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected)


def test_final_normalizer_reads_target_epsilon(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"rms_norm_eps": 1e-3}), encoding="utf-8"
    )
    save_file(
        {"model.norm.weight": torch.ones(2, dtype=torch.bfloat16)},
        tmp_path / "model.safetensors",
    )
    normalizer = load_final_normalizer(tmp_path)
    actual = normalizer(torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16))
    expected = torch.full((1, 2), 1.0 / (1.0 + 1e-3) ** 0.5, dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected)


def test_final_normalizer_scans_unindexed_weight_shard(tmp_path: Path) -> None:
    save_file(
        {"model.norm.weight": torch.ones(2, dtype=torch.bfloat16)},
        tmp_path / "model-00001.safetensors",
    )
    normalizer = load_final_normalizer(tmp_path)
    assert normalizer.weight.shape == (2,)


def test_connector_requires_exact_tokens_and_six_streams(tmp_path: Path) -> None:
    hidden = torch.randn(4, 6, 3, dtype=torch.bfloat16)
    loader = lambda _: {  # noqa: E731
        "token_ids": torch.tensor([1, 2, 3, 4]),
        "hidden_states": hidden,
    }
    tensors = load_connector_tensors(
        tmp_path / "record.safetensors",
        [1, 2, 3, 4],
        normalizer=FinalRMSNorm(torch.ones(3), 1e-5),
        hidden_size=3,
        tensor_loader=loader,
    )
    assert tensors["aux_hidden_states"].shape == (4, 5, 3)
    assert tensors["target_final_hidden"].shape == (4, 3)
    with pytest.raises(HiddenContractError, match="token IDs"):
        load_connector_tensors(
            tmp_path / "record.safetensors",
            [1, 9, 3, 4],
            normalizer=FinalRMSNorm(torch.ones(3), 1e-5),
            hidden_size=3,
            tensor_loader=loader,
        )
