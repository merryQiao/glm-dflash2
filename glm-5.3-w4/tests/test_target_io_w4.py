from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from glm53_w4.target_io import extract_w4a8_target_io, load_frozen_target_io


def _checkpoint(root: Path, marker: str = "W4A8") -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GlmMoeDsaForCausalLM"],
                "model_type": "glm_moe_dsa",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 78,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "head_dim": 192,
                "vocab_size": 7,
                "tie_word_embeddings": False,
                "quantization_config": {"model_quant_type": marker},
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    save_file(
        {
            "model.embed_tokens.weight": torch.randn(7, 4, dtype=torch.bfloat16),
            "lm_head.weight": torch.randn(7, 4, dtype=torch.bfloat16),
        },
        root / "model.safetensors",
    )
    return root


def test_w4a8_target_io_has_independent_schema(tmp_path: Path) -> None:
    manifest = extract_w4a8_target_io(
        _checkpoint(tmp_path / "source"),
        tmp_path / "io",
        expected_hidden_size=4,
        expected_vocab_size=7,
    )
    assert manifest["target_quantization"] == "W4A8"
    assert manifest["schema"] == "formal-glm53-w4a8-target-io-v2"
    frozen = load_frozen_target_io(tmp_path / "io")
    assert frozen.embed_tokens.weight.dtype == torch.bfloat16


def test_w4a8_target_io_rejects_w8a8(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="W4A8"):
        extract_w4a8_target_io(
            _checkpoint(tmp_path / "source", marker="W8A8"),
            tmp_path / "io",
            expected_hidden_size=4,
            expected_vocab_size=7,
        )
