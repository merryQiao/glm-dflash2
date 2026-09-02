from __future__ import annotations

import pytest

from glm53_w4.contracts import validate_w4a8_target_config


def _config(marker: str = "W4A8") -> dict[str, object]:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 192,
        "vocab_size": 154880,
        "tie_word_embeddings": False,
        "quantization_config": {"model_quant_type": marker},
    }


def test_w4a8_contract_accepts_w4a8_marker() -> None:
    result = validate_w4a8_target_config(_config())
    assert result["quantization"] == "W4A8"
    assert result["schema"] == "formal-glm53-w4a8-v2"


def test_w4a8_contract_rejects_other_quantization() -> None:
    with pytest.raises(ValueError, match="W4A8"):
        validate_w4a8_target_config(_config("W8A8"))
    with pytest.raises(ValueError, match="W4A8"):
        validate_w4a8_target_config(_config("W4A8C8"))


def test_w4a8_contract_accepts_dynamic_marker_but_not_mixed_w8a8() -> None:
    config = _config()
    config["quantization_config"] = {}
    description = {"layers": {"q_proj": {"quant_type": "W4A8_DYNAMIC_PER_GROUP"}}}
    result = validate_w4a8_target_config(config, quant_description=description)
    assert result["quantization"] == "W4A8"
    with pytest.raises(ValueError, match="W4A8"):
        validate_w4a8_target_config(
            config,
            quant_description={
                "layers": {
                    "q_proj": {"quant_type": "W4A8_DYNAMIC"},
                    "k_proj": {"quant_type": "W8A8"},
                }
            },
        )
