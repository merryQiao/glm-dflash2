from __future__ import annotations

import pytest

from glm53_w4.contracts import (
    DFLASH2_METHOD,
    DSPARK_METHOD,
    DRAFT_CONTRACT,
    TARGET_CONTRACT,
    validate_method_contract,
    validate_w4a8_target_config,
)


def _target_config() -> dict[str, object]:
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
        "quantize": "w4a8",
        "quantization_config": {"model_quant_type": "W4A8"},
    }


def test_formal_glm53_contract_is_fixed() -> None:
    assert TARGET_CONTRACT.layer_ids == (1, 20, 38, 56, 75)
    assert TARGET_CONTRACT.num_hidden_layers == 78
    assert TARGET_CONTRACT.hidden_size == 6144
    assert TARGET_CONTRACT.vocab_size == 154880
    assert DRAFT_CONTRACT.num_hidden_layers == 5
    assert DRAFT_CONTRACT.head_dim == 64
    assert DRAFT_CONTRACT.sliding_window == 2048


def test_w4a8_config_accepts_modelslim_metadata() -> None:
    metadata = validate_w4a8_target_config(_target_config())
    assert metadata["quantization"] == "W4A8"
    assert metadata["model_type"] == "glm_moe_dsa"


def test_w4a8_config_accepts_nested_modelslim_description() -> None:
    config = _target_config()
    config["quantize"] = None
    config["quantization_config"] = {}
    description = {
        "model.layers.0.self_attn.q_proj.weight": {
            "type": "W4A8_DYNAMIC",
            "params": {"weight": {"dtype": "int8"}},
        },
        "optional": {"format": "ascend_v1"},
    }
    metadata = validate_w4a8_target_config(config, quant_description=description)
    assert metadata["quantization"] == "W4A8"


@pytest.mark.parametrize(
    "change",
    [
        {"model_type": "glm53_flash"},
        {"hidden_size": 4096},
        {"num_hidden_layers": 45},
        {"quantize": "fp8", "quantization_config": {"quant_method": "fp8"}},
        {"quantize": None, "quantization_config": {}},
    ],
)
def test_w4a8_config_rejects_other_targets(change: dict[str, object]) -> None:
    config = _target_config()
    config.update(change)
    with pytest.raises(ValueError):
        validate_w4a8_target_config(config)


def test_method_contracts_enforce_physical_blocks() -> None:
    validate_method_contract(DSPARK_METHOD, block_size=8)
    validate_method_contract(DFLASH2_METHOD, block_size=8)
    validate_method_contract(DFLASH2_METHOD, block_size=16)
    with pytest.raises(ValueError, match="DSpark"):
        validate_method_contract(DSPARK_METHOD, block_size=16)
    with pytest.raises(ValueError):
        validate_method_contract(DFLASH2_METHOD, block_size=32)
