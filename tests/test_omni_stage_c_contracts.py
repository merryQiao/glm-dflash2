from __future__ import annotations

import pytest

from omni_stage_c.contracts import (
    DEFAULT_MASK_TOKEN_ID,
    DRAFT_CONTRACT,
    METHOD_BLOCK_SIZES,
    TARGET_CONTRACT,
    validate_method_block,
)


def test_official_nested_thinker_contract():
    assert TARGET_CONTRACT.num_layers == 48
    assert TARGET_CONTRACT.hidden_size == 2048
    assert TARGET_CONTRACT.vocab_size == 152064
    assert TARGET_CONTRACT.logical_layer_ids == (1, 12, 24, 36, 47)
    assert TARGET_CONTRACT.num_attention_heads == 32
    assert TARGET_CONTRACT.num_key_value_heads == 4
    assert TARGET_CONTRACT.head_dim == 128
    assert TARGET_CONTRACT.mrope_section == (24, 20, 20)
    assert TARGET_CONTRACT.rms_norm_eps == 1e-6
    assert DRAFT_CONTRACT.intermediate_size == 6144
    assert DEFAULT_MASK_TOKEN_ID == 152063


def test_method_block_contract():
    assert METHOD_BLOCK_SIZES == {
        "dflash": (8, 16),
        "dflash2": (8, 16),
        "dspark": (8,),
    }
    validate_method_block("dspark", 8)
    with pytest.raises(ValueError, match="DSpark"):
        validate_method_block("dspark", 16)
