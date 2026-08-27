from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from glm_dflash2.dflash2_model import build_dflash2_config
from glm_dflash2.target_io import FrozenTargetIO


def tiny_config(*, block_size: int = 8, selector_rank: int = 4):
    return build_dflash2_config(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        target_layer_ids=[1, 20, 38, 56, 75],
        num_target_layers=78,
        block_size=block_size,
        mask_token_id=16,
        conv_group_size=4,
        selector_rank=selector_rank,
        selector_top_k=4,
        sliding_window=None,
    )


def tiny_target_io() -> FrozenTargetIO:
    embed = nn.Embedding(17, 8, dtype=torch.bfloat16)
    head = nn.Linear(8, 17, bias=False, dtype=torch.bfloat16)
    embed.weight.requires_grad_(False)
    head.weight.requires_grad_(False)
    manifest = {
        "schema": "glm-drafter-target-io-v2",
        "source_model_dir": "/models/GLM-5.2-BF16",
        "source_model_fingerprint": "model-sha",
        "model_revision": "revision-sha",
        "tokenizer_fingerprint": "tokenizer-sha",
        "model_type": "glm_moe_dsa",
        "vocab_size": 17,
        "hidden_size": 8,
        "weights_sha256": "target-io-sha",
        "tensors": {},
    }
    return FrozenTargetIO(embed, head, manifest)


def flip_one_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
