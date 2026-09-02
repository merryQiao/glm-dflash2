from __future__ import annotations

import torch
import pytest

from glm53_w4.blocks import build_sliding_blocks
from glm53_w4.dflash2 import DFlash2Model
from glm53_w4.dspark import DSparkModel
from glm53_w4.modeling import DraftModelConfig, gather_context_halo


def test_draft_config_rejects_invalid_window_or_layer_contract() -> None:
    with pytest.raises(ValueError, match="unique"):
        DraftModelConfig(target_layer_ids=(1, 1, 3, 5, 7))
    with pytest.raises(ValueError, match="invalid block"):
        DraftModelConfig(conv_kernel_size=0)


def _config(block_size: int) -> DraftModelConfig:
    return DraftModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        target_layer_ids=(1, 3),
        block_size=block_size,
        mask_token_id=31,
        sliding_window=4,
        selector_top_k=4,
        selector_rank=4,
        markov_rank=4,
        conv_group_size=4,
        anchor_chunk_size=2,
    )


def _inputs(block_size: int):
    ids = torch.arange(12).reshape(1, -1) % 30
    mask = torch.ones_like(ids, dtype=torch.bool)
    anchors = torch.tensor([[2, 7, 9]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    blocks = build_sliding_blocks(
        ids,
        mask,
        anchors,
        keep,
        block_size=block_size,
        mask_token_id=31,
        sliding_window=4,
    )
    aux = torch.randn(1, 12, 2, 16)
    embedding = torch.nn.Embedding(32, 16)
    noise = embedding(blocks.noise_ids)
    return blocks, aux, noise


def test_context_halo_is_a_physical_gather() -> None:
    blocks, _, _ = _inputs(3)
    projected = torch.arange(12 * 2).reshape(1, 12, 2)
    context, valid, positions = gather_context_halo(
        projected, blocks, anchor_start=0, anchor_end=3
    )
    assert context.shape == (3, 4, 2)
    assert context[1].tolist() == projected[0, 3:7].tolist()
    assert valid[0].tolist() == [False, False, True, True]
    assert positions[1].tolist() == [3, 4, 5, 6]


def test_dflash2_chunked_forward_and_selector_shapes() -> None:
    config = _config(4)
    blocks, aux, noise = _inputs(4)
    model = DFlash2Model(config)
    hidden = model(noise_embedding=noise, target_hidden=aux, blocks=blocks)
    assert hidden.shape == (1, 3, 4, 16)
    candidate_ids = torch.randint(0, 32, (1, 3, 3, 4))
    unary = torch.randn(1, 3, 3, 4)
    scores = model.selector_scores(
        hidden[:, :, 1:], unary, candidate_ids, blocks.target_ids[:, :, :-1]
    )
    assert scores.shape == candidate_ids.shape


def test_dspark_uses_same_windowed_backbone_and_heads() -> None:
    config = _config(8)
    blocks, aux, noise = _inputs(8)
    model = DSparkModel(config)
    hidden = model(noise_embedding=noise, target_hidden=aux, blocks=blocks)
    predecessors = blocks.target_ids[:, :, :-1]
    assert hidden.shape == (1, 3, 8, 16)
    assert model.confidence_logits(hidden[:, :, 1:], predecessors).shape == (1, 3, 7)
    assert model.markov_scores(predecessors, 0, 5).shape == (1, 3, 7, 5)


def test_draft_config_records_sliding_attention_on_every_layer() -> None:
    config = _config(8)
    exported = config.to_dict()
    assert exported["sliding_window"] == 4
    assert exported["layer_types"] == ["sliding_attention", "sliding_attention"]


def test_gradient_checkpointing_preserves_dflash2_backward() -> None:
    config = _config(4)
    config = DraftModelConfig.from_dict({**config.to_dict(), "gradient_checkpointing": True})
    blocks, aux, noise = _inputs(4)
    model = DFlash2Model(config).train()
    hidden = model(noise_embedding=noise, target_hidden=aux, blocks=blocks)
    assert torch.isfinite(hidden).all()
    hidden.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
