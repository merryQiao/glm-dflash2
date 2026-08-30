from __future__ import annotations

import torch

from omni_stage_c.dflash2_model import CandidateSelector, DFlash2Model
from omni_stage_c.modeling_common import (
    DraftModelConfig,
    FullAttention,
    interleaved_mrope,
)


def tiny_config() -> DraftModelConfig:
    return DraftModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_aux_layers=5,
        vocab_size=32,
        rms_norm_eps=1e-6,
        mrope_section=(1, 1, 0),
    )


def test_gqa_projection_shapes_and_kv_repeat():
    attention = FullAttention(tiny_config())
    assert attention.query.weight.shape == (16, 16)
    assert attention.key.weight.shape == (8, 16)
    assert attention.value.weight.shape == (8, 16)
    assert attention.output.weight.shape == (16, 16)


def test_interleaved_mrope_matches_official_axis_selection():
    positions = torch.tensor(
        [[[0, 1, 2]], [[10, 11, 12]], [[20, 21, 22]]], dtype=torch.int64
    )
    cosine, sine = interleaved_mrope(
        positions,
        head_dim=6,
        theta=1000.0,
        sections=(1, 1, 1),
        dtype=torch.float32,
    )
    inv = 1.0 / (1000.0 ** (torch.arange(0, 6, 2).float() / 6.0))
    frequencies = torch.einsum("d,abl->abld", inv, positions.float())
    official = frequencies[0].clone()
    official[..., 1:3:3] = frequencies[1, ..., 1:3:3]
    official[..., 2:3:3] = frequencies[2, ..., 2:3:3]
    official = torch.cat((official, official), dim=-1)
    assert torch.allclose(cosine, official.cos())
    assert torch.allclose(sine, official.sin())


def test_selector_never_injects_target():
    selector = CandidateSelector(hidden_size=4, vocab_size=8, rank=2, top_k=2)
    hidden = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    lm_weight = torch.zeros(8, 4)
    lm_weight[0, 0] = 3.0
    lm_weight[1, 0] = 2.0
    candidates, _, hits = selector.training_forward(
        hidden, lm_weight, torch.tensor([0]), torch.tensor([7])
    )
    assert candidates.tolist() == [[0, 1]]
    assert hits.tolist() == [False]
    assert 7 not in candidates.tolist()[0]


def test_dflash2_has_official_dynamic_conv_and_selector():
    model = DFlash2Model(tiny_config())
    assert model.selector.rank == 256
    assert model.selector.top_k == 16
    first = model.layers[0]
    assert first.attention_conv.group_size == 16
    assert torch.count_nonzero(first.attention_conv.kernel_projection.weight) == 0


def test_compact_anchor_attention_matches_independent_blocks():
    torch.manual_seed(3)
    config = tiny_config()
    attention = FullAttention(config).eval()
    batch, anchors, block, context = 1, 3, 2, 4
    hidden = torch.randn(batch, anchors * block, config.hidden_size)
    target = torch.randn(batch, context, config.hidden_size)
    positions = torch.arange(context + anchors * block).view(1, 1, -1).expand(3, batch, -1)
    mask = torch.ones(batch, anchors, block, context + block, dtype=torch.bool)
    combined = attention(hidden, target, positions, mask)
    pieces = []
    for anchor in range(anchors):
        draft = positions[:, :, context + anchor * block:context + (anchor + 1) * block]
        local_positions = torch.cat((positions[:, :, :context], draft), dim=-1)
        pieces.append(attention(
            hidden[:, anchor * block:(anchor + 1) * block], target,
            local_positions, mask[:, anchor:anchor + 1],
        ))
    assert torch.allclose(combined, torch.cat(pieces, dim=1), atol=1e-5, rtol=1e-5)
