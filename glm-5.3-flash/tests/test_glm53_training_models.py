from __future__ import annotations

import pytest
import torch

from glm53_drafters.dflash2_model import (
    BlockLocalDynamicConv,
    CandidateSelector,
    DFlash2Model,
)
from glm53_drafters.dflash_model import DFlashModel
from glm53_drafters.dspark_model import DSparkModel
from glm53_drafters.modeling_common import DraftModelConfig


def tiny_config() -> DraftModelConfig:
    return DraftModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        num_aux_layers=3,
        vocab_size=31,
        rms_norm_eps=1e-5,
    )


def test_production_config_is_fixed_glm53_dense_full_attention() -> None:
    config = DraftModelConfig.production()
    assert config.hidden_size == 4096
    assert config.intermediate_size == 12288
    assert config.num_hidden_layers == 5
    assert config.num_attention_heads == config.num_key_value_heads == 64
    assert config.head_dim == 64
    assert config.num_aux_layers == 5
    assert config.vocab_size == 154880
    assert config.full_attention is True
    assert config.sliding_window is None
    assert config.initializer_range == 0.02


def test_official_style_initialization_preserves_stabilizing_zero_and_identity() -> None:
    torch.manual_seed(101)
    config = tiny_config()
    dflash = DFlashModel(config)
    assert dflash.target_projection.weight.std().item() == pytest.approx(
        config.initializer_range, abs=0.004
    )
    assert dflash.layers[0].attention.query.weight.std().item() == pytest.approx(
        config.initializer_range, abs=0.004
    )
    dflash2 = DFlash2Model(
        config, convolution_group_size=4, selector_rank=8, selector_top_k=5
    )
    assert dflash2.selector.hidden_projection.weight.std().item() == pytest.approx(
        config.initializer_range, abs=0.004
    )
    assert dflash2.selector.predecessor_codebook.std().item() == pytest.approx(
        config.initializer_range, abs=0.004
    )
    assert torch.count_nonzero(dflash2.selector.successor_codebook) == 0
    for layer in dflash2.layers:
        for convolution in (layer.attention_conv, layer.mlp_conv):
            assert convolution is not None
            assert torch.count_nonzero(convolution.kernel_projection.weight) == 0
            expected = torch.zeros_like(convolution.base_kernel)
            expected[:, 0].fill_(1.0)
            assert torch.equal(convolution.base_kernel, expected)


def test_dflash_projects_five_target_streams_and_has_finite_gradients() -> None:
    torch.manual_seed(1)
    model = DFlashModel(tiny_config())
    token_embeddings = torch.randn(2, 12, 16)
    auxiliary = torch.randn(2, 6, 3, 16)
    positions = torch.arange(18).view(1, -1).expand(2, -1)
    attention = torch.ones(2, 1, 12, 18, dtype=torch.bool)
    output = model(
        token_embeddings,
        auxiliary,
        position_ids=positions,
        attention_mask=attention,
        block_size=4,
    )
    assert output.shape == token_embeddings.shape
    assert model.target_projection.in_features == 3 * 16
    assert not hasattr(model, "recurrent_state")
    output.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_dynamic_convolution_is_identity_initialized_and_block_local() -> None:
    torch.manual_seed(2)
    convolution = BlockLocalDynamicConv(
        hidden_size=16, kernel_size=2, group_size=4
    )
    blocks = torch.randn(2, 3, 5, 16)
    baseline = convolution(blocks)
    assert torch.equal(baseline, blocks)

    changed = blocks.clone()
    changed[:, 1] += 1000
    changed_output = convolution(changed)
    assert torch.equal(changed_output[:, 0], baseline[:, 0])
    assert torch.equal(changed_output[:, 2], baseline[:, 2])


def test_candidate_selector_and_dflash2_shapes_use_tiny_dimensions() -> None:
    torch.manual_seed(3)
    config = tiny_config()
    selector = CandidateSelector(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        rank=8,
        top_k=5,
    )
    hidden = torch.randn(2, 12, config.hidden_size)
    lm_weight = torch.randn(config.vocab_size, config.hidden_size)
    predecessors = torch.randint(0, config.vocab_size, (2, 12))
    candidate_ids, candidate_logits = selector(hidden, lm_weight, predecessors)
    assert candidate_ids.shape == candidate_logits.shape == (2, 12, 5)

    model = DFlash2Model(
        config,
        convolution_group_size=4,
        selector_rank=8,
        selector_top_k=5,
    )
    auxiliary = torch.randn(2, 6, 3, config.hidden_size)
    positions = torch.arange(18).view(1, -1).expand(2, -1)
    attention = torch.ones(2, 1, 12, 18, dtype=torch.bool)
    output, ids, logits = model(
        hidden,
        auxiliary,
        lm_weight=lm_weight,
        predecessor_ids=predecessors,
        position_ids=positions,
        attention_mask=attention,
        block_size=4,
    )
    assert output.shape == hidden.shape
    assert ids.shape == logits.shape == (2, 12, 5)
    assert all(layer.attention_conv is not None and layer.mlp_conv is not None for layer in model.layers)


def test_method_models_expose_fsdp2_wrap_targets() -> None:
    config = tiny_config()
    dflash = DFlashModel(config)
    dflash2 = DFlash2Model(
        config, convolution_group_size=4, selector_rank=8, selector_top_k=5
    )
    dspark = DSparkModel(config, markov_rank=8)

    assert dflash.layers is dflash.backbone.layers
    assert dflash2.layers is dflash2.dflash.backbone.layers
    assert dflash2.candidate_selector is dflash2.selector
    assert dspark.layers is dspark.dflash.backbone.layers
    assert dspark.markov_head is dspark.markov
    assert dspark.confidence_head is dspark.confidence
