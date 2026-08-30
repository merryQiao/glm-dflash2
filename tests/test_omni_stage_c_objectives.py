from __future__ import annotations

import torch

from omni_stage_c.objectives import depth_weights, selector_cross_entropy


def test_dflash_gamma_depth_weights():
    b8 = depth_weights(block_size=8, gamma=4.0, device=torch.device("cpu"))
    b16 = depth_weights(block_size=16, gamma=7.0, device=torch.device("cpu"))
    assert b8.shape == (7,)
    assert b16.shape == (15,)
    assert torch.all(b8[1:] < b8[:-1])
    assert torch.all(b16[1:] < b16[:-1])


def test_dflash2_miss_has_zero_selector_denominator():
    result = selector_cross_entropy(
        torch.tensor([[[0, 1]]]),
        torch.tensor([[[3.0, 2.0]]]),
        torch.tensor([[7]]),
        weights=torch.ones(1, 1),
    )
    assert result.denominator.item() == 0.0
    assert result.mean.item() == 0.0
    assert result.mean.requires_grad
