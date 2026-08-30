from __future__ import annotations

import torch
import pytest

from omni_stage_c.chunked_lm_head import AdditiveScalar
from omni_stage_c.training_loop import _global_component_mean
from tools.train_thinker_drafter import run_tiny_smoke


def test_all_method_losses_reach_a_real_optimizer_step():
    for method, block_size in (("dflash", 8), ("dflash", 16),
                               ("dflash2", 8), ("dflash2", 16),
                               ("dspark", 8)):
        result = run_tiny_smoke(method=method, block_size=block_size)
        assert result["finite_loss"] is True
        assert result["optimizer_steps"] == 1


def test_component_mean_preserves_its_own_denominator_and_gradient():
    numerator = torch.tensor(6.0, requires_grad=True)
    component = AdditiveScalar(
        numerator=numerator,
        denominator=torch.tensor(3.0),
        mean=numerator / 3.0,
    )
    loss, logged, valid = _global_component_mean(component)
    loss.backward()
    assert valid is True
    assert logged.item() == 2.0
    assert numerator.grad.item() == pytest.approx(1.0 / 3.0)

    zero_loss, zero_log, zero_valid = _global_component_mean(
        AdditiveScalar(numerator * 0.0, torch.tensor(0.0), numerator * 0.0)
    )
    assert zero_valid is False
    assert zero_loss.item() == zero_log.item() == 0.0
