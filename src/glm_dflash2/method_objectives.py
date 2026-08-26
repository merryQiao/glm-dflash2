from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdditiveLoss:
    numerator: torch.Tensor
    denominator: torch.Tensor
    mean: torch.Tensor


def depth_weights(reference: torch.Tensor, gamma: float) -> torch.Tensor:
    if reference.ndim < 1:
        raise ValueError("reference must have a depth dimension")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    values = torch.exp(
        -torch.arange(reference.shape[-1], device=reference.device, dtype=torch.float32)
        / float(gamma)
    )
    return values.reshape((1,) * (reference.ndim - 1) + (-1,))


def depth_weighted_objective(
    token_losses: torch.Tensor, token_mask: torch.Tensor, *, gamma: float
) -> AdditiveLoss:
    if token_losses.shape != token_mask.shape:
        raise ValueError("token_losses and token_mask shapes differ")
    weights = depth_weights(token_losses, gamma) * token_mask.to(
        device=token_losses.device, dtype=torch.float32
    )
    numerator = (token_losses.float() * weights).sum()
    denominator = weights.sum()
    mean = numerator / denominator.clamp_min(1.0)
    # When denominator is zero numerator still depends on token_losses through
    # multiplication by zero, so ``mean`` remains a differentiable zero.
    return AdditiveLoss(numerator=numerator, denominator=denominator, mean=mean)
