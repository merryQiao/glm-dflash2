from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .chunked_lm_head import AdditiveScalar, additive_mean


@dataclass(frozen=True)
class DistributionDistance(AdditiveScalar):
    per_token: torch.Tensor


def depth_weights(*, block_size: int, gamma: float,
                  device: torch.device | None = None) -> torch.Tensor:
    if block_size < 2 or gamma <= 0:
        raise ValueError("block size must exceed one and gamma must be positive")
    return torch.exp(-torch.arange(block_size - 1, device=device, dtype=torch.float32) / gamma)


def selector_cross_entropy(candidate_ids: torch.Tensor, candidate_logits: torch.Tensor,
                           targets: torch.Tensor, *, weights: torch.Tensor | None = None
                           ) -> AdditiveScalar:
    matches = candidate_ids.eq(targets.unsqueeze(-1))
    hits = matches.any(-1)
    scale = torch.ones_like(targets, dtype=torch.float32) if weights is None else weights.float()
    scale = scale * hits.float()
    local = matches.long().argmax(-1)
    losses = F.cross_entropy(candidate_logits.reshape(-1, candidate_logits.shape[-1]).float(),
                             local.reshape(-1), reduction="none").reshape_as(targets)
    numerator, denominator = (losses * scale).sum(), scale.sum()
    mean = additive_mean(numerator, denominator)
    # A selector miss is a valid no-op microbatch, not an optimizer error.
    # Keep a differentiable scalar even in isolated CPU contract tests where
    # the supplied logits themselves do not require gradients.
    if not mean.requires_grad:
        mean = mean.detach().requires_grad_(True)
    return AdditiveScalar(numerator, denominator, mean)


def exact_total_variation(student_logits: torch.Tensor, target_logits: torch.Tensor,
                          weights: torch.Tensor) -> DistributionDistance:
    per_token = 0.5 * (
        student_logits.float().softmax(-1) - target_logits.float().softmax(-1)
    ).abs().sum(-1)
    scale = weights.float()
    numerator, denominator = (per_token * scale).sum(), scale.sum()
    return DistributionDistance(numerator, denominator,
                                additive_mean(numerator, denominator), per_token)


def confidence_bce(logits: torch.Tensor, tv: torch.Tensor,
                   weights: torch.Tensor) -> AdditiveScalar:
    target = (1.0 - tv.detach().float()).clamp(0.0, 1.0)
    losses = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    numerator, denominator = (losses * weights.float()).sum(), weights.float().sum()
    return AdditiveScalar(numerator, denominator, additive_mean(numerator, denominator))
