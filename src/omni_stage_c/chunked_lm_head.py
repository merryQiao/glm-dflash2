from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AdditiveScalar:
    numerator: torch.Tensor
    denominator: torch.Tensor
    mean: torch.Tensor


def additive_mean(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    safe = numerator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
    return torch.where(denominator > 0, safe, numerator * 0.0)


def chunked_cross_entropy(hidden: torch.Tensor, lm_weight: torch.Tensor,
                          targets: torch.Tensor, *, weights: torch.Tensor,
                          vocab_chunk_size: int = 8192) -> AdditiveScalar:
    if hidden.shape[:-1] != targets.shape or weights.shape != targets.shape:
        raise ValueError("hidden, targets, and weights are not token-aligned")
    flat_hidden, flat_targets = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1).long()
    running_lse = None
    target_logit = flat_hidden.float().sum(-1) * 0.0
    for start in range(0, lm_weight.shape[0], vocab_chunk_size):
        stop = min(start + vocab_chunk_size, lm_weight.shape[0])
        logits = F.linear(flat_hidden, lm_weight[start:stop]).float()
        chunk_lse = torch.logsumexp(logits, -1)
        running_lse = chunk_lse if running_lse is None else torch.logaddexp(running_lse, chunk_lse)
        match = (flat_targets >= start) & (flat_targets < stop)
        local = (flat_targets - start).clamp(0, stop - start - 1)
        selected = logits.gather(-1, local[:, None]).squeeze(-1)
        target_logit += torch.where(match, selected, torch.zeros_like(selected))
    if running_lse is None:
        raise ValueError("empty vocabulary")
    scale = weights.reshape(-1).float()
    numerator = ((running_lse - target_logit) * scale).sum()
    denominator = scale.sum()
    return AdditiveScalar(numerator, denominator, additive_mean(numerator, denominator))
