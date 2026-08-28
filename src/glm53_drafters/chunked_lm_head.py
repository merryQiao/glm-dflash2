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


def chunked_cross_entropy(
    hidden: torch.Tensor,
    lm_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    vocab_chunk_size: int = 8192,
) -> AdditiveScalar:
    """Exact CE using streaming log-sum-exp over vocabulary chunks."""

    if hidden.shape[:-1] != targets.shape:
        raise ValueError("targets must match hidden token dimensions")
    if lm_weight.ndim != 2 or lm_weight.shape[1] != hidden.shape[-1]:
        raise ValueError("LM head shape differs from hidden width")
    if vocab_chunk_size < 1:
        raise ValueError("vocab_chunk_size must be positive")
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1).long()
    vocabulary = lm_weight.shape[0]
    if flat_targets.numel() and (
        int(flat_targets.min()) < 0 or int(flat_targets.max()) >= vocabulary
    ):
        raise ValueError("target token is outside vocabulary")
    running_lse: torch.Tensor | None = None
    target_logit = flat_hidden.float().sum(dim=-1) * 0.0
    for start in range(0, vocabulary, vocab_chunk_size):
        stop = min(start + vocab_chunk_size, vocabulary)
        # Keep the production projection in BF16; only the normalization and
        # additive reductions are promoted to FP32.
        logits = F.linear(flat_hidden, lm_weight[start:stop]).float()
        chunk_lse = torch.logsumexp(logits, dim=-1)
        running_lse = (
            chunk_lse
            if running_lse is None
            else torch.logaddexp(running_lse, chunk_lse)
        )
        in_chunk = (flat_targets >= start) & (flat_targets < stop)
        local = (flat_targets - start).clamp(0, stop - start - 1)
        selected = logits.gather(-1, local.unsqueeze(-1)).squeeze(-1)
        target_logit = target_logit + torch.where(
            in_chunk, selected, torch.zeros_like(selected)
        )
    if running_lse is None:
        raise ValueError("LM head vocabulary cannot be empty")
    per_token = running_lse - target_logit
    flat_weights = (
        torch.ones_like(per_token)
        if weights is None
        else weights.reshape(-1).to(device=per_token.device, dtype=torch.float32)
    )
    if flat_weights.shape != per_token.shape:
        raise ValueError("weights must match targets")
    numerator = (per_token * flat_weights).sum(dtype=torch.float32)
    denominator = flat_weights.sum(dtype=torch.float32)
    return AdditiveScalar(numerator, denominator, additive_mean(numerator, denominator))
