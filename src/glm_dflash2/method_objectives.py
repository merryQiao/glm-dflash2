from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .dspark_model import LowRankMarkovHead


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


@dataclass(frozen=True)
class DSparkLoss:
    local_total: torch.Tensor
    ce: AdditiveLoss
    l1: AdditiveLoss
    confidence: AdditiveLoss
    l1_per_token: torch.Tensor
    confidence_target: torch.Tensor
    draft_top1_ids: torch.Tensor


def _bf16_mm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("DSpark frozen LM-head matmul requires BF16 inputs")
    return torch.mm(hidden, weight.T)


def reconstruct_target_logits(
    target_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    vocab_chunk_size: int,
) -> torch.Tensor:
    """Dense validation helper using the same BF16 chunked reconstruction."""

    if target_hidden.ndim != 2 or lm_head_weight.ndim != 2:
        raise ValueError("target hidden and LM head must both be matrices")
    if target_hidden.shape[1] != lm_head_weight.shape[1]:
        raise ValueError("target hidden and LM-head widths differ")
    if vocab_chunk_size < 1:
        raise ValueError("vocab_chunk_size must be positive")
    chunks = [
        _bf16_mm(target_hidden, lm_head_weight[start:end]).float()
        for start in range(0, lm_head_weight.shape[0], vocab_chunk_size)
        for end in (min(lm_head_weight.shape[0], start + vocab_chunk_size),)
    ]
    return torch.cat(chunks, dim=-1)


def compute_dspark_loss(
    *,
    draft_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    predecessor_ids: torch.Tensor,
    confidence_logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
    markov_head: LowRankMarkovHead,
    token_mask: torch.Tensor,
    gamma: float,
    vocab_chunk_size: int,
    ce_weight: float = 0.1,
    l1_weight: float = 0.9,
    confidence_weight: float = 1.0,
) -> DSparkLoss:
    """Exact full-vocabulary DSpark objective with chunk-bounded activations."""

    prefix = draft_hidden.shape[:-1]
    if (
        target_hidden.shape != draft_hidden.shape
        or target_ids.shape != prefix
        or predecessor_ids.shape != prefix
        or confidence_logits.shape != prefix
        or token_mask.shape != prefix
    ):
        raise ValueError("DSpark token tensors have incompatible shapes")
    if draft_hidden.shape[-1] != lm_head_weight.shape[-1]:
        raise ValueError("draft hidden and LM-head widths differ")
    if vocab_chunk_size < 1:
        raise ValueError("vocab_chunk_size must be positive")
    vocabulary = int(lm_head_weight.shape[0])
    if vocabulary < 1 or bool(((target_ids < 0) | (target_ids >= vocabulary)).any()):
        raise ValueError("target_ids contain an out-of-vocabulary ID")
    if (
        draft_hidden.dtype != torch.bfloat16
        or target_hidden.dtype != torch.bfloat16
        or lm_head_weight.dtype != torch.bfloat16
    ):
        raise ValueError("DSpark draft, target, and LM-head tensors must be BF16")

    flat_draft = draft_hidden.reshape(-1, draft_hidden.shape[-1])
    flat_target = target_hidden.detach().reshape_as(flat_draft)
    flat_ids = target_ids.reshape(-1).to(torch.long)
    flat_predecessors = predecessor_ids.reshape(-1).to(torch.long)

    def draft_chunk(
        hidden: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        logits = _bf16_mm(hidden, lm_head_weight[start:end]).float()
        bias = markov_head.score_chunk(hidden, flat_predecessors, start, end)
        return logits + bias.float()

    def teacher_chunk(start: int, end: int) -> torch.Tensor:
        with torch.no_grad():
            return _bf16_mm(flat_target, lm_head_weight[start:end]).float()

    draft_log_z: torch.Tensor | None = None
    target_log_z: torch.Tensor | None = None
    draft_target_score = torch.zeros(
        flat_ids.shape, device=flat_draft.device, dtype=torch.float32
    )
    for start in range(0, vocabulary, vocab_chunk_size):
        end = min(vocabulary, start + vocab_chunk_size)

        def draft_stats(
            hidden: torch.Tensor, chunk_start: int = start, chunk_end: int = end
        ) -> tuple[torch.Tensor, torch.Tensor]:
            logits = draft_chunk(hidden, chunk_start, chunk_end)
            local = (flat_ids - chunk_start).clamp(0, chunk_end - chunk_start - 1)
            gathered = logits.gather(1, local[:, None]).squeeze(1)
            active = (flat_ids >= chunk_start) & (flat_ids < chunk_end)
            return torch.logsumexp(logits, dim=-1), torch.where(
                active, gathered, torch.zeros_like(gathered)
            )

        chunk_lse, chunk_target = checkpoint(
            draft_stats, flat_draft, use_reentrant=False
        )
        draft_log_z = (
            chunk_lse
            if draft_log_z is None
            else torch.logaddexp(draft_log_z, chunk_lse)
        )
        draft_target_score = draft_target_score + chunk_target
        teacher_lse = torch.logsumexp(teacher_chunk(start, end), dim=-1)
        target_log_z = (
            teacher_lse
            if target_log_z is None
            else torch.logaddexp(target_log_z, teacher_lse)
        )
    if draft_log_z is None or target_log_z is None:  # pragma: no cover
        raise RuntimeError("empty vocabulary")

    l1_flat = torch.zeros_like(draft_log_z)
    for start in range(0, vocabulary, vocab_chunk_size):
        end = min(vocabulary, start + vocab_chunk_size)
        target_probability = torch.exp(teacher_chunk(start, end) - target_log_z[:, None])

        def l1_chunk(
            hidden: torch.Tensor,
            log_z: torch.Tensor,
            chunk_start: int = start,
            chunk_end: int = end,
            target_p: torch.Tensor = target_probability,
        ) -> torch.Tensor:
            draft_probability = torch.exp(
                draft_chunk(hidden, chunk_start, chunk_end) - log_z[:, None]
            )
            return (draft_probability - target_p).abs().sum(dim=-1)

        l1_flat = l1_flat + checkpoint(
            l1_chunk, flat_draft, draft_log_z, use_reentrant=False
        )

    ce_per_token = (draft_log_z - draft_target_score).reshape(prefix)
    l1_per_token = l1_flat.reshape(prefix)
    confidence_target = (1.0 - 0.5 * l1_per_token.detach()).clamp(0.0, 1.0)
    confidence_per_token = F.binary_cross_entropy_with_logits(
        confidence_logits.float(), confidence_target, reduction="none"
    )
    ce = depth_weighted_objective(ce_per_token, token_mask, gamma=gamma)
    l1 = depth_weighted_objective(l1_per_token, token_mask, gamma=gamma)
    confidence = depth_weighted_objective(
        confidence_per_token, token_mask, gamma=gamma
    )

    with torch.no_grad():
        best_scores = torch.full_like(draft_log_z, float("-inf"))
        best_ids = torch.zeros_like(flat_ids)
        for start in range(0, vocabulary, vocab_chunk_size):
            end = min(vocabulary, start + vocab_chunk_size)
            scores, ids = draft_chunk(flat_draft, start, end).max(dim=-1)
            replace = scores > best_scores
            best_scores = torch.where(replace, scores, best_scores)
            best_ids = torch.where(replace, ids + start, best_ids)

    return DSparkLoss(
        local_total=(
            float(ce_weight) * ce.mean
            + float(l1_weight) * l1.mean
            + float(confidence_weight) * confidence.mean
        ),
        ce=ce,
        l1=l1,
        confidence=confidence,
        l1_per_token=l1_per_token,
        confidence_target=confidence_target,
        draft_top1_ids=best_ids.reshape(prefix),
    )
