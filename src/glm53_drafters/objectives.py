from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .chunked_lm_head import AdditiveScalar, additive_mean


@dataclass(frozen=True)
class DistributionDistance(AdditiveScalar):
    per_token: torch.Tensor


def depth_weights(
    *, block_size: int, gamma: float, device: torch.device | None = None
) -> torch.Tensor:
    if block_size < 2 or gamma <= 0:
        raise ValueError("block_size must exceed one and gamma must be positive")
    # The supervised successor at physical k=1 has official depth index zero.
    depths = torch.arange(0, block_size - 1, dtype=torch.float32, device=device)
    return torch.exp(-depths / float(gamma))


def selector_cross_entropy(
    candidate_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> AdditiveScalar:
    if candidate_ids.shape != candidate_logits.shape:
        raise ValueError("candidate IDs and logits must have equal shape")
    if candidate_ids.shape[:-1] != targets.shape:
        raise ValueError("selector targets must match candidate token dimensions")
    matches = candidate_ids.eq(targets.unsqueeze(-1))
    hits = matches.any(dim=-1)
    scale = (
        torch.ones_like(targets, dtype=torch.float32)
        if weights is None
        else weights.to(device=targets.device, dtype=torch.float32)
    )
    # Official DFlash2 trains the selector only when the base top-k contains
    # the target.  A miss stays a miss and contributes neither loss nor
    # denominator; this keeps training and inference candidate sets identical.
    selector_scale = scale * hits.to(dtype=torch.float32)
    local_targets = matches.to(torch.int64).argmax(dim=-1)
    losses = F.cross_entropy(
        candidate_logits.reshape(-1, candidate_logits.shape[-1]).float(),
        local_targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    numerator = (losses * selector_scale).sum(dtype=torch.float32)
    denominator = selector_scale.sum(dtype=torch.float32)
    return AdditiveScalar(numerator, denominator, additive_mean(numerator, denominator))


def exact_total_variation(
    student_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> DistributionDistance:
    if student_logits.shape != target_logits.shape:
        raise ValueError("student and target logits must have equal shape")
    student = student_logits.float().softmax(dim=-1)
    target = target_logits.float().softmax(dim=-1)
    per_token = 0.5 * (student - target).abs().sum(dim=-1)
    scale = (
        torch.ones_like(per_token)
        if weights is None
        else weights.to(device=per_token.device, dtype=torch.float32)
    )
    numerator = (per_token * scale).sum(dtype=torch.float32)
    denominator = scale.sum(dtype=torch.float32)
    return DistributionDistance(
        numerator=numerator,
        denominator=denominator,
        mean=additive_mean(numerator, denominator),
        per_token=per_token,
    )


def confidence_bce(
    confidence_logits: torch.Tensor,
    total_variation: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> AdditiveScalar:
    if confidence_logits.shape != total_variation.shape:
        raise ValueError("confidence and TV tensors must have equal shape")
    target = (1.0 - total_variation.detach().float()).clamp(0.0, 1.0)
    losses = F.binary_cross_entropy_with_logits(
        confidence_logits.float(), target, reduction="none"
    )
    scale = (
        torch.ones_like(losses)
        if weights is None
        else weights.to(device=losses.device, dtype=torch.float32)
    )
    numerator = (losses * scale).sum(dtype=torch.float32)
    denominator = scale.sum(dtype=torch.float32)
    return AdditiveScalar(numerator, denominator, additive_mean(numerator, denominator))
