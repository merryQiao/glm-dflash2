from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DFlash2Loss:
    loss: torch.Tensor
    base_loss: torch.Tensor
    selector_loss: torch.Tensor
    base_numerator: torch.Tensor
    base_denominator: torch.Tensor
    selector_numerator: torch.Tensor
    selector_denominator: torch.Tensor
    candidate_hits: torch.Tensor
    candidate_total: torch.Tensor


def selector_supervision(
    candidate_ids: torch.Tensor, target_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top-k slot of each ground-truth successor and its hit mask."""

    if candidate_ids.ndim != 4:
        raise ValueError("candidate_ids must have shape [batch, anchors, depth, top_k]")
    if target_ids.shape != candidate_ids.shape[:-2] + (candidate_ids.shape[-2] + 1,):
        raise ValueError("target_ids must contain one predecessor plus every successor")
    successors = target_ids[..., 1:]
    matches = candidate_ids.eq(successors[..., None])
    hit = matches.any(dim=-1)
    slots = matches.to(torch.int64).argmax(dim=-1)
    return slots, hit


def _depth_weights(reference: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    depth = torch.arange(reference.shape[-1], device=reference.device, dtype=torch.float32)
    shape = (1,) * (reference.ndim - 1) + (reference.shape[-1],)
    return torch.exp(-depth / float(gamma)).reshape(shape)


def compute_dflash2_loss(
    *,
    base_nll: torch.Tensor,
    candidate_ids: torch.Tensor,
    selector_scores: torch.Tensor,
    target_ids: torch.Tensor,
    pred_mask: torch.Tensor,
    gamma: float,
    selector_loss_weight: float = 1.0,
) -> DFlash2Loss:
    """Compute the official weighted base CE plus candidate selector CE."""

    if base_nll.shape != pred_mask.shape:
        raise ValueError("base_nll and pred_mask shapes differ")
    if candidate_ids.shape[:-1] != base_nll.shape:
        raise ValueError("candidate_ids prefix shape differs from base_nll")
    if selector_scores.shape != candidate_ids.shape:
        raise ValueError("selector_scores and candidate_ids shapes differ")

    mask = pred_mask.to(device=base_nll.device, dtype=torch.float32)
    weights = mask * _depth_weights(base_nll, gamma)
    base_numerator = (base_nll.float() * weights).sum()
    base_denominator = weights.sum()
    if bool(base_denominator > 0):
        base_loss = base_numerator / base_denominator
    else:
        base_loss = base_nll.sum() * 0.0

    slots, hit = selector_supervision(candidate_ids, target_ids)
    selector_weights = weights * hit.to(weights.dtype)
    active = selector_weights > 0
    if bool(active.any()):
        active_nll = F.cross_entropy(
            selector_scores.float()[active],
            slots[active],
            reduction="none",
        )
        selector_numerator = (active_nll * selector_weights[active]).sum()
    else:
        # Indexing first is intentional: padded rows contain all -inf logits,
        # and (-inf * 0) would otherwise make a nominally masked loss NaN.
        selector_numerator = selector_scores.float()[active].sum()
    selector_denominator = selector_weights.sum()
    # Keep a differentiable zero when a microbatch has no target in top-k.
    selector_loss = selector_numerator / (selector_denominator + 1e-6)
    total = base_loss + float(selector_loss_weight) * selector_loss
    valid = mask > 0
    return DFlash2Loss(
        loss=total,
        base_loss=base_loss,
        selector_loss=selector_loss,
        base_numerator=base_numerator.detach(),
        base_denominator=base_denominator.detach(),
        selector_numerator=selector_numerator.detach(),
        selector_denominator=selector_denominator.detach(),
        candidate_hits=(hit & valid).sum().detach(),
        candidate_total=valid.sum().detach(),
    )


def compute_acceptance_stats(
    predicted_ids: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bonus-inclusive greedy acceptance over valid positions in each block."""

    if predicted_ids.shape != target_ids.shape or valid_mask.shape != predicted_ids.shape:
        raise ValueError("prediction, target, and mask shapes must match")
    flat_pred = predicted_ids.reshape(-1, predicted_ids.shape[-1])
    flat_target = target_ids.reshape_as(flat_pred)
    flat_mask = valid_mask.to(torch.bool).reshape_as(flat_pred)
    total = torch.zeros((), device=predicted_ids.device, dtype=torch.float32)
    blocks = torch.zeros((), device=predicted_ids.device, dtype=torch.int64)
    for pred, target, mask in zip(flat_pred, flat_target, flat_mask):
        if not bool(mask.any()):
            continue
        blocks += 1
        accepted = 0
        for is_correct in pred[mask].eq(target[mask]):
            if not bool(is_correct):
                break
            accepted += 1
        total += float(accepted + 1)
    mean = total / blocks.clamp_min(1).to(total.dtype)
    return mean, total, blocks
