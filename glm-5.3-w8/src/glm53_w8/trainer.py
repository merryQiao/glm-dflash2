from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import build_sliding_blocks, sample_anchor_positions
from .contracts import DFLASH2_METHOD, DSPARK_METHOD, validate_method_contract
from .dflash2 import DFlash2Model
from .dspark import DSparkModel, LowRankMarkovHead
from .target_io import FrozenTargetIO


@dataclass(frozen=True)
class TrainingRecipe:
    method: str
    block_size: int
    gamma: float
    learning_rate: float = 6e-4
    epochs: int = 3
    anchors_per_sample: int = 512
    warmup_steps: int = 1000
    gradient_accumulation: int = 8
    ce_weight: float = 0.0
    tv_weight: float = 0.0
    confidence_weight: float = 0.0


def recipe_for(method: str, block_size: int) -> TrainingRecipe:
    validate_method_contract(method, block_size=block_size)
    if method == DSPARK_METHOD:
        return TrainingRecipe(
            method=method,
            block_size=8,
            gamma=4.0,
            ce_weight=0.1,
            tv_weight=0.9,
            confidence_weight=1.0,
        )
    return TrainingRecipe(
        method=DFLASH2_METHOD,
        block_size=int(block_size),
        gamma=4.0 if int(block_size) == 8 else 7.0,
    )


def _depth_weights(reference: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    values = torch.exp(
        -torch.arange(reference.shape[-1], device=reference.device, dtype=torch.float32)
        / float(gamma)
    )
    return values.reshape((1,) * (reference.ndim - 1) + (-1,))


def _weighted_mean(
    values: torch.Tensor, mask: torch.Tensor, gamma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = _depth_weights(values, gamma) * mask.to(torch.float32)
    numerator = (values.float() * weights).sum()
    denominator = weights.sum()
    return numerator / denominator.clamp_min(1.0), numerator, denominator


def rank_loss_scale(*, include: bool, world_size: int) -> float:
    """Scale a per-rank loss before FSDP's world-size mean reduction."""

    world_size = int(world_size)
    if world_size < 1:
        raise ValueError("world_size must be positive")
    return float(world_size) if bool(include) else 0.0


def accumulation_real_count(real_rank_counts: list[int] | tuple[int, ...]) -> int:
    """Return the real-sample denominator for an accumulation interval."""

    count = sum(int(value) for value in real_rank_counts)
    if count < 1:
        raise ValueError("an accumulation interval needs at least one real sample")
    return count


@dataclass(frozen=True)
class ChunkedProjection:
    nll: torch.Tensor
    topk_scores: torch.Tensor
    topk_ids: torch.Tensor


def chunked_lm_projection(
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    weight: torch.Tensor,
    *,
    top_k: int,
    vocab_chunk_size: int,
) -> ChunkedProjection:
    if target_ids.shape != hidden.shape[:-1] or weight.shape[1] != hidden.shape[-1]:
        raise ValueError("LM projection shapes are incompatible")
    if int(top_k) < 1 or int(vocab_chunk_size) < 1:
        raise ValueError("top_k and vocab_chunk_size must be positive")
    if int(top_k) > int(weight.shape[0]):
        raise ValueError("top_k cannot exceed vocabulary size")
    if bool(((target_ids < 0) | (target_ids >= weight.shape[0])).any()):
        raise ValueError("target token IDs are outside the LM vocabulary")
    flat = hidden.reshape(-1, hidden.shape[-1])
    targets = target_ids.reshape(-1).to(torch.long)
    log_z: torch.Tensor | None = None
    target_score = torch.zeros_like(targets, dtype=torch.float32)
    running_scores = torch.empty((flat.shape[0], 0), device=flat.device)
    running_ids = torch.empty((flat.shape[0], 0), device=flat.device, dtype=torch.long)
    vocabulary = int(weight.shape[0])
    for start in range(0, vocabulary, vocab_chunk_size):
        end = min(vocabulary, start + vocab_chunk_size)
        logits = F.linear(flat.to(weight.dtype), weight[start:end]).float()
        lse = torch.logsumexp(logits, dim=-1)
        log_z = lse if log_z is None else torch.logaddexp(log_z, lse)
        active = (targets >= start) & (targets < end)
        local = (targets - start).clamp(0, end - start - 1)
        gathered = logits.gather(1, local[:, None]).squeeze(1)
        target_score = target_score + torch.where(active, gathered, torch.zeros_like(gathered))
        local_k = min(top_k, end - start)
        scores, ids = logits.topk(local_k, dim=-1)
        ids = ids + start
        merged_scores = torch.cat((running_scores, scores), dim=-1)
        merged_ids = torch.cat((running_ids, ids), dim=-1)
        keep = min(top_k, merged_scores.shape[-1])
        running_scores, slots = merged_scores.topk(keep, dim=-1)
        running_ids = merged_ids.gather(1, slots)
    if log_z is None:
        raise ValueError("empty vocabulary")
    prefix = target_ids.shape
    return ChunkedProjection(
        nll=(log_z - target_score).reshape(prefix),
        topk_scores=running_scores.reshape(prefix + (top_k,)),
        topk_ids=running_ids.reshape(prefix + (top_k,)),
    )


@dataclass(frozen=True)
class DFlash2Objective:
    total: torch.Tensor
    base: torch.Tensor
    selector: torch.Tensor
    candidate_hits: torch.Tensor
    candidate_total: torch.Tensor


def compute_dflash2_objective(
    *,
    base_nll: torch.Tensor,
    candidate_ids: torch.Tensor,
    selector_scores: torch.Tensor,
    target_ids: torch.Tensor,
    prediction_mask: torch.Tensor,
    gamma: float,
) -> DFlash2Objective:
    successors = target_ids[..., 1:]
    if (
        successors.shape != base_nll.shape
        or candidate_ids.shape[:-1] != successors.shape
        or selector_scores.shape != candidate_ids.shape
        or prediction_mask.shape != successors.shape
    ):
        raise ValueError("DFlash2 objective shapes differ")
    base, base_numerator, denominator = _weighted_mean(base_nll, prediction_mask, gamma)
    matches = candidate_ids.eq(successors[..., None])
    hits = matches.any(-1) & prediction_mask
    slots = matches.to(torch.int64).argmax(-1)
    depth = _depth_weights(base_nll, gamma)
    if bool(hits.any()):
        selector_nll = F.cross_entropy(
            selector_scores.float()[hits], slots[hits], reduction="none"
        )
        selector_numerator = (selector_nll * depth.expand_as(base_nll)[hits]).sum()
        selector = selector_numerator / (
            depth.expand_as(base_nll)[hits].sum().clamp_min(1e-6)
        )
    else:
        selector_numerator = selector_scores[hits].sum()
        selector = selector_numerator
    # A missing target cannot be selected at runtime, so official DFlash2
    # excludes those positions from the selector term and reports hit recall.
    total = base + selector
    return DFlash2Objective(
        total=total,
        base=base,
        selector=selector,
        candidate_hits=hits.sum().detach(),
        candidate_total=prediction_mask.sum().detach(),
    )


@dataclass(frozen=True)
class DSparkObjective:
    total: torch.Tensor
    ce: torch.Tensor
    tv: torch.Tensor
    confidence: torch.Tensor
    confidence_target: torch.Tensor
    top1_ids: torch.Tensor


def compute_dspark_objective(
    *,
    draft_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    predecessor_ids: torch.Tensor,
    confidence_logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    prediction_mask: torch.Tensor,
    gamma: float,
    vocab_chunk_size: int,
    ce_weight: float = 0.1,
    tv_weight: float = 0.9,
    confidence_weight: float = 1.0,
    markov_head: LowRankMarkovHead | None = None,
) -> DSparkObjective:
    prefix = target_ids.shape
    if (
        draft_hidden.shape[:-1] != prefix
        or target_hidden.shape != draft_hidden.shape
        or predecessor_ids.shape != prefix
        or confidence_logits.shape != prefix
        or prediction_mask.shape != prefix
    ):
        raise ValueError("DSpark hidden/target shapes differ")
    if int(vocab_chunk_size) < 1:
        raise ValueError("vocab_chunk_size must be positive")
    if float(ce_weight) < 0 or float(tv_weight) < 0 or float(confidence_weight) < 0:
        raise ValueError("DSpark loss weights must be non-negative")
    vocabulary = int(lm_head_weight.shape[0])
    if lm_head_weight.ndim != 2 or lm_head_weight.shape[1] != draft_hidden.shape[-1]:
        raise ValueError("DSpark LM head shape is incompatible")
    if bool(((target_ids < 0) | (target_ids >= vocabulary)).any()) or bool(
        ((predecessor_ids < 0) | (predecessor_ids >= markov_w1.shape[0])).any()
    ):
        raise ValueError("DSpark token IDs are outside the model vocabulary")
    flat_draft = draft_hidden.reshape(-1, draft_hidden.shape[-1])
    flat_teacher = target_hidden.detach().reshape_as(flat_draft)
    flat_ids = target_ids.reshape(-1).to(torch.long)
    flat_predecessors = predecessor_ids.reshape(-1)
    predecessor_features = (
        markov_head(flat_predecessors)
        if markov_head is not None
        else F.embedding(flat_predecessors, markov_w1)
    )
    draft_log_z: torch.Tensor | None = None
    teacher_log_z: torch.Tensor | None = None
    target_score = torch.zeros_like(flat_ids, dtype=torch.float32)
    best_score = torch.full_like(target_score, float("-inf"))
    best_id = torch.zeros_like(flat_ids)
    def logits_for(start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        draft_logits = F.linear(
            flat_draft.to(lm_head_weight.dtype), lm_head_weight[start:end]
        ).float()
        markov_scores = (
            markov_head(flat_predecessors, start, end)
            if markov_head is not None
            else torch.einsum(
                "nr,vr->nv", predecessor_features.float(), markov_w2[start:end].float()
            )
        )
        draft_logits = draft_logits + markov_scores.float()
        with torch.no_grad():
            teacher_logits = F.linear(
                flat_teacher.to(lm_head_weight.dtype), lm_head_weight[start:end]
            ).float()
        return draft_logits, teacher_logits

    for start in range(0, vocabulary, vocab_chunk_size):
        end = min(vocabulary, start + vocab_chunk_size)
        draft_logits, teacher_logits = logits_for(start, end)
        draft_lse = torch.logsumexp(draft_logits, -1)
        teacher_lse = torch.logsumexp(teacher_logits, -1)
        draft_log_z = draft_lse if draft_log_z is None else torch.logaddexp(draft_log_z, draft_lse)
        teacher_log_z = teacher_lse if teacher_log_z is None else torch.logaddexp(teacher_log_z, teacher_lse)
        active = (flat_ids >= start) & (flat_ids < end)
        local = (flat_ids - start).clamp(0, end - start - 1)
        gathered = draft_logits.gather(1, local[:, None]).squeeze(1)
        target_score = target_score + torch.where(active, gathered, torch.zeros_like(gathered))
        score, token = draft_logits.max(-1)
        replace = score > best_score
        best_score = torch.where(replace, score, best_score)
        best_id = torch.where(replace, token + start, best_id)
    if draft_log_z is None or teacher_log_z is None:
        raise ValueError("empty vocabulary")
    l1 = torch.zeros_like(draft_log_z)
    for start in range(0, vocabulary, vocab_chunk_size):
        end = min(vocabulary, start + vocab_chunk_size)
        draft_logits, teacher_logits = logits_for(start, end)
        draft_probability = torch.exp(draft_logits - draft_log_z[:, None])
        teacher_probability = torch.exp(teacher_logits - teacher_log_z[:, None])
        l1 = l1 + (draft_probability - teacher_probability).abs().sum(-1)
    ce_tokens = (draft_log_z - target_score).reshape(prefix)
    tv_tokens = (0.5 * l1).reshape(prefix)
    confidence_target = (1.0 - tv_tokens.detach()).clamp(0.0, 1.0)
    confidence_tokens = F.binary_cross_entropy_with_logits(
        confidence_logits.float(), confidence_target, reduction="none"
    )
    ce, _, _ = _weighted_mean(ce_tokens, prediction_mask, gamma)
    tv, _, _ = _weighted_mean(tv_tokens, prediction_mask, gamma)
    confidence, _, _ = _weighted_mean(confidence_tokens, prediction_mask, gamma)
    return DSparkObjective(
        total=float(ce_weight) * ce + float(tv_weight) * tv + float(confidence_weight) * confidence,
        ce=ce,
        tv=tv,
        confidence=confidence,
        confidence_target=confidence_target,
        top1_ids=best_id.reshape(prefix),
    )


@dataclass(frozen=True)
class OfflineStepOutput:
    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor]


class OfflineDrafterTrainer(nn.Module):
    def __init__(
        self,
        model: DFlash2Model | DSparkModel,
        target_io: FrozenTargetIO,
        *,
        method: str,
        gamma: float,
        anchors_per_sample: int = 512,
        global_seed: int = 42,
        vocab_chunk_size: int = 8192,
    ) -> None:
        super().__init__()
        validate_method_contract(method, block_size=model.config.block_size)
        if target_io.embed_tokens.weight.requires_grad or target_io.lm_head.weight.requires_grad:
            raise ValueError("target embedding/lm_head must be frozen")
        self.model = model
        object.__setattr__(self, "_target_io", target_io)
        self.method = method
        self.gamma = float(gamma)
        self.anchors_per_sample = int(anchors_per_sample)
        self.global_seed = int(global_seed)
        self.vocab_chunk_size = int(vocab_chunk_size)

    def _anchors(
        self,
        batch: Mapping[str, Any],
        epoch: int,
        anchor_positions: torch.Tensor | None,
        block_keep_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor_positions is not None and block_keep_mask is not None:
            return anchor_positions, block_keep_mask
        if anchor_positions is not None or block_keep_mask is not None:
            raise ValueError("anchor positions and keep mask must be supplied together")
        sample_ids = batch.get("sample_ids") or batch.get("sample_id")
        if not isinstance(sample_ids, (list, tuple)):
            raise ValueError("automatic anchor sampling requires sample_ids")
        return sample_anchor_positions(
            batch.get("anchor_mask", batch["loss_mask"]),
            sample_ids=sample_ids,
            global_seed=self.global_seed,
            epoch=epoch,
            attention_mask=batch.get("attention_mask"),
            block_size=self.model.config.block_size,
            num_anchors=self.anchors_per_sample,
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        epoch: int,
        anchor_positions: torch.Tensor | None = None,
        block_keep_mask: torch.Tensor | None = None,
    ) -> OfflineStepOutput:
        ids = batch["input_ids"]
        anchors, keep = self._anchors(
            batch, epoch, anchor_positions, block_keep_mask
        )
        blocks = build_sliding_blocks(
            ids,
            batch["loss_mask"],
            anchors,
            keep,
            block_size=self.model.config.block_size,
            mask_token_id=self.model.config.mask_token_id,
            sliding_window=self.model.config.sliding_window,
            attention_mask=batch.get("attention_mask"),
            position_offset=batch.get("position_offset"),
        )
        noise = self._target_io.embed_tokens(blocks.noise_ids)
        hidden = self.model(
            noise_embedding=noise,
            target_hidden=batch["aux_hidden_states"].to(noise.dtype),
            blocks=blocks,
        )
        predictions = hidden[..., 1:, :]
        targets = blocks.target_ids[..., 1:]
        if self.method == DFLASH2_METHOD:
            projection = chunked_lm_projection(
                predictions,
                targets,
                self._target_io.lm_head.weight,
                top_k=self.model.config.selector_top_k,
                vocab_chunk_size=self.vocab_chunk_size,
            )
            selector_scores = self.model.selector_scores(
                predictions,
                projection.topk_scores,
                projection.topk_ids,
                blocks.target_ids[..., :-1],
            )
            terms = compute_dflash2_objective(
                base_nll=projection.nll,
                candidate_ids=projection.topk_ids,
                selector_scores=selector_scores,
                target_ids=blocks.target_ids,
                prediction_mask=blocks.prediction_mask,
                gamma=self.gamma,
            )
            selected = projection.topk_ids.gather(
                -1, selector_scores.argmax(-1, keepdim=True)
            ).squeeze(-1)
            accuracy = ((selected == targets) & blocks.prediction_mask).sum().float() / blocks.prediction_mask.sum().clamp_min(1)
            return OfflineStepOutput(
                terms.total,
                {
                    "base_loss": terms.base.detach(),
                    "selector_loss": terms.selector.detach(),
                    "candidate_recall": terms.candidate_hits.float() / terms.candidate_total.clamp_min(1),
                    "accuracy": accuracy.detach(),
                },
            )
        if not isinstance(self.model, DSparkModel):
            raise TypeError("DSpark method requires DSparkModel")
        depth = predictions.shape[-2]
        positions = anchors[..., None] + torch.arange(depth, device=ids.device)
        # Tail positions are intentionally represented by padded target IDs and
        # removed by ``prediction_mask``.  Clamp only the teacher/predecessor
        # lookup so a short final block cannot issue an out-of-range gather.
        safe_positions = positions.clamp(max=ids.shape[1] - 1)
        batch_size, _, width = batch["target_final_hidden"].shape
        expanded = batch["target_final_hidden"][:, None].expand(
            batch_size, anchors.shape[1], ids.shape[1], width
        )
        teacher = expanded.gather(
            2, safe_positions[..., None].expand(*safe_positions.shape, width)
        ).to(predictions.dtype)
        expanded_ids = ids[:, None].expand(batch_size, anchors.shape[1], ids.shape[1])
        predecessors = expanded_ids.gather(2, safe_positions)
        confidence = self.model.confidence_logits(predictions, predecessors)
        terms = compute_dspark_objective(
            draft_hidden=predictions,
            target_hidden=teacher,
            target_ids=targets,
            predecessor_ids=predecessors,
            confidence_logits=confidence,
            lm_head_weight=self._target_io.lm_head.weight,
            markov_w1=self.model.markov_head.w1.weight,
            markov_w2=self.model.markov_head.w2.weight,
            prediction_mask=blocks.prediction_mask,
            gamma=self.gamma,
            vocab_chunk_size=self.vocab_chunk_size,
            markov_head=self.model.markov_head,
        )
        accuracy = ((terms.top1_ids == targets) & blocks.prediction_mask).sum().float() / blocks.prediction_mask.sum().clamp_min(1)
        return OfflineStepOutput(
            terms.total,
            {
                "ce_loss": terms.ce.detach(),
                "tv_loss": terms.tv.detach(),
                "confidence_loss": terms.confidence.detach(),
                "accuracy": accuracy.detach(),
            },
        )
