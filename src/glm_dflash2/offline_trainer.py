from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .chunked_lm_head import ChunkedLmProjection, chunked_lm_projection
from .dflash2_model import Qwen3DFlash2DraftModel
from .dflash2_objective import compute_acceptance_stats, compute_dflash2_loss
from .dflash_blocks import DFlashBlocks, build_dflash_blocks, sample_anchor_positions
from .distributed import global_weighted_mean
from .draft_backbone import DFlashDraftModel
from .dspark_model import DSparkDraftModel
from .method_objectives import compute_dspark_loss, depth_weighted_objective
from .target_io import FrozenTargetIO, validate_cache_io_compatibility


@dataclass(frozen=True)
class OfflineStepOutput:
    loss: torch.Tensor
    base_loss: torch.Tensor
    selector_loss: torch.Tensor
    base_accuracy: torch.Tensor
    selector_accuracy: torch.Tensor
    base_accept_len: torch.Tensor
    selector_accept_len: torch.Tensor
    candidate_recall: torch.Tensor
    valid_tokens: torch.Tensor
    valid_blocks: torch.Tensor
    loss_weight: torch.Tensor
    base_numerator: torch.Tensor
    base_denominator: torch.Tensor
    selector_numerator: torch.Tensor
    selector_denominator: torch.Tensor
    base_correct: torch.Tensor
    selector_correct: torch.Tensor
    base_accept_total: torch.Tensor
    selector_accept_total: torch.Tensor
    candidate_hits: torch.Tensor
    candidate_total: torch.Tensor
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor


@dataclass(frozen=True)
class _PreparedStep:
    blocks: DFlashBlocks
    pred_hidden: torch.Tensor
    pred_targets: torch.Tensor
    pred_mask: torch.Tensor
    projection: ChunkedLmProjection


@dataclass(frozen=True)
class _PreparedHidden:
    blocks: DFlashBlocks
    pred_hidden: torch.Tensor
    pred_targets: torch.Tensor
    pred_mask: torch.Tensor


@dataclass(frozen=True)
class OfflineDSparkStepOutput:
    loss: torch.Tensor
    ce_loss: torch.Tensor
    l1_loss: torch.Tensor
    confidence_loss: torch.Tensor
    accuracy: torch.Tensor
    accept_len: torch.Tensor
    valid_tokens: torch.Tensor
    valid_blocks: torch.Tensor
    ce_numerator: torch.Tensor
    ce_denominator: torch.Tensor
    l1_numerator: torch.Tensor
    l1_denominator: torch.Tensor
    confidence_numerator: torch.Tensor
    confidence_denominator: torch.Tensor
    correct: torch.Tensor
    accept_total: torch.Tensor
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor


def gather_dspark_teacher_hidden(
    target_final_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    prediction_depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather AR state/predecessor at ``a+d`` for target token ``a+d+1``."""

    if target_final_hidden.ndim != 3 or input_ids.shape != target_final_hidden.shape[:2]:
        raise ValueError("target_final_hidden must align with input_ids")
    if anchor_positions.ndim != 2 or block_keep_mask.shape != anchor_positions.shape:
        raise ValueError("anchor positions and keep mask must have shape [batch, anchors]")
    if anchor_positions.shape[0] != input_ids.shape[0] or prediction_depth < 1:
        raise ValueError("invalid DSpark gather dimensions")
    batch, tokens = input_ids.shape
    anchors = anchor_positions.to(device=input_ids.device, dtype=torch.long)
    depth = torch.arange(prediction_depth, device=input_ids.device)
    positions = anchors[..., None] + depth
    safe = positions.clamp(0, max(tokens - 1, 0))
    expanded_ids = input_ids[:, None, :].expand(batch, anchors.shape[1], tokens)
    predecessors = expanded_ids.gather(2, safe)
    hidden_size = target_final_hidden.shape[-1]
    expanded_hidden = target_final_hidden[:, None].expand(
        batch, anchors.shape[1], tokens, hidden_size
    )
    teacher = expanded_hidden.gather(
        2, safe[..., None].expand(*safe.shape, hidden_size)
    )
    keep = block_keep_mask.to(device=input_ids.device, dtype=torch.bool)
    teacher = teacher.masked_fill(~keep[..., None, None], 0)
    predecessors = predecessors.masked_fill(~keep[..., None], 0)
    return teacher, predecessors


class _OfflineDFlashBase(nn.Module):
    """Shared cache, anchor, block, and frozen-I/O preparation."""

    def __init__(
        self,
        draft_model: DFlashDraftModel | Qwen3DFlash2DraftModel | DSparkDraftModel,
        target_io: FrozenTargetIO,
        *,
        cache_manifest: Mapping[str, Any],
        num_anchors: int,
        gamma: float = 7.0,
        token_chunk_size: int = 256,
        vocab_chunk_size: int = 4096,
        global_seed: int = 1234,
        anchor_seed: int | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= int(num_anchors) <= 64:
            raise ValueError("num_anchors must be in [1, 64]")
        if token_chunk_size < 1 or vocab_chunk_size < 1:
            raise ValueError("projection chunk sizes must be positive")
        validate_cache_io_compatibility(
            cache_manifest,
            target_io.manifest,
            expected_layer_ids=tuple(draft_model.target_layer_ids),
        )
        if int(target_io.manifest["vocab_size"]) != int(draft_model.config.vocab_size):
            raise ValueError("target I/O vocabulary differs from draft config")
        if int(target_io.manifest["hidden_size"]) != int(draft_model.config.hidden_size):
            raise ValueError("target I/O hidden size differs from draft config")
        self.draft_model = draft_model
        # Keep the two huge frozen tensors out of FSDP/optimizer/checkpoints.
        object.__setattr__(self, "_target_io", target_io)
        object.__setattr__(
            self,
            "_frozen_io_versions",
            (target_io.embed_tokens.weight._version, target_io.lm_head.weight._version),
        )
        self.cache_manifest = dict(cache_manifest)
        self.num_anchors = int(num_anchors)
        self.gamma = float(gamma)
        self.token_chunk_size = int(token_chunk_size)
        self.vocab_chunk_size = int(vocab_chunk_size)
        self.global_seed = int(global_seed if anchor_seed is None else anchor_seed)
        # Legacy checkpoint entrypoints still serialize this object. Aligned
        # anchor choice itself is pure and never consumes its state.
        self.anchor_generator = torch.Generator(device="cpu").manual_seed(self.global_seed)

    @property
    def target_embed_weight(self) -> torch.Tensor:
        return self._target_io.embed_tokens.weight

    @property
    def target_lm_head_weight(self) -> torch.Tensor:
        return self._target_io.lm_head.weight

    def assert_frozen_io_unchanged(self) -> None:
        current = (self.target_embed_weight._version, self.target_lm_head_weight._version)
        if current != self._frozen_io_versions:
            raise RuntimeError("frozen target token I/O was modified during training")

    def _anchors(
        self,
        batch: Mapping[str, Any],
        *,
        epoch: int,
        anchor_positions: torch.Tensor | None,
        block_keep_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor_positions is not None or block_keep_mask is not None:
            if anchor_positions is None or block_keep_mask is None:
                raise ValueError("anchor_positions and block_keep_mask must be supplied together")
            return anchor_positions, block_keep_mask
        sample_ids = batch.get("sample_id")
        if not isinstance(sample_ids, (list, tuple)):
            raise ValueError("automatic aligned anchor sampling requires batch sample_id")
        return sample_anchor_positions(
            batch["loss_mask"],
            sample_ids=sample_ids,
            global_seed=self.global_seed,
            epoch=int(epoch),
            attention_mask=batch.get("attention_mask"),
            block_size=self.draft_model.block_size,
            num_anchors=self.num_anchors,
        )

    def _prepare_hidden(
        self,
        batch: Mapping[str, Any],
        *,
        epoch: int,
        anchor_positions: torch.Tensor | None,
        block_keep_mask: torch.Tensor | None,
    ) -> _PreparedHidden:
        self.assert_frozen_io_unchanged()
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"].to(torch.bool)
        anchors, keep = self._anchors(
            batch,
            epoch=epoch,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )
        blocks = build_dflash_blocks(
            input_ids,
            loss_mask,
            anchors,
            keep,
            attention_mask=batch.get("attention_mask"),
            block_size=self.draft_model.block_size,
            mask_token_id=self.draft_model.mask_token_id,
            attention_dtype=self.target_embed_weight.dtype,
        )
        if self.target_embed_weight.device != input_ids.device:
            raise ValueError("frozen target I/O and training batch must be on the same device")
        noise_embedding = self._target_io.embed_tokens(blocks.noise_ids)
        output_hidden = self.draft_model(
            position_ids=blocks.full_position_ids,
            attention_mask=blocks.attention_mask,
            noise_embedding=noise_embedding,
            target_hidden=batch["hidden_states"].to(noise_embedding.dtype),
            conv_block_size=self.draft_model.block_size,
        )
        batch_size, num_blocks, block_size = blocks.target_ids.shape
        pred_hidden = output_hidden.reshape(
            batch_size, num_blocks, block_size, output_hidden.shape[-1]
        )[:, :, 1:]
        pred_targets = blocks.target_ids[:, :, 1:]
        pred_mask = blocks.target_mask[:, :, 1:]
        return _PreparedHidden(blocks, pred_hidden, pred_targets, pred_mask)

    def _prepare(
        self,
        batch: Mapping[str, Any],
        *,
        top_k: int,
        epoch: int,
        anchor_positions: torch.Tensor | None,
        block_keep_mask: torch.Tensor | None,
    ) -> _PreparedStep:
        prepared = self._prepare_hidden(
            batch,
            epoch=epoch,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )
        projection = chunked_lm_projection(
            prepared.pred_hidden,
            prepared.pred_targets,
            self.target_lm_head_weight,
            top_k=top_k,
            token_chunk_size=self.token_chunk_size,
            vocab_chunk_size=self.vocab_chunk_size,
            token_mask=prepared.pred_mask,
        )
        return _PreparedStep(
            prepared.blocks,
            prepared.pred_hidden,
            prepared.pred_targets,
            prepared.pred_mask,
            projection,
        )

    @staticmethod
    def _metrics(
        prepared: _PreparedStep, selected_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        valid = prepared.pred_mask.to(torch.bool)
        base_ids = prepared.projection.topk_ids[..., 0]
        valid_tokens = valid.sum()
        denominator = valid_tokens.clamp_min(1).float()
        base_correct = ((base_ids == prepared.pred_targets) & valid).sum()
        selected_correct = ((selected_ids == prepared.pred_targets) & valid).sum()
        base_accept, base_accept_total, _ = compute_acceptance_stats(
            base_ids, prepared.pred_targets, valid
        )
        selected_accept, selected_accept_total, valid_blocks = compute_acceptance_stats(
            selected_ids, prepared.pred_targets, valid
        )
        return (
            base_ids,
            valid_tokens,
            valid_blocks,
            base_correct,
            selected_correct,
            base_correct.float() / denominator,
            selected_correct.float() / denominator,
            base_accept,
            selected_accept,
            base_accept_total,
            selected_accept_total,
        )


class OfflineDFlashTrainer(_OfflineDFlashBase):
    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        epoch: int = 0,
        anchor_positions: torch.Tensor | None = None,
        block_keep_mask: torch.Tensor | None = None,
    ) -> OfflineStepOutput:
        prepared = self._prepare(
            batch,
            top_k=1,
            epoch=epoch,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )
        terms = depth_weighted_objective(
            prepared.projection.nll, prepared.pred_mask, gamma=self.gamma
        )
        base_loss = global_weighted_mean(terms.mean, terms.denominator)
        selected_ids = prepared.projection.topk_ids[..., 0]
        (
            _, valid_tokens, valid_blocks, base_correct, selected_correct,
            base_accuracy, selected_accuracy, base_accept, selected_accept,
            base_accept_total, selected_accept_total,
        ) = self._metrics(prepared, selected_ids)
        zero = base_loss.detach() * 0.0
        return OfflineStepOutput(
            loss=base_loss,
            base_loss=base_loss.detach(),
            selector_loss=zero,
            base_accuracy=base_accuracy.detach(),
            selector_accuracy=selected_accuracy.detach(),
            base_accept_len=base_accept.detach(),
            selector_accept_len=selected_accept.detach(),
            candidate_recall=base_accuracy.detach(),
            valid_tokens=valid_tokens.detach(),
            valid_blocks=valid_blocks.detach(),
            loss_weight=terms.denominator.detach(),
            base_numerator=terms.numerator.detach(),
            base_denominator=terms.denominator.detach(),
            selector_numerator=zero,
            selector_denominator=zero,
            base_correct=base_correct.detach(),
            selector_correct=selected_correct.detach(),
            base_accept_total=base_accept_total.detach(),
            selector_accept_total=selected_accept_total.detach(),
            candidate_hits=base_correct.detach(),
            candidate_total=valid_tokens.detach(),
            anchor_positions=prepared.blocks.anchor_positions.detach(),
            block_keep_mask=prepared.blocks.block_keep_mask.detach(),
        )


class OfflineDFlash2Trainer(_OfflineDFlashBase):
    def __init__(self, *args: Any, selector_loss_weight: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.selector_loss_weight = float(selector_loss_weight)

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        epoch: int = 0,
        anchor_positions: torch.Tensor | None = None,
        block_keep_mask: torch.Tensor | None = None,
    ) -> OfflineStepOutput:
        top_k = self.draft_model.candidate_selector.top_k
        prepared = self._prepare(
            batch,
            top_k=top_k,
            epoch=epoch,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )
        selector_scores = self.draft_model.candidate_selector(
            prepared.pred_hidden,
            prepared.projection.topk_scores,
            prepared.projection.topk_ids,
            prepared.blocks.target_ids[:, :, :-1],
        )
        losses = compute_dflash2_loss(
            base_nll=prepared.projection.nll,
            candidate_ids=prepared.projection.topk_ids,
            selector_scores=selector_scores,
            target_ids=prepared.blocks.target_ids,
            pred_mask=prepared.pred_mask,
            gamma=self.gamma,
            selector_loss_weight=self.selector_loss_weight,
        )
        base_loss = global_weighted_mean(losses.base_loss, losses.base_denominator)
        selector_loss = global_weighted_mean(
            losses.selector_loss, losses.selector_denominator
        )
        training_loss = base_loss + self.selector_loss_weight * selector_loss
        selected_ids = prepared.projection.topk_ids.gather(
            -1, selector_scores.argmax(dim=-1, keepdim=True)
        ).squeeze(-1)
        (
            _, valid_tokens, valid_blocks, base_correct, selector_correct,
            base_accuracy, selector_accuracy, base_accept, selector_accept,
            base_accept_total, selector_accept_total,
        ) = self._metrics(prepared, selected_ids)
        candidate_recall = losses.candidate_hits.float() / losses.candidate_total.clamp_min(1).float()
        return OfflineStepOutput(
            loss=training_loss,
            base_loss=base_loss.detach(),
            selector_loss=selector_loss.detach(),
            base_accuracy=base_accuracy.detach(),
            selector_accuracy=selector_accuracy.detach(),
            base_accept_len=base_accept.detach(),
            selector_accept_len=selector_accept.detach(),
            candidate_recall=candidate_recall.detach(),
            valid_tokens=valid_tokens.detach(),
            valid_blocks=valid_blocks.detach(),
            loss_weight=losses.base_denominator,
            base_numerator=losses.base_numerator,
            base_denominator=losses.base_denominator,
            selector_numerator=losses.selector_numerator,
            selector_denominator=losses.selector_denominator,
            base_correct=base_correct.detach(),
            selector_correct=selector_correct.detach(),
            base_accept_total=base_accept_total.detach(),
            selector_accept_total=selector_accept_total.detach(),
            candidate_hits=losses.candidate_hits,
            candidate_total=losses.candidate_total,
            anchor_positions=prepared.blocks.anchor_positions.detach(),
            block_keep_mask=prepared.blocks.block_keep_mask.detach(),
        )


class OfflineDSparkTrainer(_OfflineDFlashBase):
    def __init__(
        self,
        *args: Any,
        ce_weight: float = 0.1,
        l1_weight: float = 0.9,
        confidence_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        # DSpark does not use top-k token projection, but accepts the common
        # chunk argument surface so unified launchers stay method-independent.
        kwargs.setdefault("token_chunk_size", 1)
        super().__init__(*args, **kwargs)
        if not isinstance(self.draft_model, DSparkDraftModel):
            raise TypeError("OfflineDSparkTrainer requires DSparkDraftModel")
        self.ce_weight = float(ce_weight)
        self.l1_weight = float(l1_weight)
        self.confidence_weight = float(confidence_weight)

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        epoch: int = 0,
        anchor_positions: torch.Tensor | None = None,
        block_keep_mask: torch.Tensor | None = None,
    ) -> OfflineDSparkStepOutput:
        final_hidden = batch.get("target_final_hidden")
        if not isinstance(final_hidden, torch.Tensor):
            raise ValueError("DSpark requires target_final_hidden from cache schema v2")
        prepared = self._prepare_hidden(
            batch,
            epoch=epoch,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )
        teacher_hidden, predecessors = gather_dspark_teacher_hidden(
            final_hidden,
            batch["input_ids"],
            prepared.blocks.anchor_positions,
            prepared.blocks.block_keep_mask,
            prediction_depth=prepared.pred_hidden.shape[-2],
        )
        confidence_logits = self.draft_model.confidence_logits(prepared.pred_hidden)
        losses = compute_dspark_loss(
            draft_hidden=prepared.pred_hidden,
            target_hidden=teacher_hidden.to(prepared.pred_hidden.dtype),
            target_ids=prepared.pred_targets,
            predecessor_ids=predecessors,
            confidence_logits=confidence_logits,
            lm_head_weight=self.target_lm_head_weight,
            markov_head=self.draft_model.markov_head,
            token_mask=prepared.pred_mask,
            gamma=self.gamma,
            vocab_chunk_size=self.vocab_chunk_size,
            ce_weight=self.ce_weight,
            l1_weight=self.l1_weight,
            confidence_weight=self.confidence_weight,
        )
        ce_loss = global_weighted_mean(losses.ce.mean, losses.ce.denominator)
        l1_loss = global_weighted_mean(losses.l1.mean, losses.l1.denominator)
        confidence_loss = global_weighted_mean(
            losses.confidence.mean, losses.confidence.denominator
        )
        total = (
            self.ce_weight * ce_loss
            + self.l1_weight * l1_loss
            + self.confidence_weight * confidence_loss
        )
        valid = prepared.pred_mask.to(torch.bool)
        valid_tokens = valid.sum()
        correct = ((losses.draft_top1_ids == prepared.pred_targets) & valid).sum()
        accuracy = correct.float() / valid_tokens.clamp_min(1).float()
        accept_len, accept_total, valid_blocks = compute_acceptance_stats(
            losses.draft_top1_ids, prepared.pred_targets, valid
        )
        return OfflineDSparkStepOutput(
            loss=total,
            ce_loss=ce_loss.detach(),
            l1_loss=l1_loss.detach(),
            confidence_loss=confidence_loss.detach(),
            accuracy=accuracy.detach(),
            accept_len=accept_len.detach(),
            valid_tokens=valid_tokens.detach(),
            valid_blocks=valid_blocks.detach(),
            ce_numerator=losses.ce.numerator.detach(),
            ce_denominator=losses.ce.denominator.detach(),
            l1_numerator=losses.l1.numerator.detach(),
            l1_denominator=losses.l1.denominator.detach(),
            confidence_numerator=losses.confidence.numerator.detach(),
            confidence_denominator=losses.confidence.denominator.detach(),
            correct=correct.detach(),
            accept_total=accept_total.detach(),
            anchor_positions=prepared.blocks.anchor_positions.detach(),
            block_keep_mask=prepared.blocks.block_keep_mask.detach(),
        )
