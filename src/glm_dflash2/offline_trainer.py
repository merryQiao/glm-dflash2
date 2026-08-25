from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .chunked_lm_head import chunked_lm_projection
from .dflash2_blocks import build_dflash_blocks, sample_anchor_positions
from .dflash2_model import Qwen3DFlash2DraftModel
from .dflash2_objective import compute_acceptance_stats, compute_dflash2_loss
from .distributed import global_weighted_mean
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


class OfflineDFlash2Trainer(nn.Module):
    """Trainable draft wrapper over frozen, non-registered target token I/O."""

    def __init__(
        self,
        draft_model: Qwen3DFlash2DraftModel,
        target_io: FrozenTargetIO,
        *,
        cache_manifest: Mapping[str, Any],
        num_anchors: int,
        gamma: float = 7.0,
        selector_loss_weight: float = 1.0,
        token_chunk_size: int = 256,
        vocab_chunk_size: int = 4096,
        anchor_seed: int = 1234,
    ) -> None:
        super().__init__()
        if num_anchors < 1 or num_anchors > 64:
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
        # Frozen I/O is intentionally outside the registered module tree.  FSDP2,
        # optimizers and sharded checkpoints therefore see only draft parameters.
        object.__setattr__(self, "_target_io", target_io)
        object.__setattr__(
            self,
            "_frozen_io_versions",
            (target_io.embed_tokens.weight._version, target_io.lm_head.weight._version),
        )
        self.cache_manifest = dict(cache_manifest)
        self.num_anchors = int(num_anchors)
        self.gamma = float(gamma)
        self.selector_loss_weight = float(selector_loss_weight)
        self.token_chunk_size = int(token_chunk_size)
        self.vocab_chunk_size = int(vocab_chunk_size)
        self.anchor_generator = torch.Generator(device="cpu").manual_seed(int(anchor_seed))

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

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        anchor_positions: torch.Tensor | None = None,
        block_keep_mask: torch.Tensor | None = None,
    ) -> OfflineStepOutput:
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"].to(torch.bool)
        target_hidden = batch["hidden_states"]
        if anchor_positions is None or block_keep_mask is None:
            if anchor_positions is not None or block_keep_mask is not None:
                raise ValueError("anchor_positions and block_keep_mask must be supplied together")
            anchor_positions, block_keep_mask = sample_anchor_positions(
                loss_mask,
                attention_mask=batch.get("attention_mask"),
                block_size=self.draft_model.block_size,
                num_anchors=self.num_anchors,
                generator=self.anchor_generator,
            )

        blocks = build_dflash_blocks(
            input_ids,
            loss_mask,
            anchor_positions,
            block_keep_mask,
            attention_mask=batch.get("attention_mask"),
            block_size=self.draft_model.block_size,
            mask_token_id=self.draft_model.mask_token_id,
            sliding_window=int(self.draft_model.config.sliding_window),
            attention_dtype=self.target_embed_weight.dtype,
        )
        if self.target_embed_weight.device != input_ids.device:
            raise ValueError("frozen target I/O and training batch must be on the same device")
        noise_embedding = self._target_io.embed_tokens(blocks.noise_ids)
        output_hidden = self.draft_model(
            position_ids=blocks.full_position_ids,
            attention_mask=blocks.attention_mask,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden.to(noise_embedding.dtype),
            conv_block_size=self.draft_model.block_size,
        )

        batch_size, num_blocks, block_size = blocks.target_ids.shape
        hidden_size = output_hidden.shape[-1]
        pred_hidden = output_hidden.reshape(
            batch_size, num_blocks, block_size, hidden_size
        )[:, :, 1:]
        pred_targets = blocks.target_ids[:, :, 1:]
        pred_mask = blocks.target_mask[:, :, 1:]
        projection = chunked_lm_projection(
            pred_hidden,
            pred_targets,
            self.target_lm_head_weight,
            top_k=self.draft_model.candidate_selector.top_k,
            token_chunk_size=self.token_chunk_size,
            vocab_chunk_size=self.vocab_chunk_size,
            token_mask=pred_mask,
        )
        selector_scores = self.draft_model.candidate_selector(
            pred_hidden,
            projection.topk_scores,
            projection.topk_ids,
            blocks.target_ids[:, :, :-1],
        )
        losses = compute_dflash2_loss(
            base_nll=projection.nll,
            candidate_ids=projection.topk_ids,
            selector_scores=selector_scores,
            target_ids=blocks.target_ids,
            pred_mask=pred_mask,
            gamma=self.gamma,
            selector_loss_weight=self.selector_loss_weight,
        )
        base_loss = global_weighted_mean(losses.base_loss, losses.base_denominator)
        selector_loss = global_weighted_mean(
            losses.selector_loss, losses.selector_denominator
        )
        training_loss = base_loss + self.selector_loss_weight * selector_loss

        with torch.no_grad():
            valid = pred_mask.to(torch.bool)
            base_ids = projection.topk_ids[..., 0]
            selected_ids = projection.topk_ids.gather(
                -1, selector_scores.argmax(dim=-1, keepdim=True)
            ).squeeze(-1)
            valid_tokens = valid.sum()
            denominator = valid_tokens.clamp_min(1).float()
            base_correct = ((base_ids == pred_targets) & valid).sum()
            selector_correct = ((selected_ids == pred_targets) & valid).sum()
            base_accuracy = base_correct.float() / denominator
            selector_accuracy = selector_correct.float() / denominator
            base_accept, base_accept_total, _ = compute_acceptance_stats(
                base_ids, pred_targets, valid
            )
            selector_accept, selector_accept_total, valid_blocks = compute_acceptance_stats(
                selected_ids, pred_targets, valid
            )
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
            anchor_positions=blocks.anchor_positions.detach(),
            block_keep_mask=blocks.block_keep_mask.detach(),
        )
