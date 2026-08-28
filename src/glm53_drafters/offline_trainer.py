from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .chunked_lm_head import AdditiveScalar, additive_mean, chunked_cross_entropy
from .dflash2_model import DFlash2Model
from .dflash_model import DFlashModel
from .dspark_model import DSparkModel
from .dspark_model import teacher_forced_predecessor_ids
from .objectives import (
    confidence_bce,
    depth_weights,
    exact_total_variation,
    selector_cross_entropy,
)


@dataclass(frozen=True)
class TrainingBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    position_ids: torch.Tensor
    auxiliary_hidden: torch.Tensor
    target_final_hidden: torch.Tensor | None
    keep_mask: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class TrainingResult:
    loss: torch.Tensor
    metrics: Mapping[str, AdditiveScalar]


def _weighted_scalar(values: torch.Tensor, weights: torch.Tensor) -> AdditiveScalar:
    numerator = (values.float() * weights.float()).sum(dtype=torch.float32)
    denominator = weights.float().sum(dtype=torch.float32)
    return AdditiveScalar(numerator, denominator, additive_mean(numerator, denominator))


def acceptance_statistics(
    student_logits: torch.Tensor,
    target_logits: torch.Tensor,
    weights: torch.Tensor,
) -> AdditiveScalar:
    accepted = student_logits.argmax(-1).eq(target_logits.argmax(-1)).float()
    return _weighted_scalar(accepted, weights)


class OfflineMethodTrainer:
    """Method objective owner; frozen target I/O is never registered on the model."""

    def __init__(
        self,
        *,
        method: str,
        block_size: int,
        model: DFlashModel | DFlash2Model | DSparkModel,
        target_embedding: nn.Embedding,
        target_lm_head: nn.Module,
        vocab_chunk_size: int = 8192,
    ) -> None:
        allowed = {"dflash": (8, 16), "dflash2": (8, 16), "dspark": (8,)}
        if method not in allowed or block_size not in allowed[method]:
            if method == "dspark":
                raise ValueError("DSpark supports physical block size 8 only")
            raise ValueError(f"invalid method/block combination: {method}/{block_size}")
        if getattr(target_lm_head, "bias", None) is not None:
            raise ValueError("frozen target LM head must not have a bias")
        if not isinstance(getattr(target_lm_head, "weight", None), nn.Parameter):
            raise TypeError("frozen target LM head must expose a weight parameter")
        self.method = method
        self.block_size = int(block_size)
        self.model = model
        self.target_embedding = target_embedding
        self.target_lm_head = target_lm_head
        self.vocab_chunk_size = int(vocab_chunk_size)
        for module in (target_embedding, target_lm_head):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)

    def _validate_batch(self, batch: TrainingBatch) -> None:
        if batch.input_ids.shape != batch.target_ids.shape:
            raise ValueError("input and target IDs must have equal shape")
        if batch.input_ids.ndim != 3 or batch.input_ids.shape[-1] != self.block_size:
            raise ValueError("batch physical block size differs from method contract")
        if batch.keep_mask.shape != batch.input_ids.shape[:-1]:
            raise ValueError("keep mask must have shape [batch,anchors]")
        if batch.auxiliary_hidden.ndim != 4 or (
            batch.auxiliary_hidden.shape[0] != batch.input_ids.shape[0]
        ):
            raise ValueError("auxiliary context must be [batch,sequence,layers,hidden]")
        batch_size, anchors, depth = batch.input_ids.shape
        context = batch.auxiliary_hidden.shape[1]
        queries = anchors * depth
        if batch.position_ids.shape != (batch_size, context + queries):
            raise ValueError("positions must cover [context | flattened blocks]")
        if batch.attention_mask.shape != (batch_size, 1, queries, context + queries):
            raise ValueError("attention mask shape differs from DFlash layout")
        if self.method == "dspark" and batch.target_final_hidden is None:
            raise ValueError("DSpark requires target_final_hidden")
        if batch.target_final_hidden is not None and batch.target_final_hidden.shape[:3] != (
            batch_size, anchors, depth
        ):
            raise ValueError("target final hidden must align with physical blocks")

    def _weights(self, batch: TrainingBatch) -> torch.Tensor:
        gamma = 7.0 if self.block_size == 16 else 4.0
        depth = depth_weights(
            block_size=self.block_size,
            gamma=gamma,
            device=batch.input_ids.device,
        )
        return batch.keep_mask.float().unsqueeze(-1) * depth

    def compute_loss(self, batch: TrainingBatch) -> TrainingResult:
        self._validate_batch(batch)
        with torch.no_grad():
            embedding_blocks = self.target_embedding(batch.input_ids)
            embeddings = embedding_blocks.flatten(1, 2)
        weights = self._weights(batch)
        targets = batch.target_ids[..., 1:]
        lm_weight = self.target_lm_head.weight
        batch_size, anchors, depth = batch.input_ids.shape
        predecessor_ids = teacher_forced_predecessor_ids(batch.target_ids).flatten(1, 2)

        if self.method == "dflash":
            if not isinstance(self.model, DFlashModel):
                raise TypeError("dflash method requires DFlashModel")
            hidden = self.model(
                embeddings,
                batch.auxiliary_hidden,
                position_ids=batch.position_ids,
                attention_mask=batch.attention_mask,
                block_size=self.block_size,
            ).reshape(batch_size, anchors, depth, -1)[..., 1:, :]
            base = chunked_cross_entropy(
                hidden,
                lm_weight,
                targets,
                weights=weights,
                vocab_chunk_size=self.vocab_chunk_size,
            )
            return TrainingResult(base.mean, {"base": base, "total": base})

        if self.method == "dflash2":
            if not isinstance(self.model, DFlash2Model):
                raise TypeError("dflash2 method requires DFlash2Model")
            hidden, candidate_ids, candidate_logits, unary_hits = self.model.training_forward(
                embeddings,
                batch.auxiliary_hidden,
                lm_weight=lm_weight,
                predecessor_ids=predecessor_ids,
                target_ids=batch.target_ids.flatten(1, 2),
                position_ids=batch.position_ids,
                attention_mask=batch.attention_mask,
                block_size=self.block_size,
            )
            hidden = hidden.reshape(batch_size, anchors, depth, -1)[..., 1:, :]
            candidate_ids = candidate_ids.reshape(batch_size, anchors, depth, -1)
            candidate_logits = candidate_logits.reshape(batch_size, anchors, depth, -1)
            unary_hits = unary_hits.reshape(batch_size, anchors, depth)
            base = chunked_cross_entropy(
                hidden,
                lm_weight,
                targets,
                weights=weights,
                vocab_chunk_size=self.vocab_chunk_size,
            )
            selector = selector_cross_entropy(
                candidate_ids[..., 1:, :],
                candidate_logits[..., 1:, :],
                targets,
                weights=weights,
            )
            unary_recall = _weighted_scalar(unary_hits[..., 1:].float(), weights)
            total_numerator = base.numerator + selector.numerator
            total = AdditiveScalar(
                total_numerator,
                base.denominator,
                additive_mean(total_numerator, base.denominator),
            )
            return TrainingResult(
                total.mean,
                {
                    "base": base,
                    "selector": selector,
                    "unary_recall": unary_recall,
                    "total": total,
                },
            )

        if not isinstance(self.model, DSparkModel):
            raise TypeError("dspark method requires DSparkModel")
        # Official DSpark B8 has seven queries: anchor plus six masks.  All
        # seven outputs predict successors a+1..a+7; unlike DFlash, output
        # zero is supervised rather than discarded.
        dspark_depth = depth - 1
        dspark_embeddings = embedding_blocks[..., :dspark_depth, :].flatten(1, 2)
        dspark_predecessors = batch.target_ids[..., :dspark_depth].flatten(1, 2)
        query_indices = (
            torch.arange(anchors, device=batch.input_ids.device).unsqueeze(-1) * depth
            + torch.arange(dspark_depth, device=batch.input_ids.device)
        ).flatten()
        context = batch.auxiliary_hidden.shape[1]
        key_indices = torch.cat(
            (
                torch.arange(context, device=batch.input_ids.device),
                context + query_indices,
            )
        )
        dspark_attention = batch.attention_mask.index_select(2, query_indices).index_select(
            3, key_indices
        )
        draft_positions = batch.position_ids[:, context:].reshape(
            batch_size, anchors, depth
        )[..., :dspark_depth].flatten(1, 2)
        dspark_positions = torch.cat(
            (batch.position_ids[:, :context], draft_positions), dim=1
        )
        hidden, markov_residual, confidence_logits = self.model(
            dspark_embeddings,
            batch.auxiliary_hidden,
            predecessor_token_ids=dspark_predecessors,
            position_ids=dspark_positions,
            attention_mask=dspark_attention,
            block_size=dspark_depth,
        )
        hidden = hidden.reshape(batch_size, anchors, dspark_depth, -1)
        markov_residual = markov_residual.reshape(
            batch_size, anchors, dspark_depth, -1
        )
        confidence_logits = confidence_logits.reshape(
            batch_size, anchors, dspark_depth
        )
        student_logits = F.linear(hidden, lm_weight).float()
        student_logits = student_logits + markov_residual.float()
        assert batch.target_final_hidden is not None
        target_logits = F.linear(
            batch.target_final_hidden[..., :-1, :], lm_weight
        ).float()
        ce_values = F.cross_entropy(
            student_logits.reshape(-1, student_logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        ce = _weighted_scalar(ce_values, weights)
        tv = exact_total_variation(student_logits, target_logits, weights=weights)
        confidence = confidence_bce(
            confidence_logits, tv.per_token, weights=weights
        )
        numerator = 0.1 * ce.numerator + 0.9 * tv.numerator + confidence.numerator
        total = AdditiveScalar(
            numerator,
            ce.denominator,
            additive_mean(numerator, ce.denominator),
        )
        acceptance = acceptance_statistics(student_logits, target_logits, weights)
        return TrainingResult(
            total.mean,
            {
                "ce": ce,
                "tv": tv,
                "confidence": confidence,
                "acceptance": acceptance,
                "total": total,
            },
        )
