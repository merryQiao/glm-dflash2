from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .chunked_lm_head import AdditiveScalar, additive_mean, chunked_cross_entropy
from .dflash_model import DFlashModel
from .dflash2_model import DFlash2Model
from .dspark_model import DSparkModel, teacher_forced_predecessor_ids
from .objectives import confidence_bce, depth_weights, exact_total_variation, selector_cross_entropy


@dataclass(frozen=True)
class TrainingBatch:
    input_ids: torch.Tensor                 # [B,A,K], anchor + masks
    target_ids: torch.Tensor                # [B,A,K], real trajectory tokens
    position_ids: torch.Tensor              # [3,B,C+A*K]
    auxiliary_hidden: torch.Tensor          # [B,C,5,H]
    target_final_hidden: torch.Tensor | None # [B,A,K,H]
    keep_mask: torch.Tensor                 # [B,A]
    attention_mask: torch.Tensor            # [B,A,K,C+K], compact block mask


@dataclass(frozen=True)
class TrainingResult:
    loss: torch.Tensor
    metrics: Mapping[str, AdditiveScalar]


def _weighted(values: torch.Tensor, weights: torch.Tensor) -> AdditiveScalar:
    numerator, denominator = (values.float() * weights.float()).sum(), weights.float().sum()
    return AdditiveScalar(numerator, denominator, additive_mean(numerator, denominator))


class OfflineMethodTrainer:
    """Owns exact Stage C objectives while target embedding/head stay frozen."""

    def __init__(self, *, method: str, block_size: int, model: nn.Module,
                 target_embedding: nn.Embedding, target_lm_head: nn.Linear,
                 vocab_chunk_size: int = 8192) -> None:
        from .contracts import validate_method_block

        validate_method_block(method, block_size)
        if target_lm_head.bias is not None:
            raise ValueError("Thinker LM head must be unbiased")
        self.method, self.block_size, self.model = method, int(block_size), model
        self.target_embedding, self.target_lm_head = target_embedding, target_lm_head
        self.vocab_chunk_size = int(vocab_chunk_size)
        for module in (target_embedding, target_lm_head):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _validate(self, batch: TrainingBatch) -> None:
        if batch.input_ids.shape != batch.target_ids.shape or batch.input_ids.ndim != 3:
            raise ValueError("input_ids and target_ids must be equal [B,A,K] tensors")
        batch_size, anchors, depth = batch.input_ids.shape
        if depth != self.block_size or batch.keep_mask.shape != (batch_size, anchors):
            raise ValueError("physical blocks or keep mask violate the recipe")
        context = batch.auxiliary_hidden.shape[1]
        total = context + anchors * depth
        if batch.position_ids.shape != (3, batch_size, total):
            raise ValueError("position_ids must be exact [3,B,context+queries] mRoPE")
        if batch.attention_mask.shape != (batch_size, anchors, depth, context + depth):
            raise ValueError("attention mask differs from compact block layout")
        if self.method == "dspark" and batch.target_final_hidden is None:
            raise ValueError("DSpark requires the final Thinker hidden stream")

    def _weights(self, batch: TrainingBatch) -> torch.Tensor:
        gamma = 7.0 if self.block_size == 16 else 4.0
        decay = depth_weights(block_size=self.block_size, gamma=gamma,
                              device=batch.input_ids.device)
        return batch.keep_mask.float().unsqueeze(-1) * decay

    def compute_loss(self, batch: TrainingBatch) -> TrainingResult:
        self._validate(batch)
        batch_size, anchors, depth = batch.input_ids.shape
        context = batch.auxiliary_hidden.shape[1]
        weights, targets = self._weights(batch), batch.target_ids[..., 1:]
        with torch.no_grad():
            block_embeddings = self.target_embedding(batch.input_ids)
            embeddings = block_embeddings.flatten(1, 2)
        lm_weight = self.target_lm_head.weight

        if self.method == "dflash":
            assert isinstance(self.model, DFlashModel)
            hidden = self.model(
                embeddings, batch.auxiliary_hidden, position_ids=batch.position_ids,
                attention_mask=batch.attention_mask, block_size=depth,
            ).reshape(batch_size, anchors, depth, -1)[..., 1:, :]
            base = chunked_cross_entropy(hidden, lm_weight, targets, weights=weights,
                                         vocab_chunk_size=self.vocab_chunk_size)
            return TrainingResult(base.mean, {"base": base, "total": base})

        predecessors = teacher_forced_predecessor_ids(batch.target_ids).flatten(1, 2)
        if self.method == "dflash2":
            assert isinstance(self.model, DFlash2Model)
            hidden, ids, logits, hits = self.model.training_forward(
                embeddings, batch.auxiliary_hidden, lm_weight=lm_weight,
                predecessor_ids=predecessors, target_ids=batch.target_ids.flatten(1, 2),
                position_ids=batch.position_ids, attention_mask=batch.attention_mask,
                block_size=depth,
            )
            hidden = hidden.reshape(batch_size, anchors, depth, -1)[..., 1:, :]
            ids = ids.reshape(batch_size, anchors, depth, -1)[..., 1:, :]
            logits = logits.reshape(batch_size, anchors, depth, -1)[..., 1:, :]
            hits = hits.reshape(batch_size, anchors, depth)[..., 1:]
            base = chunked_cross_entropy(hidden, lm_weight, targets, weights=weights,
                                         vocab_chunk_size=self.vocab_chunk_size)
            selector = selector_cross_entropy(ids, logits, targets, weights=weights)
            recall = _weighted(hits.float(), batch.keep_mask.float().unsqueeze(-1).expand_as(weights))
            # Official DFlash2: independently normalized base CE + hit-only selector CE.
            total_mean = base.mean + selector.mean
            valid = (base.denominator > 0).float()
            total = AdditiveScalar(total_mean * valid, valid,
                                   additive_mean(total_mean * valid, valid))
            return TrainingResult(total.mean, {
                "base": base, "selector": selector, "candidate_recall": recall, "total": total,
            })

        assert isinstance(self.model, DSparkModel) and batch.target_final_hidden is not None
        # Physical B8 -> seven successor queries (anchor plus six masks).
        qdepth = depth - 1
        dspark_mask = batch.attention_mask[:, :, :qdepth, :context + qdepth]
        draft_positions = batch.position_ids[:, :, context:].reshape(
            3, batch_size, anchors, depth
        )[..., :qdepth].flatten(2, 3)
        positions = torch.cat((batch.position_ids[:, :, :context], draft_positions), dim=2)
        hidden, markov, confidence = self.model(
            block_embeddings[..., :qdepth, :].flatten(1, 2), batch.auxiliary_hidden,
            predecessor_token_ids=batch.target_ids[..., :qdepth].flatten(1, 2),
            position_ids=positions, attention_mask=dspark_mask, block_size=qdepth,
        )
        hidden = hidden.reshape(batch_size, anchors, qdepth, -1)
        markov = markov.reshape(batch_size, anchors, qdepth, -1)
        confidence = confidence.reshape(batch_size, anchors, qdepth)
        student_logits = F.linear(hidden, lm_weight).float() + markov.float()
        target_logits = F.linear(batch.target_final_hidden[..., :-1, :], lm_weight).float()
        ce = _weighted(F.cross_entropy(
            student_logits.flatten(0, -2), targets.flatten(), reduction="none"
        ).reshape_as(targets), weights)
        tv = exact_total_variation(student_logits, target_logits, weights)
        conf = confidence_bce(confidence, tv.per_token, weights)
        # Requested DSpark recipe: 0.1 CE + 0.9 full-vocabulary TV + confidence.
        numerator = 0.1 * ce.numerator + 0.9 * tv.numerator + conf.numerator
        total = AdditiveScalar(numerator, weights.sum(), additive_mean(numerator, weights.sum()))
        accept = _weighted(student_logits.argmax(-1).eq(target_logits.argmax(-1)).float(), weights)
        return TrainingResult(total.mean, {
            "ce": ce, "tv": tv, "confidence": conf, "acceptance": accept, "total": total,
        })
