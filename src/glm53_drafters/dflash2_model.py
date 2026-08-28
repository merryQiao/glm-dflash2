from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dflash_model import DFlashModel
from .modeling_common import (
    DraftModelConfig,
    GroupedDynamicCausalConv,
    initialize_draft_model,
)


class BlockLocalDynamicConv(GroupedDynamicCausalConv):
    """Public two-tap grouped convolution used inside every DFlash2 layer."""


class CandidateSelector(nn.Module):
    """Predecessor/hidden low-rank reranker over base top-k candidates."""

    def __init__(self, *, hidden_size: int, vocab_size: int,
                 rank: int = 256, top_k: int = 16) -> None:
        super().__init__()
        if rank < 1 or not 1 <= top_k <= vocab_size:
            raise ValueError("invalid selector rank/top_k")
        self.hidden_size, self.vocab_size = hidden_size, vocab_size
        self.rank, self.top_k = rank, top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        with torch.no_grad():
            self.predecessor_codebook.normal_(mean=0.0, std=0.02)
            self.successor_codebook.zero_()

    def reset_stabilized_parameters(self, *, initializer_range: float) -> None:
        with torch.no_grad():
            self.predecessor_codebook.normal_(mean=0.0, std=initializer_range)
            self.successor_codebook.zero_()

    def _base_candidates(
        self, hidden: torch.Tensor, lm_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tuple(lm_weight.shape) != (self.vocab_size, self.hidden_size):
            raise ValueError("LM head shape differs from selector contract")
        base_logits = F.linear(hidden, lm_weight).float()
        unary, candidate_ids = torch.topk(base_logits, self.top_k, dim=-1, sorted=True)
        return base_logits, unary, candidate_ids

    def _score_candidates(
        self,
        hidden: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        unary: torch.Tensor,
    ) -> torch.Tensor:
        if predecessor_ids.shape != hidden.shape[:-1]:
            raise ValueError("predecessor IDs must match hidden token dimensions")
        gate = F.embedding(predecessor_ids, self.predecessor_codebook)
        gate = gate * self.hidden_projection(hidden).to(gate.dtype)
        successor = F.embedding(candidate_ids, self.successor_codebook)
        pairwise = torch.einsum("...r,...kr->...k", gate, successor)
        return unary + pairwise.float()

    def forward(self, hidden: torch.Tensor, lm_weight: torch.Tensor,
                predecessor_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, unary, candidate_ids = self._base_candidates(hidden, lm_weight)
        return candidate_ids, self._score_candidates(
            hidden, predecessor_ids, candidate_ids, unary
        )

    def training_forward(
        self,
        hidden: torch.Tensor,
        lm_weight: torch.Tensor,
        predecessor_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return target-bearing candidates and pre-injection unary recall."""

        if target_ids.shape != hidden.shape[:-1]:
            raise ValueError("training target IDs must match hidden token dimensions")
        base_logits, unary, candidate_ids = self._base_candidates(hidden, lm_weight)
        unary_hits = candidate_ids.eq(target_ids.unsqueeze(-1)).any(-1)
        target_unary = base_logits.gather(-1, target_ids.long().unsqueeze(-1)).squeeze(-1)
        final_ids = torch.where(unary_hits, candidate_ids[..., -1], target_ids.long())
        final_unary = torch.where(unary_hits, unary[..., -1], target_unary)
        candidate_ids = torch.cat((candidate_ids[..., :-1], final_ids.unsqueeze(-1)), -1)
        unary = torch.cat((unary[..., :-1], final_unary.unsqueeze(-1)), -1)
        logits = self._score_candidates(
            hidden, predecessor_ids, candidate_ids, unary
        )
        return candidate_ids, logits, unary_hits


class DFlash2Model(nn.Module):
    def __init__(self, config: DraftModelConfig, *, convolution_group_size: int = 16,
                 selector_rank: int = 256, selector_top_k: int = 16) -> None:
        super().__init__()
        self.config = config
        self.dflash = DFlashModel(
            config,
            dynamic_convolution=True,
            convolution_group_size=convolution_group_size,
        )
        self.selector = CandidateSelector(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            rank=selector_rank,
            top_k=selector_top_k,
        )
        initialize_draft_model(self, initializer_range=config.initializer_range)

    @property
    def target_projection(self) -> nn.Linear:
        return self.dflash.target_projection

    @property
    def layers(self) -> nn.ModuleList:
        return self.dflash.layers

    @property
    def candidate_selector(self) -> CandidateSelector:
        return self.selector

    def forward(self, noise_embedding: torch.Tensor, target_context: torch.Tensor, *,
                lm_weight: torch.Tensor, predecessor_ids: torch.Tensor,
                position_ids: torch.Tensor, attention_mask: torch.Tensor,
                block_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.dflash(
            noise_embedding,
            target_context,
            position_ids=position_ids,
            attention_mask=attention_mask,
            block_size=block_size,
        )
        candidate_ids, candidate_logits = self.selector(
            hidden, lm_weight, predecessor_ids
        )
        return hidden, candidate_ids, candidate_logits

    def training_forward(
        self,
        noise_embedding: torch.Tensor,
        target_context: torch.Tensor,
        *,
        lm_weight: torch.Tensor,
        predecessor_ids: torch.Tensor,
        target_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.dflash(
            noise_embedding,
            target_context,
            position_ids=position_ids,
            attention_mask=attention_mask,
            block_size=block_size,
        )
        candidate_ids, candidate_logits, unary_hits = self.selector.training_forward(
            hidden, lm_weight, predecessor_ids, target_ids
        )
        return hidden, candidate_ids, candidate_logits, unary_hits
