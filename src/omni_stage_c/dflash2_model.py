from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dflash_model import DFlashModel
from .modeling_common import DraftModelConfig, initialize_draft_model


class CandidateSelector(nn.Module):
    """Official-style predecessor-conditioned reranker over the real base top-k."""

    def __init__(self, hidden_size: int, vocab_size: int, rank: int = 256, top_k: int = 16) -> None:
        super().__init__()
        if rank < 1 or not 1 <= top_k <= vocab_size:
            raise ValueError("invalid selector rank/top_k")
        self.rank, self.top_k = int(rank), int(top_k)
        self.hidden_size, self.vocab_size = int(hidden_size), int(vocab_size)
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.zeros(vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        nn.init.normal_(self.predecessor_codebook, std=0.02)

    def reset_stabilized_parameters(self, initializer_range: float) -> None:
        nn.init.normal_(self.predecessor_codebook, std=initializer_range)
        nn.init.zeros_(self.successor_codebook)

    def _candidates(self, hidden: torch.Tensor, lm_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if lm_weight.shape != (self.vocab_size, self.hidden_size):
            raise ValueError("LM-head shape differs from selector contract")
        base = F.linear(hidden, lm_weight).float()
        unary, ids = torch.topk(base, self.top_k, dim=-1, sorted=True)
        return unary, ids

    def _scores(self, hidden: torch.Tensor, predecessor_ids: torch.Tensor,
                ids: torch.Tensor, unary: torch.Tensor) -> torch.Tensor:
        if predecessor_ids.shape != hidden.shape[:-1]:
            raise ValueError("predecessor IDs must match token dimensions")
        gate = F.embedding(predecessor_ids, self.predecessor_codebook)
        gate = gate * self.hidden_projection(hidden).to(gate.dtype)
        successors = F.embedding(ids, self.successor_codebook)
        return unary + torch.einsum("...r,...kr->...k", gate, successors).float()

    def forward(self, hidden: torch.Tensor, lm_weight: torch.Tensor,
                predecessor_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unary, ids = self._candidates(hidden, lm_weight)
        return ids, self._scores(hidden, predecessor_ids, ids, unary)

    def training_forward(self, hidden: torch.Tensor, lm_weight: torch.Tensor,
                         predecessor_ids: torch.Tensor, target_ids: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Crucial train/inference invariant: never inject a missing target token.
        unary, ids = self._candidates(hidden, lm_weight)
        hits = ids.eq(target_ids.unsqueeze(-1)).any(-1)
        return ids, self._scores(hidden, predecessor_ids, ids, unary), hits


class DFlash2Model(nn.Module):
    def __init__(self, config: DraftModelConfig, selector_rank: int = 256,
                 selector_top_k: int = 16) -> None:
        super().__init__()
        self.config = config
        self.dflash = DFlashModel(config, dynamic_convolution=True)
        self.selector = CandidateSelector(
            config.hidden_size, config.vocab_size, selector_rank, selector_top_k
        )
        initialize_draft_model(self, config.initializer_range)

    @property
    def layers(self) -> nn.ModuleList:
        return self.dflash.layers

    @property
    def candidate_selector(self) -> CandidateSelector:
        return self.selector

    def training_forward(self, noise: torch.Tensor, context: torch.Tensor, *,
                         lm_weight: torch.Tensor, predecessor_ids: torch.Tensor,
                         target_ids: torch.Tensor, position_ids: torch.Tensor,
                         attention_mask: torch.Tensor, block_size: int):
        hidden = self.dflash(
            noise, context, position_ids=position_ids,
            attention_mask=attention_mask, block_size=block_size,
        )
        ids, scores, hits = self.selector.training_forward(
            hidden, lm_weight, predecessor_ids, target_ids
        )
        return hidden, ids, scores, hits
