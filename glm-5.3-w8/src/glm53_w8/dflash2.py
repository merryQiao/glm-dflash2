from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import SlidingBlocks
from .modeling import DraftModelConfig, SlidingDraftBackbone


class CandidateSelector(nn.Module):
    def __init__(self, config: DraftModelConfig) -> None:
        super().__init__()
        rank = config.selector_rank
        self.top_k = config.selector_top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(config.vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(config.vocab_size, rank))
        self.hidden_projection = nn.Linear(config.hidden_size, rank, bias=False)
        nn.init.normal_(self.predecessor_codebook, std=config.initializer_range)
        nn.init.zeros_(self.successor_codebook)

    def forward(
        self,
        hidden: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        gate = F.embedding(predecessor_ids, self.predecessor_codebook)
        gate = gate * self.hidden_projection(hidden)
        successor = F.embedding(candidate_ids, self.successor_codebook)
        return unary + torch.einsum("...r,...kr->...k", gate, successor).to(unary.dtype)


class DFlash2Model(nn.Module):
    def __init__(self, config: DraftModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = SlidingDraftBackbone(config, dynamic_convolution=True)
        self.candidate_selector = CandidateSelector(config)

    @property
    def layers(self) -> nn.ModuleList:
        return self.backbone.layers

    def forward(
        self,
        *,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        blocks: SlidingBlocks,
    ) -> torch.Tensor:
        return self.backbone(
            noise_embedding=noise_embedding, target_hidden=target_hidden, blocks=blocks
        )

    def selector_scores(
        self,
        hidden: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.candidate_selector(hidden, unary, candidate_ids, predecessor_ids)
