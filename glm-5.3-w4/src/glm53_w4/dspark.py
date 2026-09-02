from __future__ import annotations

import torch
from torch import nn

from .blocks import SlidingBlocks
from .modeling import DraftModelConfig, SlidingDraftBackbone


class LowRankMarkovHead(nn.Module):
    def __init__(self, config: DraftModelConfig) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.w1 = nn.Embedding(config.vocab_size, config.markov_rank)
        self.w2 = nn.Linear(config.markov_rank, config.vocab_size, bias=False)
        nn.init.normal_(self.w1.weight, std=config.initializer_range)
        nn.init.normal_(self.w2.weight, std=config.initializer_range)

    def forward(
        self,
        predecessor_ids: torch.Tensor,
        vocab_start: int | None = None,
        vocab_end: int | None = None,
    ) -> torch.Tensor:
        features = self.w1(predecessor_ids.to(torch.long))
        if vocab_start is None and vocab_end is None:
            return features
        if vocab_start is None or vocab_end is None:
            raise ValueError("both vocabulary bounds are required")
        if not 0 <= int(vocab_start) < int(vocab_end) <= self.vocab_size:
            raise ValueError("invalid vocabulary chunk")
        return torch.einsum(
            "...r,vr->...v", features, self.w2.weight[int(vocab_start) : int(vocab_end)]
        )


class DSparkModel(nn.Module):
    def __init__(self, config: DraftModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = SlidingDraftBackbone(config, dynamic_convolution=False)
        self.markov_head = LowRankMarkovHead(config)
        self.confidence_head = nn.Linear(
            config.hidden_size + config.markov_rank, 1, bias=True
        )
        nn.init.normal_(self.confidence_head.weight, std=config.initializer_range)
        nn.init.zeros_(self.confidence_head.bias)

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

    def markov_scores(
        self, predecessor_ids: torch.Tensor, vocab_start: int, vocab_end: int
    ) -> torch.Tensor:
        return self.markov_head(predecessor_ids, vocab_start, vocab_end)

    def confidence_logits(
        self, hidden: torch.Tensor, predecessor_ids: torch.Tensor
    ) -> torch.Tensor:
        features = self.markov_head(predecessor_ids).to(hidden.dtype)
        return self.confidence_head(torch.cat((hidden, features), dim=-1)).squeeze(-1)
