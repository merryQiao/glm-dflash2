from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .draft_backbone import DFlashDraftModel


class LowRankMarkovHead(nn.Module):
    """Vanilla predecessor-token low-rank vocabulary bias used by DSpark."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        rank: int = 256,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if vocab_size < 1 or hidden_size < 1 or rank < 1:
            raise ValueError("vocab_size, hidden_size, and rank must be positive")
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.markov_head_type = "vanilla"
        self.markov_w1 = nn.Embedding(self.vocab_size, self.rank)
        self.markov_w2 = nn.Linear(self.rank, self.vocab_size, bias=False)
        self.reset_parameters(float(initializer_range))

    def reset_parameters(self, initializer_range: float = 0.02) -> None:
        nn.init.normal_(self.markov_w1.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.markov_w2.weight, mean=0.0, std=initializer_range)

    def get_prev_embeddings(self, predecessor_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(predecessor_ids.to(torch.long))

    def score_chunk(
        self,
        hidden: torch.Tensor,
        predecessor_ids: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
    ) -> torch.Tensor:
        if not 0 <= int(vocab_start) < int(vocab_end) <= self.vocab_size:
            raise ValueError("invalid vocabulary chunk")
        if predecessor_ids.shape != hidden.shape[:-1]:
            raise ValueError("predecessor_ids must match hidden token dimensions")
        predecessor = self.get_prev_embeddings(predecessor_ids)
        successor = self.markov_w2.weight[int(vocab_start) : int(vocab_end)]
        return torch.einsum("...r,vr->...v", predecessor, successor)

    def forward(
        self,
        hidden: torch.Tensor,
        predecessor_ids: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
    ) -> torch.Tensor:
        """Score one vocabulary chunk through ``nn.Module.__call__``.

        This is deliberately a real ``forward`` so FSDP2 hooks can materialize
        the head when it is invoked after the common backbone forward.
        """

        return self.score_chunk(hidden, predecessor_ids, vocab_start, vocab_end)


class DSparkDraftModel(DFlashDraftModel):
    """Common plain GLM draft backbone with DSpark method heads."""

    def __init__(self, config: Qwen3Config, *, markov_rank: int = 256) -> None:
        super().__init__(config)
        self.markov_head = LowRankMarkovHead(
            config.vocab_size,
            config.hidden_size,
            rank=markov_rank,
            initializer_range=config.initializer_range,
        )
        self.confidence_head = nn.Linear(
            config.hidden_size + int(markov_rank), 1, bias=True
        )
        nn.init.normal_(
            self.confidence_head.weight,
            mean=0.0,
            std=float(config.initializer_range),
        )
        nn.init.zeros_(self.confidence_head.bias)
        config.architectures = ["DSparkDraftModel"]
        config.dspark_config = {
            "markov_rank": int(markov_rank),
            "markov_head_type": "vanilla",
            "confidence_head_with_markov": True,
        }

    def confidence_logits(
        self, hidden_states: torch.Tensor, predecessor_ids: torch.Tensor
    ) -> torch.Tensor:
        if predecessor_ids.shape != hidden_states.shape[:-1]:
            raise ValueError("predecessor_ids must match hidden token dimensions")
        markov_embedding = self.markov_head.get_prev_embeddings(predecessor_ids).to(
            hidden_states.dtype
        )
        features = torch.cat((hidden_states, markov_embedding), dim=-1)
        return self.confidence_head(features).squeeze(-1)
