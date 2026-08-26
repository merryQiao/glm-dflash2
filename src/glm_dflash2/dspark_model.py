from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .draft_backbone import DFlashDraftModel


class LowRankMarkovHead(nn.Module):
    """Predecessor-conditioned low-rank bias over the frozen LM-head logits."""

    def __init__(self, vocab_size: int, hidden_size: int, rank: int = 256) -> None:
        super().__init__()
        if vocab_size < 1 or hidden_size < 1 or rank < 1:
            raise ValueError("vocab_size, hidden_size, and rank must be positive")
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.predecessor_codebook, mean=0.0, std=0.02)
        nn.init.normal_(self.hidden_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.successor_codebook)

    def _gate(self, hidden: torch.Tensor, predecessor_ids: torch.Tensor) -> torch.Tensor:
        if predecessor_ids.shape != hidden.shape[:-1]:
            raise ValueError("predecessor_ids must match hidden token dimensions")
        predecessor = F.embedding(predecessor_ids, self.predecessor_codebook)
        return predecessor * self.hidden_projection(hidden)

    def score_chunk(
        self,
        hidden: torch.Tensor,
        predecessor_ids: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
    ) -> torch.Tensor:
        if not 0 <= int(vocab_start) < int(vocab_end) <= self.vocab_size:
            raise ValueError("invalid vocabulary chunk")
        gate = self._gate(hidden, predecessor_ids)
        successor = self.successor_codebook[int(vocab_start) : int(vocab_end)]
        return torch.einsum("...r,vr->...v", gate, successor)

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
            config.vocab_size, config.hidden_size, rank=markov_rank
        )
        self.confidence_head = nn.Linear(config.hidden_size, 1, bias=True)
        config.architectures = ["DSparkDraftModel"]
        config.dspark_config = {"markov_rank": int(markov_rank)}

    def confidence_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.confidence_head(hidden_states).squeeze(-1)
