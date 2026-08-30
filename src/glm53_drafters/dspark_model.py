from __future__ import annotations

import torch
from torch import nn

from .dflash_model import DFlashModel
from .modeling_common import DraftModelConfig, initialize_draft_model


def align_predecessors(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim < 2:
        raise ValueError("hidden tensor needs a physical-block dimension")
    result = torch.zeros_like(hidden)
    result[..., 1:, :] = hidden[..., :-1, :]
    return result


def teacher_forced_predecessor_ids(target_ids: torch.Tensor) -> torch.Tensor:
    if target_ids.ndim < 1 or target_ids.shape[-1] < 1:
        raise ValueError("target_ids must have a non-empty block dimension")
    result = target_ids.clone()
    result[..., 1:] = target_ids[..., :-1]
    return result


class LowRankMarkovHead(nn.Module):
    def __init__(self, *, vocab_size: int, rank: int = 256) -> None:
        super().__init__()
        if rank < 1 or vocab_size < 1:
            raise ValueError("Markov rank and vocabulary must be positive")
        self.rank = int(rank)
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)
        nn.init.normal_(self.markov_w1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.markov_w2.weight, mean=0.0, std=0.02)

    def features(self, predecessor_token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(predecessor_token_ids.long())

    def forward(self, predecessor_token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.features(predecessor_token_ids))


class MarkovConfidenceHead(nn.Module):
    def __init__(self, *, hidden_size: int, rank: int = 256) -> None:
        super().__init__()
        self.output = nn.Linear(hidden_size + rank, 1, bias=True)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden: torch.Tensor, markov_features: torch.Tensor) -> torch.Tensor:
        if hidden.shape[:-1] != markov_features.shape[:-1]:
            raise ValueError("confidence token dimensions must be equal")
        return self.output(torch.cat((hidden, markov_features), -1)).squeeze(-1)


class DSparkModel(nn.Module):
    def __init__(self, config: DraftModelConfig, *, markov_rank: int = 256) -> None:
        super().__init__()
        self.config = config
        self.dflash = DFlashModel(config)
        self.markov = LowRankMarkovHead(vocab_size=config.vocab_size, rank=markov_rank)
        self.confidence = MarkovConfidenceHead(
            hidden_size=config.hidden_size, rank=markov_rank
        )
        initialize_draft_model(self, initializer_range=config.initializer_range)

    @property
    def layers(self) -> nn.ModuleList:
        return self.dflash.layers

    @property
    def markov_head(self) -> LowRankMarkovHead:
        return self.markov

    @property
    def confidence_head(self) -> MarkovConfidenceHead:
        return self.confidence

    def forward(self, noise_embedding: torch.Tensor, target_context: torch.Tensor, *,
                predecessor_token_ids: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: torch.Tensor, block_size: int
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.dflash(
            noise_embedding,
            target_context,
            position_ids=position_ids,
            attention_mask=attention_mask,
            block_size=block_size,
        )
        if predecessor_token_ids.shape != hidden.shape[:-1]:
            raise ValueError("predecessor token IDs must match draft token dimensions")
        features = self.markov.features(predecessor_token_ids)
        return hidden, self.markov.markov_w2(features), self.confidence(hidden, features)
