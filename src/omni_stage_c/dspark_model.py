from __future__ import annotations

import torch
from torch import nn

from .dflash_model import DFlashModel
from .modeling_common import DraftModelConfig, initialize_draft_model


def teacher_forced_predecessor_ids(target_ids: torch.Tensor) -> torch.Tensor:
    result = target_ids.clone()
    result[..., 1:] = target_ids[..., :-1]
    return result


class LowRankMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int = 256) -> None:
        super().__init__()
        self.rank = int(rank)
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(ids.long())


class DSparkModel(nn.Module):
    def __init__(self, config: DraftModelConfig, markov_rank: int = 256) -> None:
        super().__init__()
        self.config = config
        self.dflash = DFlashModel(config)
        self.markov = LowRankMarkovHead(config.vocab_size, markov_rank)
        self.confidence = nn.Linear(config.hidden_size + markov_rank, 1)
        initialize_draft_model(self, config.initializer_range)

    @property
    def layers(self) -> nn.ModuleList:
        return self.dflash.layers

    @property
    def markov_head(self) -> LowRankMarkovHead:
        return self.markov

    @property
    def confidence_head(self) -> nn.Linear:
        return self.confidence

    def forward(self, noise: torch.Tensor, context: torch.Tensor, *,
                predecessor_token_ids: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: torch.Tensor, block_size: int):
        hidden = self.dflash(
            noise, context, position_ids=position_ids,
            attention_mask=attention_mask, block_size=block_size,
        )
        features = self.markov.features(predecessor_token_ids)
        residual = self.markov.markov_w2(features)
        confidence = self.confidence(torch.cat((hidden, features), -1)).squeeze(-1)
        return hidden, residual, confidence
