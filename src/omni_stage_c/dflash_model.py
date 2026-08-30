from __future__ import annotations

import torch
from torch import nn

from .modeling_common import DenseDraftBackbone, DraftModelConfig, initialize_draft_model


class DFlashModel(nn.Module):
    """Embedding/head-free five-layer Thinker DFlash backbone."""

    def __init__(self, config: DraftModelConfig, *, dynamic_convolution: bool = False) -> None:
        super().__init__()
        self.config = config
        self.backbone = DenseDraftBackbone(config, dynamic_convolution=dynamic_convolution)
        initialize_draft_model(self, config.initializer_range)

    @property
    def layers(self) -> nn.ModuleList:
        return self.backbone.layers

    def forward(self, noise_embedding: torch.Tensor, target_context: torch.Tensor, *,
                position_ids: torch.Tensor, attention_mask: torch.Tensor,
                block_size: int) -> torch.Tensor:
        return self.backbone(
            noise_embedding,
            target_context,
            position_ids=position_ids,
            attention_mask=attention_mask,
            block_size=block_size,
        )
