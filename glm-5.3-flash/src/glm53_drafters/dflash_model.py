from __future__ import annotations

import torch
from torch import nn

from .modeling_common import (
    DenseDraftBackbone,
    DraftModelConfig,
    RMSNorm,
    initialize_draft_model,
)


class DFlashModel(nn.Module):
    """Embedding/head-free official DFlash full-context draft backbone."""

    def __init__(
        self,
        config: DraftModelConfig,
        *,
        dynamic_convolution: bool = False,
        convolution_group_size: int = 16,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = DenseDraftBackbone(
            config,
            dynamic_convolution=dynamic_convolution,
            convolution_group_size=convolution_group_size,
        )
        initialize_draft_model(self, initializer_range=config.initializer_range)

    @property
    def target_projection(self) -> nn.Linear:
        return self.backbone.target_projection

    @property
    def target_norm(self) -> RMSNorm:
        return self.backbone.target_norm

    @property
    def layers(self) -> nn.ModuleList:
        return self.backbone.layers

    def forward(
        self,
        noise_embedding: torch.Tensor,
        target_context: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        return self.backbone(
            noise_embedding,
            target_context,
            position_ids=position_ids,
            attention_mask=attention_mask,
            block_size=block_size,
        )
