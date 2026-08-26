from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .dflash2_model import (
    DFlash2DecoderLayer,
    DFlashAttention,
    DFlashMLP,
    DFlashRMSNorm,
    DFlashRotaryEmbedding,
)


class DFlashDecoderLayer(nn.Module):
    """A plain DFlash layer using the common GLM full-attention backbone."""

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = DFlashRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = DFlashAttention(config, layer_idx)
        self.post_attention_layernorm = DFlashRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = DFlashMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        conv_block_size: int,
    ) -> torch.Tensor:
        del conv_block_size
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            target_hidden,
            position_embeddings,
            attention_mask,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class GLMDraftBackbone(nn.Module):
    """Embedding/head-free five-layer draft backbone shared by all methods."""

    def __init__(self, config: Qwen3Config, *, dynamic_convolution: bool) -> None:
        super().__init__()
        self.config = config
        layer_type = DFlash2DecoderLayer if dynamic_convolution else DFlashDecoderLayer
        self.layers = nn.ModuleList(
            [layer_type(config, index) for index in range(config.num_hidden_layers)]
        )
        self.norm = DFlashRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = DFlashRotaryEmbedding(config.head_dim, config.rope_theta)
        layer_ids = tuple(config.dflash_config["target_layer_ids"])
        self.fc = nn.Linear(
            len(layer_ids) * config.hidden_size, config.hidden_size, bias=False
        )
        self.hidden_norm = DFlashRMSNorm(config.hidden_size, config.rms_norm_eps)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        if target_hidden.ndim != 3 or target_hidden.shape[-1] != self.fc.in_features:
            raise ValueError(
                "target_hidden must have shape "
                f"[batch, tokens, {self.fc.in_features}], got {tuple(target_hidden.shape)}"
            )
        return self.hidden_norm(self.fc(target_hidden))

    def forward(
        self,
        *,
        position_ids: torch.Tensor,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        conv_block_size: int,
    ) -> torch.Tensor:
        projected_target = self.project_target_hidden(target_hidden)
        expected_positions = projected_target.shape[1] + noise_embedding.shape[1]
        if position_ids.shape != (noise_embedding.shape[0], expected_positions):
            raise ValueError("position_ids do not cover [context | noise]")
        position_embeddings = self.rotary_emb(noise_embedding, position_ids)
        hidden_states = noise_embedding
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                projected_target,
                attention_mask,
                position_embeddings,
                conv_block_size,
            )
        return self.norm(hidden_states)


class DFlashDraftModel(GLMDraftBackbone):
    """Plain DFlash consumer of the common five-layer GLM backbone."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config, dynamic_convolution=False)
        dflash = config.dflash_config
        self.block_size = int(dflash["block_size"])
        self.mask_token_id = int(dflash["mask_token_id"])
        self.target_layer_ids = tuple(int(value) for value in dflash["target_layer_ids"])
