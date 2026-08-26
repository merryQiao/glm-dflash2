from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .glm_draft_config import GLM52_DRAFT_SPEC


GLM52_TARGET_LAYER_IDS = [1, 20, 38, 56, 75]


def build_dflash2_config(
    *,
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    target_layer_ids: Sequence[int],
    num_target_layers: int,
    block_size: int,
    mask_token_id: int,
    conv_group_size: int,
    selector_rank: int,
    selector_top_k: int,
    sliding_window: int | None,
    conv_kernel_size: int = 2,
    rms_norm_eps: float = 1e-5,
    rope_theta: float = 8_000_000.0,
    max_position_embeddings: int = 131072,
) -> Qwen3Config:
    if len(target_layer_ids) < 1:
        raise ValueError("target_layer_ids cannot be empty")
    dflash_config = {
        "block_size": int(block_size),
        "mask_token_id": int(mask_token_id),
        "target_layer_ids": [int(value) for value in target_layer_ids],
        "num_target_layers": int(num_target_layers),
        "conv_kernel_size": int(conv_kernel_size),
        "conv_group_size": int(conv_group_size),
        "selector_rank": int(selector_rank),
        "selector_top_k": int(selector_top_k),
    }
    use_sliding_window = sliding_window is not None
    config = Qwen3Config(
        vocab_size=int(vocab_size),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_hidden_layers=int(num_hidden_layers),
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        head_dim=int(head_dim),
        hidden_act="silu",
        max_position_embeddings=int(max_position_embeddings),
        rms_norm_eps=float(rms_norm_eps),
        rope_theta=float(rope_theta),
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=True,
        use_sliding_window=use_sliding_window,
        sliding_window=None if sliding_window is None else int(sliding_window),
        max_window_layers=int(num_hidden_layers) if use_sliding_window else 0,
        layer_types=[
            "sliding_attention" if use_sliding_window else "full_attention"
        ]
        * int(num_hidden_layers),
    )
    config.architectures = ["DFlash2DraftModel"]
    config.is_causal = False
    config.block_size = int(block_size)
    config.num_target_layers = int(num_target_layers)
    config.dflash_config = dflash_config
    config._attn_implementation = "sdpa"
    return config


def build_glm52_dflash2_config(*, vocab_size: int, mask_token_id: int) -> Qwen3Config:
    spec = GLM52_DRAFT_SPEC
    return build_dflash2_config(
        vocab_size=vocab_size,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        num_key_value_heads=spec.num_key_value_heads,
        head_dim=spec.head_dim,
        target_layer_ids=spec.target_layer_ids,
        num_target_layers=spec.target_num_hidden_layers,
        block_size=spec.block_size,
        mask_token_id=mask_token_id,
        conv_group_size=16,
        selector_rank=256,
        selector_top_k=16,
        sliding_window=spec.sliding_window,
        rms_norm_eps=spec.rms_norm_eps,
        rope_theta=spec.rope_theta,
        max_position_embeddings=1_048_576,
    )


class DFlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        value = hidden_states.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight.float() * value).to(dtype)


class DFlashMLP(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DFlashRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.theta = float(theta)
        self.register_buffer("inv_freq", self._fresh_inv_freq(), persistent=False)

    def _fresh_inv_freq(self) -> torch.Tensor:
        return 1.0 / (
            self.theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )

    def _apply(self, fn, recurse: bool = True):
        module = super()._apply(fn, recurse=recurse)
        if self.inv_freq.dtype != torch.float32:
            self.inv_freq = self._fresh_inv_freq().to(device=self.inv_freq.device)
        return module

    def forward(
        self, reference: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv = self.inv_freq.to(device=position_ids.device)
        with torch.autocast(device_type=reference.device.type, enabled=False):
            frequencies = position_ids.float().unsqueeze(-1) * inv.view(1, 1, -1)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            cos, sin = embedding.cos(), embedding.sin()
        return cos.to(reference.dtype), sin.to(reference.dtype)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_length = query.shape[-2]
    query = query * cos[..., -query_length:, :] + _rotate_half(query) * sin[..., -query_length:, :]
    key = key * cos + _rotate_half(key) * sin
    return query, key


class DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        self.head_dim = int(config.head_dim)
        self.scaling = self.head_dim**-0.5
        self.dropout = float(config.attention_dropout)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = DFlashRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = DFlashRMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, query_length, _ = hidden_states.shape
        context_length = target_hidden.shape[1]
        query = self.q_proj(hidden_states).view(batch, query_length, self.num_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)
        key_states = torch.cat((target_hidden, hidden_states), dim=1)
        key = self.k_proj(key_states).view(
            batch, context_length + query_length, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(key_states).view(
            batch, context_length + query_length, self.num_kv_heads, self.head_dim
        )
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = _apply_rotary(query, key, *position_embeddings)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=query.dtype)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scaling,
            enable_gqa=True,
        )
        output = output.transpose(1, 2).contiguous().view(batch, query_length, -1)
        return self.o_proj(output)


def _grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base_kernel: torch.Tensor,
    group_size: int,
    block_size: int,
) -> torch.Tensor:
    batch, sequence, hidden_size = hidden.shape
    if sequence % block_size:
        raise ValueError("conv block_size must divide the draft sequence length")
    groups = hidden_size // group_size
    kernel_size = base_kernel.shape[0]
    blocks = hidden.reshape(batch, sequence // block_size, block_size, groups, group_size)
    dynamic = dynamic.reshape(
        batch, sequence // block_size, block_size, kernel_size, groups, 1
    )
    output = torch.zeros_like(blocks)
    for offset in range(kernel_size):
        values = (
            blocks
            if offset == 0
            else F.pad(blocks[:, :, :-offset], (0, 0, 0, 0, offset, 0))
        )
        kernel = base_kernel[offset].reshape(1, 1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, :, offset], values)
    return output.reshape(batch, sequence, hidden_size)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, group_size: int) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(f"group_size={group_size} must divide hidden_size={hidden_size}")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = int(kernel_size)
        self.group_size = int(group_size)
        groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.zeros(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * kernel_size * groups, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.kernel_projection.weight)
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0].fill_(1.0)

    def prepare(
        self, hidden: torch.Tensor, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        convolved = _grouped_dynamic_convolve(
            hidden,
            dynamic[..., 0, :, :],
            self.base_kernel[0],
            self.group_size,
            block_size,
        )
        return convolved, dynamic[..., 1, :, :]

    def finish(
        self, hidden: torch.Tensor, dynamic: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        return _grouped_dynamic_convolve(
            hidden, dynamic, self.base_kernel[1], self.group_size, block_size
        )


class CandidateSelector(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_size: int, rank: int, top_k: int
    ) -> None:
        super().__init__()
        if rank < 1 or not 1 <= top_k <= vocab_size:
            raise ValueError("invalid selector rank or top_k")
        self.rank = int(rank)
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.predecessor_codebook.normal_(mean=0.0, std=0.02)
            self.successor_codebook.zero_()

    def forward(
        self,
        hidden: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        gate = F.embedding(predecessor_ids, self.predecessor_codebook)
        gate = gate * self.hidden_projection(hidden)
        successors = F.embedding(candidate_ids, self.successor_codebook)
        pairwise = torch.einsum("...r,...kr->...k", gate, successors)
        return unary + pairwise.to(unary.dtype)

    def pair_scores(
        self,
        hidden: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(hidden, unary, candidate_ids, predecessor_ids)


class DFlash2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        dflash = config.dflash_config
        self.input_layernorm = DFlashRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = DFlashAttention(config, layer_idx)
        self.post_attention_layernorm = DFlashRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DFlashMLP(config)
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size, dflash["conv_kernel_size"], dflash["conv_group_size"]
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size, dflash["conv_kernel_size"], dflash["conv_group_size"]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        conv_block_size: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, dynamic = self.attention_conv.prepare(hidden_states, conv_block_size)
        hidden_states = self.self_attn(
            hidden_states, target_hidden, position_embeddings, attention_mask
        )
        hidden_states = self.attention_conv.finish(hidden_states, dynamic, conv_block_size)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, dynamic = self.mlp_conv.prepare(hidden_states, conv_block_size)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, dynamic, conv_block_size)
        return residual + hidden_states


class Qwen3DFlash2DraftModel(nn.Module):
    """Five-layer Qwen3-shaped DFlash2 draft with no embedding or LM head."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        # Imported lazily because draft_backbone reuses the layer primitives
        # defined in this module.
        from .draft_backbone import GLMDraftBackbone

        self.config = config
        dflash = config.dflash_config
        self.block_size = int(dflash["block_size"])
        self.mask_token_id = int(dflash["mask_token_id"])
        self.target_layer_ids = tuple(int(value) for value in dflash["target_layer_ids"])
        self.backbone = GLMDraftBackbone(config, dynamic_convolution=True)
        self.candidate_selector = CandidateSelector(
            config.vocab_size,
            config.hidden_size,
            int(dflash["selector_rank"]),
            int(dflash["selector_top_k"]),
        )

    @property
    def layers(self) -> nn.ModuleList:
        return self.backbone.layers

    @property
    def norm(self) -> DFlashRMSNorm:
        return self.backbone.norm

    @property
    def rotary_emb(self) -> DFlashRotaryEmbedding:
        return self.backbone.rotary_emb

    @property
    def fc(self) -> nn.Linear:
        return self.backbone.fc

    @property
    def hidden_norm(self) -> DFlashRMSNorm:
        return self.backbone.hidden_norm

    def resolve_conv_block_size(self, query_length: int, explicit: int | None) -> int:
        block_size = self.block_size if explicit is None else int(explicit)
        if block_size < 1 or query_length % block_size:
            raise ValueError("conv_block_size must be positive and divide query length")
        return block_size

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.backbone.project_target_hidden(target_hidden)

    def forward(
        self,
        *,
        position_ids: torch.Tensor,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        conv_block_size: int | None = None,
    ) -> torch.Tensor:
        if noise_embedding.ndim != 3:
            raise ValueError("noise_embedding must have shape [batch, draft, hidden]")
        block_size = self.resolve_conv_block_size(noise_embedding.shape[1], conv_block_size)
        return self.backbone(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            conv_block_size=block_size,
        )
