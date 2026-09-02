from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import SlidingBlocks


@dataclass(frozen=True)
class DraftModelConfig:
    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288
    num_hidden_layers: int = 5
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    head_dim: int = 64
    target_layer_ids: tuple[int, ...] = (1, 20, 38, 56, 75)
    block_size: int = 8
    mask_token_id: int = 0
    sliding_window: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 8_000_000.0
    initializer_range: float = 0.02
    selector_top_k: int = 16
    selector_rank: int = 256
    markov_rank: int = 256
    conv_kernel_size: int = 2
    conv_group_size: int = 16
    anchor_chunk_size: int = 8
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if len(self.target_layer_ids) != self.num_hidden_layers:
            raise ValueError("one target hidden stream is required per draft layer")
        if len(set(self.target_layer_ids)) != len(self.target_layer_ids) or any(
            int(layer) < 0 for layer in self.target_layer_ids
        ):
            raise ValueError("target layer IDs must be unique and non-negative")
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError(
                "formal GLM-5.3 drafter uses equal Q/KV heads (64/64); "
                "GQA would require a different runtime contract"
            )
        if self.conv_group_size < 1 or self.hidden_size % self.conv_group_size:
            raise ValueError("conv_group_size must divide hidden_size")
        if (
            self.block_size < 2
            or self.sliding_window < 1
            or self.anchor_chunk_size < 1
            or self.conv_kernel_size < 1
            or self.rms_norm_eps <= 0
            or self.rope_theta <= 0
        ):
            raise ValueError("invalid block/window/anchor chunk size")
        if self.vocab_size < 1 or self.selector_top_k < 1 or self.selector_top_k > self.vocab_size:
            raise ValueError("selector_top_k must be within the vocabulary")
        if self.selector_rank < 1 or self.markov_rank < 1:
            raise ValueError("selector/Markov ranks must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_layer_ids"] = list(self.target_layer_ids)
        result["layer_types"] = ["sliding_attention"] * self.num_hidden_layers
        result["use_sliding_window"] = True
        result["sample_from_anchor"] = False
        result["num_speculative_tokens"] = self.block_size - 1
        result["architectures"] = ["GLM53W8DraftModel"]
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DraftModelConfig":
        allowed = set(cls.__dataclass_fields__)
        data = {key: item for key, item in value.items() if key in allowed}
        if "target_layer_ids" in data:
            data["target_layer_ids"] = tuple(int(item) for item in data["target_layer_ids"])
        return cls(**data)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.theta = float(theta)
        inv = 1.0 / (
            self.theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequency = position_ids.float().unsqueeze(-1) * self.inv_freq.to(
            position_ids.device
        ).view(1, 1, -1)
        embedding = torch.cat((frequency, frequency), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class SlidingAttention(nn.Module):
    def __init__(self, config: DraftModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        local_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        context_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, local_tokens, _ = local_hidden.shape
        context_tokens = context_hidden.shape[1]
        query = self.q_proj(local_hidden).view(
            batch, local_tokens, self.num_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        key_value = torch.cat((context_hidden, local_hidden), dim=1)
        key = self.k_proj(key_value).view(
            batch, context_tokens + local_tokens, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(key_value).view_as(key)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        full_cos, full_sin = cos.unsqueeze(1), sin.unsqueeze(1)
        query_cos, query_sin = full_cos[..., -local_tokens:, :], full_sin[..., -local_tokens:, :]
        query = query * query_cos + _rotate_half(query) * query_sin
        key = key * full_cos + _rotate_half(key) * full_sin
        visible = torch.cat(
            (
                context_mask,
                torch.ones(
                    (batch, local_tokens), device=local_hidden.device, dtype=torch.bool
                ),
            ),
            dim=-1,
        )
        visible = visible[:, None, None, :].expand(
            batch, 1, local_tokens, context_tokens + local_tokens
        )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=visible,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch, local_tokens, -1))


class MLP(nn.Module):
    def __init__(self, config: DraftModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


def _grouped_convolution(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    batch, tokens, width = hidden.shape
    groups = width // group_size
    kernel_size = base.shape[0]
    grouped = hidden.reshape(batch, tokens, groups, group_size)
    result = torch.zeros_like(grouped)
    for offset in range(kernel_size):
        shifted = (
            grouped
            if offset == 0
            else F.pad(grouped[:, :-offset], (0, 0, 0, 0, offset, 0))
        )
        kernel = base[offset].reshape(1, 1, groups, group_size).to(hidden.dtype)
        result = result + kernel * shifted
        result = torch.addcmul(result, dynamic[:, :, offset, :, None], shifted)
    return result.reshape(batch, tokens, width)


class DynamicCausalConv(nn.Module):
    def __init__(self, config: DraftModelConfig) -> None:
        super().__init__()
        groups = config.hidden_size // config.conv_group_size
        self.kernel_size = config.conv_kernel_size
        self.group_size = config.conv_group_size
        self.base = nn.Parameter(torch.zeros(2, self.kernel_size, config.hidden_size))
        self.projection = nn.Linear(
            config.hidden_size, 2 * self.kernel_size * groups, bias=False
        )
        nn.init.zeros_(self.projection.weight)
        with torch.no_grad():
            self.base[:, 0].fill_(1.0)

    def prepare(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        return (
            _grouped_convolution(hidden, dynamic[..., 0, :, :], self.base[0], self.group_size),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        return _grouped_convolution(hidden, dynamic, self.base[1], self.group_size)


class DecoderLayer(nn.Module):
    def __init__(self, config: DraftModelConfig, *, dynamic_convolution: bool) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = SlidingAttention(config)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)
        self.attention_conv = DynamicCausalConv(config) if dynamic_convolution else None
        self.mlp_conv = DynamicCausalConv(config) if dynamic_convolution else None

    def forward(
        self,
        local: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = local
        value = self.input_norm(local)
        dynamic = None
        if self.attention_conv is not None:
            value, dynamic = self.attention_conv.prepare(value)
        value = self.attention(value, context, context_mask, cos, sin)
        if self.attention_conv is not None:
            value = self.attention_conv.finish(value, dynamic)
        local = residual + value
        residual = local
        value = self.post_norm(local)
        dynamic = None
        if self.mlp_conv is not None:
            value, dynamic = self.mlp_conv.prepare(value)
        value = self.mlp(value)
        if self.mlp_conv is not None:
            value = self.mlp_conv.finish(value, dynamic)
        return residual + value


def gather_context_halo(
    projected_context: torch.Tensor,
    blocks: SlidingBlocks,
    *,
    anchor_start: int,
    anchor_end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, tokens, width = projected_context.shape
    indices = blocks.context_indices[:, anchor_start:anchor_end]
    anchors = indices.shape[1]
    window = indices.shape[2]
    expanded = projected_context[:, None].expand(batch, anchors, tokens, width)
    context = expanded.gather(2, indices[..., None].expand(batch, anchors, window, width))
    return (
        context.reshape(batch * anchors, window, width),
        blocks.context_mask[:, anchor_start:anchor_end].reshape(batch * anchors, window),
        blocks.context_position_ids[:, anchor_start:anchor_end].reshape(
            batch * anchors, window
        ),
    )


class SlidingDraftBackbone(nn.Module):
    def __init__(self, config: DraftModelConfig, *, dynamic_convolution: bool) -> None:
        super().__init__()
        self.config = config
        self.target_projection = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.target_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, dynamic_convolution=dynamic_convolution)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta)

    def project_target_hidden(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 4:
            value = value.flatten(-2)
        expected = len(self.config.target_layer_ids) * self.config.hidden_size
        if value.ndim != 3 or value.shape[-1] != expected:
            raise ValueError(f"target_hidden must have final width {expected}")
        return self.target_norm(self.target_projection(value))

    def forward(
        self,
        *,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        blocks: SlidingBlocks,
    ) -> torch.Tensor:
        if noise_embedding.ndim != 4:
            raise ValueError("noise_embedding must have shape [batch, anchors, block, hidden]")
        batch, anchors, block_size, width = noise_embedding.shape
        if block_size != self.config.block_size or width != self.config.hidden_size:
            raise ValueError("noise embedding differs from draft config")
        if blocks.context_indices.shape[-1] != self.config.sliding_window:
            raise ValueError("block halo differs from configured sliding window")
        expected_local = blocks.block_keep_mask[..., None, None].expand_as(
            blocks.local_visibility
        )
        if not torch.equal(blocks.local_visibility, expected_local):
            raise ValueError("draft runtime only supports fully visible local blocks")
        projected = self.project_target_hidden(target_hidden)
        outputs: list[torch.Tensor] = []
        for start in range(0, anchors, self.config.anchor_chunk_size):
            end = min(anchors, start + self.config.anchor_chunk_size)
            count = end - start
            context, context_mask, context_positions = gather_context_halo(
                projected, blocks, anchor_start=start, anchor_end=end
            )
            local = noise_embedding[:, start:end].reshape(
                batch * count, block_size, width
            )
            draft_positions = blocks.draft_position_ids[:, start:end].reshape(
                batch * count, block_size
            )
            positions = torch.cat((context_positions, draft_positions), dim=-1)
            cos, sin = self.rotary(positions, local.dtype)
            for layer in self.layers:
                if self.training and self.config.gradient_checkpointing:
                    local = checkpoint(
                        layer,
                        local,
                        context,
                        context_mask,
                        cos,
                        sin,
                        use_reentrant=False,
                    )
                else:
                    local = layer(local, context, context_mask, cos, sin)
            outputs.append(self.norm(local).reshape(batch, count, block_size, width))
        return torch.cat(outputs, dim=1)
