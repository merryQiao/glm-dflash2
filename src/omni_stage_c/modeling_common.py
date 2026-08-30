from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DraftModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_aux_layers: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    initializer_range: float = 0.02
    full_attention: bool = True
    sliding_window: int | None = None
    attention_anchor_chunk_size: int = 16
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        values = (
            self.hidden_size, self.intermediate_size, self.num_hidden_layers,
            self.num_attention_heads, self.num_key_value_heads, self.head_dim,
            self.num_aux_layers, self.vocab_size,
        )
        if any(value < 1 for value in values):
            raise ValueError("all draft dimensions must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.head_dim % 2 or sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError("mRoPE sections must cover head_dim / 2")
        if not self.full_attention or self.sliding_window is not None:
            raise ValueError("Omni drafter uses dense full attention")
        if self.attention_anchor_chunk_size < 1:
            raise ValueError("attention anchor chunk size must be positive")

    @classmethod
    def production(cls) -> "DraftModelConfig":
        return cls(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=5,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            num_aux_layers=5,
            vocab_size=152064,
            gradient_checkpointing=True,
        )

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        normalized = value.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(dtype)


def initialize_draft_model(model: nn.Module, initializer_range: float) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=initializer_range)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        for module in model.modules():
            reset = getattr(module, "reset_stabilized_parameters", None)
            if callable(reset):
                reset(initializer_range=initializer_range)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def normalize_mrope_positions(position_ids: torch.Tensor, batch: int, length: int) -> torch.Tensor:
    """Return exact Qwen3-Omni positions as [3,batch,length]."""

    if position_ids.ndim == 2 and position_ids.shape == (batch, length):
        return position_ids.unsqueeze(0).expand(3, -1, -1)
    if position_ids.ndim == 3 and position_ids.shape == (3, batch, length):
        return position_ids
    if position_ids.ndim == 3 and position_ids.shape == (batch, length, 3):
        return position_ids.permute(2, 0, 1)
    raise ValueError(
        f"position_ids must be [3,{batch},{length}] or [{batch},{length},3]"
    )


def interleaved_mrope(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    theta: float,
    sections: tuple[int, int, int],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Transformers Qwen3-Omni interleaved T/H/W rotary layout."""

    axes, batch, length = position_ids.shape
    if axes != 3 or sum(sections) != head_dim // 2:
        raise ValueError("invalid mRoPE positions or sections")
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.einsum("d,abl->abld", inv_freq, position_ids.float())
    mixed = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        index = slice(offset, sections[dim] * 3, 3)
        mixed[..., index] = freqs[dim, ..., index]
    embedding = torch.cat((mixed, mixed), dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def repeat_kv(value: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return value
    batch, heads, length, width = value.shape
    return (
        value[:, :, None, :, :]
        .expand(batch, heads, repetitions, length, width)
        .reshape(batch, heads * repetitions, length, width)
    )


class FullAttention(nn.Module):
    """GQA: draft queries attend to [five-stream target context | draft block]."""

    def __init__(self, config: DraftModelConfig) -> None:
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.query_width, bias=False)
        self.key = nn.Linear(config.hidden_size, config.kv_width, bias=False)
        self.value = nn.Linear(config.hidden_size, config.kv_width, bias=False)
        self.output = nn.Linear(config.query_width, config.hidden_size, bias=False)
        self.query_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.key_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        target_context: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or target_context.ndim != 3:
            raise ValueError("attention inputs must be [batch,tokens,hidden]")
        batch, query_length, _ = hidden.shape
        context_length = target_context.shape[1]
        if attention_mask.ndim != 4:
            raise ValueError("compact block mask must be [B,A,K,K_context+K]")
        mask_batch, anchors, block_size, mask_keys = attention_mask.shape
        if mask_batch != batch or query_length != anchors * block_size:
            raise ValueError("compact attention mask is not aligned to flattened blocks")
        total = context_length + query_length
        positions = normalize_mrope_positions(position_ids, batch, total)
        if mask_keys != context_length + block_size or attention_mask.dtype != torch.bool:
            raise ValueError("compact attention mask has invalid key layout or dtype")

        blocks = hidden.reshape(batch, anchors, block_size, self.config.hidden_size)
        query = self.query(blocks).view(
            batch, anchors, block_size, self.config.num_attention_heads, self.config.head_dim
        ).permute(0, 1, 3, 2, 4)
        local_key = self.key(blocks).view(
            batch, anchors, block_size, self.config.num_key_value_heads, self.config.head_dim
        ).permute(0, 1, 3, 2, 4)
        local_value = self.value(blocks).view(
            batch, anchors, block_size, self.config.num_key_value_heads, self.config.head_dim
        ).permute(0, 1, 3, 2, 4)
        context_key = self.key(target_context).view(
            batch, context_length, self.config.num_key_value_heads, self.config.head_dim
        ).permute(0, 2, 1, 3)
        context_value = self.value(target_context).view(
            batch, context_length, self.config.num_key_value_heads, self.config.head_dim
        ).permute(0, 2, 1, 3)
        query, local_key = self.query_norm(query), self.key_norm(local_key)
        context_key = self.key_norm(context_key)
        cosine, sine = interleaved_mrope(
            positions,
            head_dim=self.config.head_dim,
            theta=self.config.rope_theta,
            sections=self.config.mrope_section,
            dtype=query.dtype,
        )
        context_cos, context_sin = cosine[:, :context_length], sine[:, :context_length]
        draft_cos = cosine[:, context_length:].reshape(batch, anchors, block_size, -1)
        draft_sin = sine[:, context_length:].reshape(batch, anchors, block_size, -1)
        query = query * draft_cos.unsqueeze(2) + rotate_half(query) * draft_sin.unsqueeze(2)
        local_key = (
            local_key * draft_cos.unsqueeze(2)
            + rotate_half(local_key) * draft_sin.unsqueeze(2)
        )
        context_key = (
            context_key * context_cos.unsqueeze(1)
            + rotate_half(context_key) * context_sin.unsqueeze(1)
        )
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        local_key = local_key.repeat_interleave(groups, dim=2)
        local_value = local_value.repeat_interleave(groups, dim=2)
        context_key, context_value = repeat_kv(context_key, groups), repeat_kv(context_value, groups)

        outputs: list[torch.Tensor] = []
        chunk_size = self.config.attention_anchor_chunk_size
        for start in range(0, anchors, chunk_size):
            stop = min(start + chunk_size, anchors)
            count = stop - start
            q = query[:, start:stop].reshape(
                batch * count, self.config.num_attention_heads, block_size, self.config.head_dim
            )
            context_k = context_key[:, None].expand(-1, count, -1, -1, -1).reshape(
                batch * count, self.config.num_attention_heads, context_length, self.config.head_dim
            )
            context_v = context_value[:, None].expand(-1, count, -1, -1, -1).reshape_as(context_k)
            local_k = local_key[:, start:stop].reshape(
                batch * count, self.config.num_attention_heads, block_size, self.config.head_dim
            )
            local_v = local_value[:, start:stop].reshape_as(local_k)
            mask = attention_mask[:, start:stop].reshape(
                batch * count, 1, block_size, context_length + block_size
            )
            outputs.append(F.scaled_dot_product_attention(
                q, torch.cat((context_k, local_k), dim=2),
                torch.cat((context_v, local_v), dim=2),
                attn_mask=mask, dropout_p=0.0, is_causal=False,
            ).reshape(batch, count, self.config.num_attention_heads, block_size,
                      self.config.head_dim))
        attended = torch.cat(outputs, dim=1).permute(0, 1, 3, 2, 4)
        return self.output(attended.reshape(batch, query_length, self.config.query_width))


class GatedMLP(nn.Module):
    def __init__(self, config: DraftModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


def _grouped_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base_kernel: torch.Tensor,
    *,
    group_size: int,
    block_size: int,
) -> torch.Tensor:
    batch, sequence, width = hidden.shape
    if sequence % block_size:
        raise ValueError("block size must divide flattened draft length")
    groups, taps = width // group_size, base_kernel.shape[0]
    blocks = hidden.reshape(batch, sequence // block_size, block_size, groups, group_size)
    dynamic = dynamic.reshape(batch, sequence // block_size, block_size, taps, groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(taps):
        values = blocks if offset == 0 else F.pad(blocks[:, :, :-offset], (0, 0, 0, 0, offset, 0))
        base = base_kernel[offset].reshape(1, 1, 1, groups, group_size)
        output = output + base.to(hidden.dtype) * values
        output = torch.addcmul(output, dynamic[..., offset, :, :], values)
    return output.reshape_as(hidden)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 2, group_size: int = 16) -> None:
        super().__init__()
        if kernel_size != 2 or hidden_size % group_size:
            raise ValueError("DFlash2 requires two taps and a dividing group size")
        self.group_size, self.kernel_size = int(group_size), int(kernel_size)
        groups = hidden_size // group_size
        # DFlash2 applies one learned block-local two-tap convolution before
        # and another after both attention and MLP sublayers.
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

    def reset_stabilized_parameters(self, initializer_range: float) -> None:
        del initializer_range
        self.reset_parameters()

    def prepare(self, hidden: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        prepared = _grouped_convolve(
            hidden, dynamic[..., 0, :, :], self.base_kernel[0],
            group_size=self.group_size, block_size=block_size,
        )
        return prepared, dynamic[..., 1, :, :]

    def finish(self, hidden: torch.Tensor, dynamic: torch.Tensor, block_size: int) -> torch.Tensor:
        return _grouped_convolve(
            hidden, dynamic, self.base_kernel[1],
            group_size=self.group_size, block_size=block_size,
        )

    def forward(self, hidden: torch.Tensor, block_size: int) -> torch.Tensor:
        prepared, dynamic = self.prepare(hidden, block_size)
        return self.finish(prepared, dynamic, block_size)


class DenseDecoderLayer(nn.Module):
    def __init__(self, config: DraftModelConfig, dynamic_convolution: bool = False) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = FullAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config)
        self.attention_conv = GroupedDynamicCausalConv(config.hidden_size) if dynamic_convolution else None
        self.mlp_conv = GroupedDynamicCausalConv(config.hidden_size) if dynamic_convolution else None

    def forward(self, hidden: torch.Tensor, context: torch.Tensor, positions: torch.Tensor,
                mask: torch.Tensor, block_size: int) -> torch.Tensor:
        value = self.input_norm(hidden)
        dynamic = None
        if self.attention_conv is not None:
            value, dynamic = self.attention_conv.prepare(value, block_size)
        value = self.attention(value, context, positions, mask)
        if self.attention_conv is not None:
            value = self.attention_conv.finish(value, dynamic, block_size)
        hidden = hidden + value
        value = self.post_attention_norm(hidden)
        dynamic = None
        if self.mlp_conv is not None:
            value, dynamic = self.mlp_conv.prepare(value, block_size)
        value = self.mlp(value)
        if self.mlp_conv is not None:
            value = self.mlp_conv.finish(value, dynamic, block_size)
        return hidden + value


class DenseDraftBackbone(nn.Module):
    def __init__(self, config: DraftModelConfig, dynamic_convolution: bool = False) -> None:
        super().__init__()
        self.config = config
        self.target_projection = nn.Linear(
            config.num_aux_layers * config.hidden_size, config.hidden_size, bias=False
        )
        self.target_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            DenseDecoderLayer(config, dynamic_convolution=dynamic_convolution)
            for _ in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def project_target_hidden(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 4 or context.shape[-2:] != (
            self.config.num_aux_layers, self.config.hidden_size
        ):
            raise ValueError("target context must be [batch,sequence,5,hidden]")
        return self.target_norm(self.target_projection(context.flatten(-2)))

    def forward(self, noise: torch.Tensor, context: torch.Tensor, *, position_ids: torch.Tensor,
                attention_mask: torch.Tensor, block_size: int) -> torch.Tensor:
        projected = self.project_target_hidden(context)
        hidden = noise
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint

                hidden = checkpoint(
                    layer, hidden, projected, position_ids, attention_mask, block_size,
                    use_reentrant=False,
                )
            else:
                hidden = layer(hidden, projected, position_ids, attention_mask, block_size)
        return self.final_norm(hidden)
