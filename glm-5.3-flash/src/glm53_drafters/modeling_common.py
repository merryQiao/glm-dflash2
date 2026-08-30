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
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    initializer_range: float = 0.02
    full_attention: bool = True
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        dimensions = (self.hidden_size, self.intermediate_size, self.num_hidden_layers,
                      self.num_attention_heads, self.num_key_value_heads, self.head_dim,
                      self.num_aux_layers, self.vocab_size)
        if any(value < 1 for value in dimensions):
            raise ValueError("all draft dimensions must be positive")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal heads times head_dim")
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError("GLM-5.3 draft attention requires equal Q and KV heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary positions")
        if not self.full_attention or self.sliding_window is not None:
            raise ValueError("draft backbone requires dense full attention")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")

    @classmethod
    def production(cls) -> "DraftModelConfig":
        return cls(4096, 12288, 5, 64, 64, 64, 5, 154880, 1e-5)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        normalized = value.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype)


def initialize_draft_model(model: nn.Module, *, initializer_range: float) -> None:
    """Match Transformers post_init while restoring intentional stable zeros."""

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        for module in model.modules():
            reset = getattr(module, "reset_stabilized_parameters", None)
            if callable(reset):
                reset(initializer_range=initializer_range)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rotary(position_ids: torch.Tensor, *, head_dim: int, theta: float,
            dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    frequency = 1.0 / (theta ** (torch.arange(
        0, head_dim, 2, device=position_ids.device, dtype=torch.float32
    ) / head_dim))
    angles = position_ids.float().unsqueeze(-1) * frequency
    angles = torch.cat((angles, angles), -1).unsqueeze(1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


class FullAttention(nn.Module):
    """Draft queries attend to [projected full target context | local blocks]."""

    def __init__(self, config: DraftModelConfig) -> None:
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.query_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.key_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, target_context: torch.Tensor,
                position_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or target_context.ndim != 3:
            raise ValueError("attention inputs must be [batch,tokens,hidden]")
        batch, query_length, width = hidden.shape
        context_length = target_context.shape[1]
        key_states = torch.cat((target_context, hidden), 1)
        if position_ids.shape != (batch, context_length + query_length):
            raise ValueError("positions must cover [target context | draft blocks]")
        expected = (batch, 1, query_length, context_length + query_length)
        if attention_mask.shape != expected or attention_mask.dtype != torch.bool:
            raise ValueError(f"attention mask must be bool {expected}")

        def shape(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, value.shape[1], self.config.num_attention_heads,
                              self.config.head_dim).transpose(1, 2)

        query = self.query_norm(shape(self.query(hidden)))
        key = self.key_norm(shape(self.key(key_states)))
        value = shape(self.value(key_states))
        cosine, sine = _rotary(position_ids, head_dim=self.config.head_dim,
                               theta=self.config.rope_theta, dtype=query.dtype)
        query = (query * cosine[..., -query_length:, :] +
                 _rotate_half(query) * sine[..., -query_length:, :])
        key = key * cosine + _rotate_half(key) * sine
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=False,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, query_length, width))


class GatedMLP(nn.Module):
    def __init__(self, config: DraftModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


def _grouped_convolve(hidden: torch.Tensor, dynamic: torch.Tensor,
                      base_kernel: torch.Tensor, *, group_size: int,
                      block_size: int) -> torch.Tensor:
    batch, sequence, width = hidden.shape
    if sequence % block_size:
        raise ValueError("block size must divide flattened draft length")
    groups = width // group_size
    kernel_size = base_kernel.shape[0]
    blocks = hidden.reshape(batch, sequence // block_size, block_size, groups, group_size)
    dynamic = dynamic.reshape(batch, sequence // block_size, block_size,
                              kernel_size, groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(kernel_size):
        values = blocks if offset == 0 else F.pad(
            blocks[:, :, :-offset], (0, 0, 0, 0, offset, 0)
        )
        kernel = base_kernel[offset].reshape(1, 1, 1, groups, group_size)
        output = output + kernel.to(hidden.dtype) * values
        output = torch.addcmul(output, dynamic[..., offset, :, :], values)
    return output.reshape_as(hidden)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size: int, *, kernel_size: int = 2,
                 group_size: int = 16) -> None:
        super().__init__()
        if kernel_size != 2 or group_size < 1 or hidden_size % group_size:
            raise ValueError("DFlash2 requires two taps and a dividing group size")
        self.hidden_size, self.kernel_size, self.group_size = hidden_size, kernel_size, group_size
        groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.zeros(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * groups, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.kernel_projection.weight)
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0].fill_(1.0)

    def reset_stabilized_parameters(self, *, initializer_range: float) -> None:
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

    def finish(self, hidden: torch.Tensor, dynamic: torch.Tensor,
               block_size: int) -> torch.Tensor:
        return _grouped_convolve(hidden, dynamic, self.base_kernel[1],
                                 group_size=self.group_size, block_size=block_size)

    def forward(self, hidden: torch.Tensor, *, block_size: int | None = None) -> torch.Tensor:
        shaped = hidden.ndim == 4
        if shaped:
            batch, anchors, depth, width = hidden.shape
            flat, block_size = hidden.reshape(batch, anchors * depth, width), depth
        elif hidden.ndim == 3 and block_size is not None:
            flat = hidden
        else:
            raise ValueError("dynamic convolution expects block-shaped hidden states")
        prepared, dynamic = self.prepare(flat, int(block_size))
        result = self.finish(prepared, dynamic, int(block_size))
        return result.reshape_as(hidden) if shaped else result


class DenseDecoderLayer(nn.Module):
    def __init__(self, config: DraftModelConfig, *, dynamic_convolution: bool = False,
                 convolution_group_size: int = 16) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = FullAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config)
        factory = lambda: GroupedDynamicCausalConv(
            config.hidden_size, group_size=convolution_group_size
        )
        self.attention_conv = factory() if dynamic_convolution else None
        self.mlp_conv = factory() if dynamic_convolution else None

    def forward(self, hidden: torch.Tensor, target_context: torch.Tensor,
                position_ids: torch.Tensor, attention_mask: torch.Tensor,
                block_size: int) -> torch.Tensor:
        residual, value = hidden, self.input_norm(hidden)
        dynamic = None
        if self.attention_conv is not None:
            value, dynamic = self.attention_conv.prepare(value, block_size)
        value = self.attention(value, target_context, position_ids, attention_mask)
        if self.attention_conv is not None:
            value = self.attention_conv.finish(value, dynamic, block_size)
        hidden = residual + value
        residual, value = hidden, self.post_attention_norm(hidden)
        dynamic = None
        if self.mlp_conv is not None:
            value, dynamic = self.mlp_conv.prepare(value, block_size)
        value = self.mlp(value)
        if self.mlp_conv is not None:
            value = self.mlp_conv.finish(value, dynamic, block_size)
        return residual + value


class DenseDraftBackbone(nn.Module):
    def __init__(self, config: DraftModelConfig, *, dynamic_convolution: bool = False,
                 convolution_group_size: int = 16) -> None:
        super().__init__()
        self.config = config
        self.target_projection = nn.Linear(config.num_aux_layers * config.hidden_size,
                                           config.hidden_size, bias=False)
        self.target_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(DenseDecoderLayer(
            config, dynamic_convolution=dynamic_convolution,
            convolution_group_size=convolution_group_size
        ) for _ in range(config.num_hidden_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def project_target_hidden(self, target_context: torch.Tensor) -> torch.Tensor:
        expected = (self.config.num_aux_layers, self.config.hidden_size)
        if target_context.ndim != 4 or target_context.shape[-2:] != expected:
            raise ValueError("target context must be [batch,sequence,layers,hidden]")
        return self.target_norm(self.target_projection(target_context.flatten(-2)))

    def forward(self, noise_embedding: torch.Tensor, target_context: torch.Tensor, *,
                position_ids: torch.Tensor, attention_mask: torch.Tensor,
                block_size: int) -> torch.Tensor:
        if noise_embedding.ndim != 3 or noise_embedding.shape[-1] != self.config.hidden_size:
            raise ValueError("noise embedding must be [batch,draft,hidden]")
        projected = self.project_target_hidden(target_context)
        hidden = noise_embedding
        for layer in self.layers:
            hidden = layer(hidden, projected, position_ids, attention_mask, block_size)
        return self.final_norm(hidden)
