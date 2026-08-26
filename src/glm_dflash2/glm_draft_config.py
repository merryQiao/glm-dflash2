from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GLMDraftSpec:
    target_layer_ids: tuple[int, ...]
    target_num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    block_size: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int | None

    def __post_init__(self) -> None:
        if len(self.target_layer_ids) != self.num_hidden_layers:
            raise ValueError("one ordered target layer is required per draft layer")
        if tuple(sorted(self.target_layer_ids)) != self.target_layer_ids:
            raise ValueError("target_layer_ids must be strictly ordered")
        if self.target_layer_ids[-1] >= self.target_num_hidden_layers:
            raise ValueError("target layer is outside the decoder depth")
        if self.num_attention_heads < 1 or self.num_key_value_heads < 1:
            raise ValueError("attention head counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("KV heads must divide query heads")
        if self.head_dim < 1:
            raise ValueError("head_dim must be positive")

    @property
    def q_projection_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_projection_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


GLM52_DRAFT_SPEC = GLMDraftSpec(
    target_layer_ids=(1, 20, 38, 56, 75),
    target_num_hidden_layers=78,
    hidden_size=6144,
    intermediate_size=12288,
    num_hidden_layers=5,
    num_attention_heads=64,
    num_key_value_heads=64,
    head_dim=64,
    block_size=16,
    rms_norm_eps=1e-5,
    rope_theta=8_000_000.0,
    sliding_window=None,
)

