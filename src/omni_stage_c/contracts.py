from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class TargetContract:
    num_layers: int = 48
    hidden_size: int = 2048
    vocab_size: int = 152064
    logical_layer_ids: tuple[int, ...] = (1, 12, 24, 36, 47)
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 65536

    def __post_init__(self) -> None:
        # Qwen3-Omni has a 2048-wide residual stream, a 4096-wide Q
        # projection (32 x 128), and 512-wide K/V projections (4 x 128).
        if (
            self.hidden_size != 2048
            or self.num_attention_heads * self.head_dim != 4096
            or self.num_key_value_heads * self.head_dim != 512
        ):
            raise ValueError("invalid Qwen3-Omni Thinker attention contract")
        if len(self.logical_layer_ids) != 5:
            raise ValueError("exactly five Thinker hidden streams are required")
        if tuple(sorted(set(self.logical_layer_ids))) != self.logical_layer_ids:
            raise ValueError("logical hidden layer IDs must be unique and increasing")
        if self.logical_layer_ids[-1] >= self.num_layers:
            raise ValueError("hidden layer ID lies outside the Thinker decoder")
        if sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError("mRoPE sections must cover half the attention head")

    @property
    def hidden_state_indices(self) -> tuple[int, ...]:
        """Transformers hidden-state indices (embedding occupies index zero)."""

        return tuple(layer + 1 for layer in self.logical_layer_ids)


@dataclass(frozen=True)
class DraftContract:
    num_layers: int = 5
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    full_attention: bool = True
    sliding_window: None = None


@dataclass(frozen=True)
class CacheContract:
    schema: str = "omni-thinker-hidden-cache-v3"
    num_aux_layers: int = 5
    hidden_size: int = 2048
    final_hidden_size: int = 2048
    position_axes: int = 3
    hidden_dtype: str = "bfloat16"

    @property
    def bytes_per_token(self) -> int:
        hidden_values = (self.num_aux_layers + 1) * self.hidden_size
        return hidden_values * 2 + 8 + 1 + self.position_axes * 8


TARGET_CONTRACT: Final = TargetContract()
DRAFT_CONTRACT: Final = DraftContract()
CACHE_CONTRACT: Final = CacheContract()
DEFAULT_MASK_TOKEN_ID: Final[int] = 152063
METHOD_BLOCK_SIZES: Final[dict[str, tuple[int, ...]]] = {
    "dflash": (8, 16),
    "dflash2": (8, 16),
    "dspark": (8,),
}


def validate_method_block(method: str, block_size: int) -> None:
    allowed = METHOD_BLOCK_SIZES.get(method)
    if allowed is None:
        raise ValueError(f"unknown method: {method!r}")
    if block_size not in allowed:
        if method == "dspark":
            raise ValueError("DSpark supports physical block size 8 only")
        raise ValueError(f"{method} block size must be one of {allowed}")
