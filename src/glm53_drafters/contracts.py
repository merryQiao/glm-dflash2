from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class TargetContract:
    num_layers: int = 45
    hidden_size: int = 4096
    vocab_size: int = 154880
    logical_layer_ids: tuple[int, ...] = (1, 11, 22, 32, 42)
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.num_layers < 1 or self.hidden_size < 1 or self.vocab_size < 1:
            raise ValueError("target dimensions must be positive")
        if len(self.logical_layer_ids) != 5:
            raise ValueError("target contract requires exactly five logical layers")
        if tuple(sorted(set(self.logical_layer_ids))) != self.logical_layer_ids:
            raise ValueError("logical layer IDs must be unique and increasing")
        if any(layer < 0 or layer >= self.num_layers for layer in self.logical_layer_ids):
            raise ValueError("logical layer ID is outside the target decoder")

    @property
    def hidden_state_indices(self) -> tuple[int, ...]:
        """Transformers indices for post-block outputs (embedding is index zero)."""

        return tuple(layer + 1 for layer in self.logical_layer_ids)

    @property
    def final_hidden_size(self) -> int:
        return self.hidden_size


@dataclass(frozen=True)
class DraftContract:
    num_layers: int = 5
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    head_dim: int = 64
    rms_norm_eps: float = 1e-5
    full_attention: bool = True
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        dimensions = (
            self.num_layers,
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("draft dimensions must be positive")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden size must equal attention heads times head dimension")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if not self.full_attention or self.sliding_window is not None:
            raise ValueError("GLM-5.3 draft contract requires full attention")


@dataclass(frozen=True)
class CacheContract:
    schema_version: int = 2
    num_aux_layers: int = 5
    hidden_size: int = 4096
    final_hidden_size: int = 4096
    hidden_dtype: str = "bfloat16"
    mask_semantics: str = "dflash_target_token"

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("aligned GLM-5.3 cache requires schema version 2")
        if self.num_aux_layers < 1 or self.hidden_size < 1 or self.final_hidden_size < 1:
            raise ValueError("cache dimensions must be positive")
        if self.hidden_dtype != "bfloat16":
            raise ValueError("production hidden cache dtype must be bfloat16")

    def row_shapes(self, token_count: int) -> dict[str, tuple[int, ...]]:
        if not isinstance(token_count, int) or token_count < 0:
            raise ValueError("token_count must be a non-negative integer")
        return {
            "input_ids": (token_count,),
            "loss_mask": (token_count,),
            "aux_hidden_states": (
                token_count,
                self.num_aux_layers,
                self.hidden_size,
            ),
            "target_final_hidden": (token_count, self.final_hidden_size),
        }

    @property
    def bytes_per_token(self) -> int:
        hidden_values = self.num_aux_layers * self.hidden_size + self.final_hidden_size
        return hidden_values * 2 + 8 + 1


TARGET_CONTRACT = TargetContract()
DRAFT_CONTRACT = DraftContract()
CACHE_CONTRACT = CacheContract()

METHOD_BLOCK_SIZES: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {"dflash": (8, 16), "dflash2": (8, 16), "dspark": (8,)}
)


def _flat_values(values: Any, *, label: str) -> list[Any]:
    """Return a one-dimensional value list without numeric coercion."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a one-dimensional sequence")
    if hasattr(values, "ndim") and int(values.ndim) != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if hasattr(values, "tolist"):
        raw = values.tolist()
    elif isinstance(values, Sequence):
        raw = list(values)
    else:
        raise ValueError(f"{label} must be a one-dimensional sequence")
    if not isinstance(raw, list) or any(isinstance(item, (list, tuple)) for item in raw):
        raise ValueError(f"{label} must be one-dimensional")
    return raw


def validate_token_ids(
    values: Any, *, vocab_size: int = TARGET_CONTRACT.vocab_size
) -> tuple[int, ...]:
    """Validate exact token IDs; values are never accepted via coercion."""

    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 1:
        raise ValueError("vocab_size must be a positive integer")
    raw = _flat_values(values, label="token IDs")
    if not raw:
        raise ValueError("token IDs cannot be empty")
    result: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"token ID at index {index} is not a non-bool integer")
        integer = int(value)
        if not 0 <= integer < vocab_size:
            raise ValueError(
                f"token ID at index {index} is outside [0,{vocab_size}): {integer}"
            )
        result.append(integer)
    return tuple(result)


def validate_loss_mask(values: Any, *, expected_length: int | None = None) -> tuple[bool, ...]:
    """Validate exact binary mask values; floats and strings are rejected."""

    raw = _flat_values(values, label="loss mask")
    if expected_length is not None and len(raw) != expected_length:
        raise ValueError("loss mask length differs from input_ids")
    result: list[bool] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool):
            result.append(value)
            continue
        if not isinstance(value, Integral) or int(value) not in (0, 1):
            raise ValueError(
                f"loss mask at index {index} must be bool or integer 0/1"
            )
        result.append(bool(int(value)))
    return tuple(result)


def validate_method_block(method: str, block_size: int) -> None:
    allowed = METHOD_BLOCK_SIZES.get(method)
    if allowed is None:
        raise ValueError(f"unknown method: {method!r}")
    if block_size not in allowed:
        if method == "dspark":
            raise ValueError("DSpark supports physical block size 8 only")
        raise ValueError(f"{method} block size must be one of {allowed}")


def estimate_cache_bytes(
    token_count: int, *, contract: CacheContract = CACHE_CONTRACT
) -> int:
    if not isinstance(token_count, int) or token_count < 0:
        raise ValueError("token_count must be a non-negative integer")
    return token_count * contract.bytes_per_token
