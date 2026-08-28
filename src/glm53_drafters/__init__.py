"""Standalone GLM-5.3-Flash Stage B and offline drafter utilities."""

from .contracts import (
    CACHE_CONTRACT,
    DRAFT_CONTRACT,
    METHOD_BLOCK_SIZES,
    TARGET_CONTRACT,
    CacheContract,
    DraftContract,
    TargetContract,
    estimate_cache_bytes,
    validate_method_block,
)

__all__ = [
    "CACHE_CONTRACT",
    "DRAFT_CONTRACT",
    "METHOD_BLOCK_SIZES",
    "TARGET_CONTRACT",
    "CacheContract",
    "DraftContract",
    "TargetContract",
    "estimate_cache_bytes",
    "validate_method_block",
]

__version__ = "0.1.0"
