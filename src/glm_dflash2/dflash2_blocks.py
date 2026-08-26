"""Compatibility layer for the former DFlash2-only block utilities."""

from __future__ import annotations

from typing import Sequence

import torch

from .dflash_blocks import (
    MAX_DENSE_SDPA_ANCHORS,
    DFlashBlocks,
    NoValidAnchorsError,
    build_dflash_blocks as _build_dflash_blocks,
    sample_anchor_positions as _sample_anchor_positions,
)


def sample_anchor_positions(
    loss_mask: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    block_size: int,
    num_anchors: int,
    generator: torch.Generator | None = None,
    sample_ids: Sequence[str] | None = None,
    global_seed: int | None = None,
    epoch: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the pure sampler; legacy generator calls get stable row IDs."""

    if sample_ids is None:
        if generator is None:
            raise ValueError("sample_ids are required by the aligned sampler")
        sample_ids = tuple(f"legacy-row-{row}" for row in range(loss_mask.shape[0]))
        global_seed = int(generator.initial_seed())
    if global_seed is None:
        raise ValueError("global_seed is required with sample_ids")
    return _sample_anchor_positions(
        loss_mask,
        sample_ids=sample_ids,
        global_seed=global_seed,
        epoch=epoch,
        attention_mask=attention_mask,
        block_size=block_size,
        num_anchors=num_anchors,
    )


def build_dflash_blocks(*args, sliding_window: int | None = None, **kwargs) -> DFlashBlocks:
    """Ignore the removed sliding-window option and build full-attention blocks."""

    del sliding_window
    return _build_dflash_blocks(*args, **kwargs)


__all__ = [
    "MAX_DENSE_SDPA_ANCHORS",
    "DFlashBlocks",
    "NoValidAnchorsError",
    "build_dflash_blocks",
    "sample_anchor_positions",
]
