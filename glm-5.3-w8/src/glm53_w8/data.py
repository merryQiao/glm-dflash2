from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingWindow:
    start: int
    end: int


@dataclass(frozen=True)
class TrainingReadWindow:
    """Physical cache slice plus the local interval allowed to supply anchors."""

    start: int
    end: int
    anchor_start: int
    anchor_end: int


def expand_training_window(
    core: TrainingWindow,
    *,
    total_tokens: int,
    sliding_window: int,
    block_size: int,
) -> TrainingReadWindow:
    """Add the exact left context halo and enough real tokens for block targets."""

    if not 0 <= core.start < core.end <= int(total_tokens):
        raise ValueError("core training window is outside the trajectory")
    if sliding_window < 1 or block_size < 2:
        raise ValueError("invalid sliding window or block size")
    start = max(0, core.start - int(sliding_window))
    end = min(int(total_tokens), core.end + int(block_size) - 1)
    return TrainingReadWindow(
        start=start,
        end=end,
        anchor_start=core.start - start,
        anchor_end=core.end - start,
    )


def select_training_window(
    loss_mask: torch.Tensor,
    *,
    sample_id: str,
    epoch: int,
    max_tokens: int,
    block_size: int,
    seed: int = 42,
) -> TrainingWindow:
    mask = torch.as_tensor(loss_mask, dtype=torch.bool).reshape(-1).cpu()
    tokens = int(mask.numel())
    if tokens < 2 or max_tokens < 2 * block_size:
        raise ValueError("trajectory/window is too short")
    if tokens <= max_tokens:
        return TrainingWindow(0, tokens)
    eligible = torch.nonzero(mask[:-1] & mask[1:], as_tuple=False).flatten()
    if not eligible.numel():
        raise ValueError("trajectory has no supervised anchor with a successor")
    digest = hashlib.sha256(
        f"{seed}\0{epoch}\0{sample_id}\0window".encode()
    ).digest()
    selected = int(eligible[int.from_bytes(digest[:8], "little") % eligible.numel()])
    start = max(0, min(tokens - max_tokens, selected - max_tokens // 2))
    return TrainingWindow(start, start + max_tokens)
