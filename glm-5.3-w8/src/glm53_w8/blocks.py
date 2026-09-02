from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch


class NoValidAnchorsError(RuntimeError):
    pass


@dataclass(frozen=True)
class SlidingBlocks:
    noise_ids: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    prediction_mask: torch.Tensor
    context_indices: torch.Tensor
    context_mask: torch.Tensor
    context_position_ids: torch.Tensor
    draft_position_ids: torch.Tensor
    local_visibility: torch.Tensor
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor


def _seed(global_seed: int, epoch: int, sample_id: str) -> int:
    raw = f"{int(global_seed)}\0{int(epoch)}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & ((1 << 63) - 1)


def _attention(loss_mask: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones_like(loss_mask, dtype=torch.bool)
    if attention_mask.shape != loss_mask.shape:
        raise ValueError("attention_mask shape differs from loss_mask")
    result = attention_mask.to(dtype=torch.bool)
    for row in result.detach().cpu():
        length = int(row.sum())
        if not bool(row[:length].all()) or bool(row[length:].any()):
            raise ValueError("attention_mask must be a contiguous true prefix")
    return result


def sample_anchor_positions(
    loss_mask: torch.Tensor,
    *,
    sample_ids: Sequence[str],
    global_seed: int,
    epoch: int,
    block_size: int,
    num_anchors: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loss_mask.ndim != 2 or len(sample_ids) != loss_mask.shape[0]:
        raise ValueError("loss_mask/sample_ids shape mismatch")
    if block_size < 2 or num_anchors < 1:
        raise ValueError("block_size >= 2 and num_anchors >= 1 are required")
    attention = _attention(loss_mask, attention_mask).detach().cpu()
    supervised = loss_mask.detach().to(device="cpu", dtype=torch.bool)
    selections: list[torch.Tensor] = []
    for row, sample_id in enumerate(sample_ids):
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        length = int(attention[row].sum())
        if length < 2:
            eligible = torch.empty(0, dtype=torch.long)
        else:
            eligible = torch.nonzero(
                supervised[row, : length - 1]
                & supervised[row, 1:length]
                & attention[row, : length - 1]
                & attention[row, 1:length],
                as_tuple=False,
            ).flatten()
        count = min(num_anchors, int(eligible.numel()))
        generator = torch.Generator().manual_seed(_seed(global_seed, epoch, sample_id))
        selected = (
            eligible[torch.randperm(eligible.numel(), generator=generator)[:count]].sort().values
            if count
            else torch.empty(0, dtype=torch.long)
        )
        selections.append(selected)
    width = max((int(value.numel()) for value in selections), default=0)
    if width == 0:
        raise NoValidAnchorsError("microbatch has no supervised anchor with a successor")
    anchors = torch.zeros((len(selections), width), dtype=torch.long)
    keep = torch.zeros_like(anchors, dtype=torch.bool)
    for row, selected in enumerate(selections):
        anchors[row, : selected.numel()] = selected
        keep[row, : selected.numel()] = True
    return anchors.to(loss_mask.device), keep.to(loss_mask.device)


def build_sliding_blocks(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    block_size: int,
    mask_token_id: int,
    sliding_window: int,
    attention_mask: torch.Tensor | None = None,
    position_offset: torch.Tensor | int | None = None,
) -> SlidingBlocks:
    if input_ids.ndim != 2 or loss_mask.shape != input_ids.shape:
        raise ValueError("input_ids and loss_mask must have shape [batch, tokens]")
    if anchor_positions.ndim != 2 or block_keep_mask.shape != anchor_positions.shape:
        raise ValueError("anchors and keep mask must have shape [batch, anchors]")
    if anchor_positions.shape[0] != input_ids.shape[0]:
        raise ValueError("anchor batch differs from token batch")
    if block_size < 2 or sliding_window < 1:
        raise ValueError("invalid block_size/sliding_window")
    device = input_ids.device
    batch, tokens = input_ids.shape
    anchors = anchor_positions.to(device=device, dtype=torch.long)
    keep = block_keep_mask.to(device=device, dtype=torch.bool)
    attention = _attention(loss_mask, attention_mask).to(device)
    lengths = attention.sum(-1)
    if bool((((anchors < 0) | (anchors + 1 >= lengths[:, None])) & keep).any()):
        raise ValueError("kept anchor must have an in-range successor")

    depth = torch.arange(block_size, device=device)
    source = anchors[..., None] + depth
    safe_source = source.clamp(0, max(tokens - 1, 0))
    expanded_ids = input_ids[:, None, :].expand(batch, anchors.shape[1], tokens)
    targets = expanded_ids.gather(2, safe_source)
    raw_mask = loss_mask.to(device=device, dtype=torch.bool)[:, None, :].expand_as(
        expanded_ids
    ).gather(2, safe_source)
    in_range = source < lengths[:, None, None]
    target_mask = (
        raw_mask & in_range & keep[..., None]
    ).to(torch.int64).cumprod(-1).to(torch.bool)
    noise = torch.full_like(targets, int(mask_token_id))
    noise[..., 0] = targets[..., 0]

    offsets = torch.arange(sliding_window, device=device)
    starts = (anchors - sliding_window).clamp_min(0)
    prefix_lengths = anchors - starts
    left_padding = sliding_window - prefix_lengths
    context_mask = offsets.view(1, 1, -1) >= left_padding[..., None]
    context_mask &= keep[..., None]
    context_indices = starts[..., None] + (offsets - left_padding[..., None]).clamp_min(0)
    context_indices = context_indices.clamp(0, max(tokens - 1, 0))
    if position_offset is None:
        offsets_per_row = torch.zeros(batch, device=device, dtype=torch.long)
    else:
        offsets_per_row = torch.as_tensor(
            position_offset, device=device, dtype=torch.long
        ).reshape(-1)
        if offsets_per_row.numel() == 1:
            offsets_per_row = offsets_per_row.expand(batch)
        if offsets_per_row.shape != (batch,) or bool((offsets_per_row < 0).any()):
            raise ValueError("position_offset must contain one non-negative value per row")
    context_positions = context_indices + offsets_per_row[:, None, None]
    context_positions.masked_fill_(~context_mask, 0)
    absolute_source = source + offsets_per_row[:, None, None]
    local_visibility = keep[..., None, None].expand(
        batch, anchors.shape[1], block_size, block_size
    )
    return SlidingBlocks(
        noise_ids=noise,
        target_ids=targets,
        target_mask=target_mask,
        prediction_mask=target_mask[..., 1:],
        context_indices=context_indices,
        context_mask=context_mask,
        context_position_ids=context_positions,
        draft_position_ids=absolute_source,
        local_visibility=local_visibility,
        anchor_positions=anchors,
        block_keep_mask=keep,
    )
