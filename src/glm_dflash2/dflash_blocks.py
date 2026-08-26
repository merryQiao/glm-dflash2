from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch


MAX_DENSE_SDPA_ANCHORS = 64


class NoValidAnchorsError(RuntimeError):
    """Raised when a microbatch has no two-token supervised anchor."""


@dataclass(frozen=True)
class DFlashBlocks:
    noise_ids: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    context_position_ids: torch.Tensor
    draft_position_ids: torch.Tensor
    full_position_ids: torch.Tensor
    attention_mask: torch.Tensor
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor


def _sample_seed(global_seed: int, epoch: int, sample_id: str) -> int:
    payload = f"{int(global_seed)}\0{int(epoch)}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _validated_attention_mask(
    loss_mask: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones_like(loss_mask, dtype=torch.bool, device="cpu")
    if attention_mask.shape != loss_mask.shape:
        raise ValueError("attention_mask shape differs from loss_mask")
    result = attention_mask.detach().to(device="cpu", dtype=torch.bool)
    for row in result:
        length = int(row.sum().item())
        if not bool(row[:length].all()) or bool(row[length:].any()):
            raise ValueError("attention_mask must be a contiguous true prefix")
    return result


def sample_anchor_positions(
    loss_mask: torch.Tensor,
    *,
    sample_ids: Sequence[str],
    global_seed: int,
    epoch: int,
    attention_mask: torch.Tensor | None = None,
    block_size: int,
    num_anchors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample anchors as a pure function of seed, epoch, and stable sample ID."""

    if loss_mask.ndim != 2:
        raise ValueError("loss_mask must have shape [batch, tokens]")
    if len(sample_ids) != loss_mask.shape[0]:
        raise ValueError("sample_ids must contain one stable ID per batch row")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if not 1 <= num_anchors <= MAX_DENSE_SDPA_ANCHORS:
        raise ValueError(f"num_anchors must be in [1, {MAX_DENSE_SDPA_ANCHORS}]")

    masks = loss_mask.detach().to(device="cpu", dtype=torch.bool)
    attention = _validated_attention_mask(loss_mask, attention_mask)
    sampled: list[torch.Tensor] = []
    for row, sample_id in enumerate(sample_ids):
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a non-empty string")
        length = int(attention[row].sum().item())
        if length < 2:
            eligible = torch.empty(0, dtype=torch.long)
        else:
            pair_valid = (
                masks[row, : length - 1]
                & masks[row, 1:length]
                & attention[row, : length - 1]
                & attention[row, 1:length]
            )
            eligible = torch.nonzero(pair_valid, as_tuple=False).flatten()
        count = min(int(eligible.numel()), int(num_anchors))
        if count:
            generator = torch.Generator(device="cpu").manual_seed(
                _sample_seed(global_seed, epoch, sample_id)
            )
            order = torch.randperm(eligible.numel(), generator=generator)[:count]
            selected = eligible[order].sort().values
        else:
            selected = torch.empty(0, dtype=torch.long)
        sampled.append(selected)

    width = max((int(row.numel()) for row in sampled), default=0)
    if width == 0:
        raise NoValidAnchorsError("microbatch has no supervised anchor with a successor")
    anchors = torch.zeros((len(sampled), width), dtype=torch.long)
    keep = torch.zeros((len(sampled), width), dtype=torch.bool)
    for row, selected in enumerate(sampled):
        count = int(selected.numel())
        anchors[row, :count] = selected
        keep[row, :count] = True
    return anchors.to(loss_mask.device), keep.to(loss_mask.device)


def build_dflash_blocks(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    block_size: int,
    mask_token_id: int,
    attention_dtype: torch.dtype = torch.float32,
) -> DFlashBlocks:
    """Build the shared one-anchor/remaining-mask full-attention block layout."""

    if input_ids.ndim != 2 or loss_mask.shape != input_ids.shape:
        raise ValueError("input_ids and loss_mask must both have shape [batch, tokens]")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if anchor_positions.ndim != 2 or block_keep_mask.shape != anchor_positions.shape:
        raise ValueError("anchor_positions and block_keep_mask must have shape [batch, anchors]")
    if anchor_positions.shape[0] != input_ids.shape[0]:
        raise ValueError("anchor batch dimension differs from input batch")

    device = input_ids.device
    batch, tokens = input_ids.shape
    attention = _validated_attention_mask(loss_mask, attention_mask).to(device)
    real_lengths = attention.sum(dim=-1)
    anchors = anchor_positions.to(device=device, dtype=torch.long)
    keep = block_keep_mask.to(device=device, dtype=torch.bool)
    invalid_anchor = (anchors < 0) | (anchors + 1 >= real_lengths[:, None])
    if bool((invalid_anchor & keep).any()):
        raise ValueError("a kept anchor must have an in-range successor")

    depth = torch.arange(block_size, device=device)
    source_positions = anchors[..., None] + depth
    in_range = source_positions < real_lengths[:, None, None]
    safe_positions = source_positions.clamp(0, max(tokens - 1, 0))
    expanded_ids = input_ids[:, None, :].expand(batch, anchors.shape[1], tokens)
    target_ids = expanded_ids.gather(2, safe_positions)
    raw_mask = loss_mask.to(device=device, dtype=torch.bool)[:, None, :].expand_as(
        expanded_ids
    ).gather(2, safe_positions)
    valid = raw_mask & in_range & keep[..., None]
    target_mask = valid.to(torch.int64).cumprod(dim=-1).to(torch.bool)

    noise_ids = torch.full_like(target_ids, int(mask_token_id))
    noise_ids[..., 0] = target_ids[..., 0]
    noise_ids = noise_ids.reshape(batch, -1)
    draft_position_ids = source_positions.reshape(batch, -1)
    context_position_ids = torch.arange(tokens, device=device).expand(batch, tokens)
    full_position_ids = torch.cat((context_position_ids, draft_position_ids), dim=1)

    num_blocks = anchors.shape[1]
    queries = num_blocks * block_size
    visible = torch.zeros(
        (batch, queries, tokens + queries), device=device, dtype=torch.bool
    )
    context_positions = torch.arange(tokens, device=device)
    for row in range(batch):
        for block in range(num_blocks):
            query_start = block * block_size
            query_end = query_start + block_size
            local_start = tokens + query_start
            local_end = local_start + block_size
            visible[row, query_start:query_end, local_start:local_end] = True
            if bool(keep[row, block]):
                prefix = (context_positions < anchors[row, block]) & attention[row]
                visible[row, query_start:query_end, :tokens] = prefix

    additive = torch.full(
        (batch, 1, queries, tokens + queries),
        float("-inf"),
        device=device,
        dtype=attention_dtype,
    )
    additive.masked_fill_(visible[:, None], 0.0)
    return DFlashBlocks(
        noise_ids=noise_ids,
        target_ids=target_ids,
        target_mask=target_mask,
        context_position_ids=context_position_ids,
        draft_position_ids=draft_position_ids,
        full_position_ids=full_position_ids,
        attention_mask=additive,
        anchor_positions=anchors,
        block_keep_mask=keep,
    )
