from __future__ import annotations

from dataclasses import dataclass

import torch


MAX_DENSE_SDPA_ANCHORS = 64


class NoValidAnchorsError(RuntimeError):
    """Raised when a microbatch cannot form a complete supervised block."""


@dataclass(frozen=True)
class DFlashBlocks:
    """Packed DFlash2 blocks and the exact dense SDPA metadata they require."""

    noise_ids: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    draft_position_ids: torch.Tensor
    full_position_ids: torch.Tensor
    attention_mask: torch.Tensor
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor


def sample_anchor_positions(
    loss_mask: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    block_size: int,
    num_anchors: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample complete supervised anchors without replacement.

    Sampling deliberately happens on CPU.  This makes the dedicated generator
    state portable and checkpointable across CPU, CUDA, and Ascend runs.
    """

    if loss_mask.ndim != 2:
        raise ValueError("loss_mask must have shape [batch, tokens]")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if num_anchors < 1:
        raise ValueError("num_anchors must be positive")
    if num_anchors > MAX_DENSE_SDPA_ANCHORS:
        raise ValueError(f"num_anchors must be at most {MAX_DENSE_SDPA_ANCHORS}")
    if generator.device.type != "cpu":
        raise ValueError("anchor generator must be a CPU torch.Generator")

    mask_cpu = loss_mask.detach().to(device="cpu", dtype=torch.bool)
    if attention_mask is None:
        attention_cpu = torch.ones_like(mask_cpu)
    else:
        if attention_mask.shape != loss_mask.shape:
            raise ValueError("attention_mask shape differs from loss_mask")
        attention_cpu = attention_mask.detach().to(device="cpu", dtype=torch.bool)
    batch, tokens = mask_cpu.shape
    sampled: list[torch.Tensor] = []
    for row in range(batch):
        length = int(attention_cpu[row].sum().item())
        if not bool(attention_cpu[row, :length].all()) or bool(attention_cpu[row, length:].any()):
            raise ValueError("attention_mask must be a contiguous true prefix")
        if length < block_size:
            eligible = torch.empty(0, dtype=torch.long)
        else:
            windows = mask_cpu[row, :length].unfold(0, block_size, 1)
            # The anchor token itself and at least one predicted successor must
            # be supervised.  This excludes prompt-only/zero-loss blocks while
            # retaining partial assistant turns near an answer boundary.
            eligible_mask = windows[:, 0] & windows[:, 1:].any(dim=-1)
            eligible = torch.nonzero(eligible_mask, as_tuple=False).flatten()
        count = min(num_anchors, int(eligible.numel()))
        if count:
            order = torch.randperm(eligible.numel(), generator=generator)[:count]
            selected = eligible[order].sort().values
        else:
            selected = torch.empty(0, dtype=torch.long)
        sampled.append(selected)

    width = max((int(value.numel()) for value in sampled), default=0)
    if width == 0:
        raise NoValidAnchorsError("microbatch has no supervised full-block anchor")
    anchors = torch.zeros((batch, width), dtype=torch.long)
    keep = torch.zeros((batch, width), dtype=torch.bool)
    for row, selected in enumerate(sampled):
        count = int(selected.numel())
        if count:
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
    sliding_window: int,
    attention_dtype: torch.dtype = torch.float32,
) -> DFlashBlocks:
    """Construct unshifted labels, noisy blocks, positions, and additive mask."""

    if input_ids.ndim != 2 or loss_mask.shape != input_ids.shape:
        raise ValueError("input_ids and loss_mask must both have shape [batch, tokens]")
    if attention_mask is None:
        attention_mask = torch.ones_like(loss_mask, dtype=torch.bool)
    elif attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask shape differs from input_ids")
    if anchor_positions.ndim != 2 or block_keep_mask.shape != anchor_positions.shape:
        raise ValueError("anchor_positions and block_keep_mask must have shape [batch, anchors]")
    if anchor_positions.shape[0] != input_ids.shape[0]:
        raise ValueError("anchor batch dimension differs from input batch")
    if block_size < 2 or sliding_window < 1:
        raise ValueError("invalid block_size or sliding_window")

    device = input_ids.device
    batch, tokens = input_ids.shape
    anchors = anchor_positions.to(device=device, dtype=torch.long)
    keep = block_keep_mask.to(device=device, dtype=torch.bool)
    if bool(((anchors < 0) | (anchors + block_size > tokens)).logical_and(keep).any()):
        raise ValueError("a kept anchor cannot form a complete block")
    real_lengths = attention_mask.to(device=device, dtype=torch.bool).sum(dim=-1)
    if bool(((anchors + block_size) > real_lengths[:, None]).logical_and(keep).any()):
        raise ValueError("a kept anchor crosses the real sequence length")

    depth = torch.arange(block_size, device=device)
    source_positions = anchors[..., None] + depth
    safe_positions = source_positions.clamp(0, max(tokens - 1, 0))
    target_ids = input_ids[:, None, :].expand(batch, anchors.shape[1], tokens).gather(
        2, safe_positions
    )
    target_mask = loss_mask.to(torch.bool)[:, None, :].expand(
        batch, anchors.shape[1], tokens
    ).gather(2, safe_positions)
    target_mask = target_mask & keep[..., None]

    noise = torch.full_like(target_ids, int(mask_token_id))
    noise[..., 0] = target_ids[..., 0]
    noise_ids = noise.reshape(batch, -1)
    draft_position_ids = source_positions.reshape(batch, -1)
    context_position_ids = torch.arange(tokens, device=device).expand(batch, tokens)
    full_position_ids = torch.cat((context_position_ids, draft_position_ids), dim=1)

    num_blocks = anchors.shape[1]
    queries = num_blocks * block_size
    keys = tokens + queries
    visible = torch.zeros((batch, queries, keys), device=device, dtype=torch.bool)
    context_positions = torch.arange(tokens, device=device)
    for row in range(batch):
        for block in range(num_blocks):
            anchor = anchors[row, block]
            query_start = block * block_size
            query_end = query_start + block_size
            own_key_start = tokens + query_start
            own_key_end = own_key_start + block_size
            # Even padded blocks retain their own block to avoid an all-masked
            # SDPA row.  Their target mask is zero, so they carry no objective.
            visible[row, query_start:query_end, own_key_start:own_key_end] = True
            if not bool(keep[row, block]):
                continue
            absolute_query = anchor + depth
            prefix = context_positions[None, :] < anchor
            in_window = (absolute_query[:, None] - context_positions[None, :]) < sliding_window
            visible[row, query_start:query_end, :tokens] = prefix & in_window

    additive = torch.full(
        (batch, 1, queries, keys),
        float("-inf"),
        device=device,
        dtype=attention_dtype,
    )
    additive.masked_fill_(visible[:, None], 0.0)
    return DFlashBlocks(
        noise_ids=noise_ids,
        target_ids=target_ids,
        target_mask=target_mask,
        draft_position_ids=draft_position_ids,
        full_position_ids=full_position_ids,
        attention_mask=additive,
        anchor_positions=anchors,
        block_keep_mask=keep,
    )
