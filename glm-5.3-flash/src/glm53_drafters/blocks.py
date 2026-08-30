from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AnchorPositions:
    positions: torch.Tensor
    keep_mask: torch.Tensor


@dataclass(frozen=True)
class PhysicalBlocks:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    position_ids: torch.Tensor
    auxiliary_hidden: torch.Tensor
    keep_mask: torch.Tensor
    target_final_hidden: torch.Tensor | None = None
    full_position_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class TrainingWindow:
    start: int
    end: int


def select_training_window(
    loss_mask: torch.Tensor,
    *,
    sample_id: str,
    epoch: int,
    block_size: int,
    max_tokens: int = 4096,
    seed: int = 42,
) -> TrainingWindow:
    """Choose a bounded deterministic window containing a valid full block."""

    if loss_mask.ndim != 1 or max_tokens < block_size:
        raise ValueError("invalid training-window arguments")
    tokens = int(loss_mask.numel())
    if tokens <= max_tokens:
        return TrainingWindow(0, tokens)
    mask = loss_mask.to(device="cpu", dtype=torch.bool)
    valid = torch.nonzero(mask.unfold(0, block_size, 1).all(-1), as_tuple=False).flatten()
    if not valid.numel():
        return TrainingWindow(0, max_tokens)
    stride = max_tokens - block_size + 1
    last = tokens - max_tokens
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    starts = [
        start
        for start in starts
        if mask[start : start + max_tokens].unfold(0, block_size, 1).all(-1).any()
    ]
    if not starts:
        return TrainingWindow(0, max_tokens)
    phase = _stable_seed(sample_id, 0, seed) % len(starts)
    start = starts[(phase + int(epoch)) % len(starts)]
    return TrainingWindow(start, start + max_tokens)


def _stable_seed(sample_id: str, epoch: int, seed: int) -> int:
    raw = f"{sample_id}\0{int(epoch)}\0{int(seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def sample_anchor_positions(
    loss_mask: torch.Tensor,
    *,
    sample_id: str,
    epoch: int,
    block_size: int,
    count: int = 64,
    seed: int = 42,
) -> AnchorPositions:
    """Sample mask-contained physical blocks independent of rank and row order."""

    if loss_mask.ndim != 1:
        raise ValueError("loss_mask must be one-dimensional")
    if block_size < 2 or count < 1 or epoch < 0:
        raise ValueError("invalid anchor sampling arguments")
    mask = loss_mask.to(dtype=torch.bool, device="cpu")
    if mask.numel() < block_size:
        valid = torch.empty(0, dtype=torch.long)
    else:
        valid = torch.nonzero(
            mask.unfold(0, block_size, 1).all(dim=-1), as_tuple=False
        ).flatten()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(str(sample_id), 0, seed))
    if valid.numel():
        valid = valid[torch.randperm(valid.numel(), generator=generator)]
    if valid.numel() > count:
        offset = (int(epoch) * count) % valid.numel()
        selected = valid[(torch.arange(count) + offset) % valid.numel()]
    else:
        selected = valid
    positions = torch.full((count,), -1, dtype=torch.long)
    keep_mask = torch.zeros(count, dtype=torch.bool)
    positions[: selected.numel()] = selected
    keep_mask[: selected.numel()] = True
    return AnchorPositions(positions=positions, keep_mask=keep_mask)


def build_physical_blocks(
    input_ids: torch.Tensor,
    auxiliary_hidden: torch.Tensor,
    anchors: AnchorPositions,
    *,
    block_size: int,
    mask_token_id: int,
    target_final_hidden: torch.Tensor | None = None,
    absolute_position_offset: int = 0,
) -> PhysicalBlocks:
    if input_ids.ndim != 1:
        raise ValueError("input_ids must be one-dimensional")
    if auxiliary_hidden.ndim != 3 or auxiliary_hidden.shape[0] != input_ids.numel():
        raise ValueError("auxiliary_hidden must have shape [tokens,layers,hidden]")
    if target_final_hidden is not None and (
        target_final_hidden.ndim != 2
        or target_final_hidden.shape[0] != input_ids.numel()
    ):
        raise ValueError("target_final_hidden must have shape [tokens,hidden]")
    if anchors.positions.ndim != 1 or anchors.keep_mask.shape != anchors.positions.shape:
        raise ValueError("anchor tensors must be aligned vectors")
    safe = anchors.positions.clamp_min(0)
    offsets = torch.arange(block_size, device=input_ids.device)
    positions = safe.to(input_ids.device).unsqueeze(-1) + offsets
    if anchors.keep_mask.any() and int(positions[anchors.keep_mask].max()) >= input_ids.numel():
        raise ValueError("physical block exceeds token sequence")
    keep = anchors.keep_mask.to(input_ids.device)
    positions[~keep] = 0
    if input_ids.numel():
        target_ids = input_ids[positions]
        auxiliary = auxiliary_hidden[safe.to(auxiliary_hidden.device)]
        final = (
            target_final_hidden[
                positions.to(target_final_hidden.device)
            ]
            if target_final_hidden is not None
            else None
        )
    else:
        target_ids = torch.zeros(
            (anchors.positions.numel(), block_size),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        auxiliary = torch.zeros(
            (
                anchors.positions.numel(),
                auxiliary_hidden.shape[1],
                auxiliary_hidden.shape[2],
            ),
            dtype=auxiliary_hidden.dtype,
            device=auxiliary_hidden.device,
        )
        final = (
            torch.zeros(
                (anchors.positions.numel(), block_size, target_final_hidden.shape[1]),
                dtype=target_final_hidden.dtype,
                device=target_final_hidden.device,
            )
            if target_final_hidden is not None
            else None
        )
    model_ids = target_ids.clone()
    model_ids[:, 1:] = int(mask_token_id)
    if (~keep).any():
        model_ids[~keep] = 0
        target_ids[~keep] = 0
        auxiliary[~keep.to(auxiliary.device)] = 0
        if final is not None:
            final[~keep.to(final.device)] = 0
    context_positions = (
        torch.arange(input_ids.numel(), device=input_ids.device)
        + int(absolute_position_offset)
    )
    draft_positions = positions + int(absolute_position_offset)
    queries = int(anchors.positions.numel()) * block_size
    visible = torch.zeros(
        (1, queries, input_ids.numel() + queries),
        dtype=torch.bool,
        device=input_ids.device,
    )
    for block in range(anchors.positions.numel()):
        query_start = block * block_size
        query_end = query_start + block_size
        local_start = input_ids.numel() + query_start
        visible[:, query_start:query_end, local_start:local_start + block_size] = True
        if bool(keep[block]):
            prefix = torch.arange(input_ids.numel(), device=input_ids.device) < safe[block]
            visible[:, query_start:query_end, :input_ids.numel()] = prefix
    full_positions = torch.cat((context_positions, draft_positions.flatten()))
    return PhysicalBlocks(
        input_ids=model_ids,
        target_ids=target_ids,
        position_ids=draft_positions,
        auxiliary_hidden=auxiliary_hidden,
        keep_mask=keep,
        target_final_hidden=final,
        full_position_ids=full_positions,
        attention_mask=visible,
    )
