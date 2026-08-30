from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AnchorPositions:
    positions: torch.Tensor
    keep_mask: torch.Tensor


@dataclass(frozen=True)
class TrainingWindow:
    start: int
    end: int


def _stable_seed(sample_id: str, seed: int) -> int:
    raw = f"{sample_id}\0{seed}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def select_training_window(loss_mask: torch.Tensor, *, sample_id: str, epoch: int,
                           block_size: int, max_tokens: int = 4096,
                           seed: int = 42) -> TrainingWindow:
    if loss_mask.ndim != 1 or max_tokens < block_size:
        raise ValueError("invalid training-window arguments")
    tokens = int(loss_mask.numel())
    if tokens <= max_tokens:
        return TrainingWindow(0, tokens)
    mask = loss_mask.cpu().bool()
    starts = list(range(0, tokens - max_tokens + 1, max_tokens - block_size + 1))
    if starts[-1] != tokens - max_tokens:
        starts.append(tokens - max_tokens)
    starts = [s for s in starts if mask[s:s + max_tokens].unfold(0, block_size, 1).all(-1).any()]
    if not starts:
        return TrainingWindow(0, max_tokens)
    phase = _stable_seed(sample_id, seed) % len(starts)
    start = starts[(phase + int(epoch)) % len(starts)]
    return TrainingWindow(start, start + max_tokens)


def sample_anchor_positions(loss_mask: torch.Tensor, *, sample_id: str, epoch: int,
                            block_size: int, count: int = 512,
                            seed: int = 42) -> AnchorPositions:
    if loss_mask.ndim != 1 or count < 1 or block_size < 2:
        raise ValueError("invalid anchor sampling arguments")
    mask = loss_mask.cpu().bool()
    valid = (
        torch.nonzero(mask.unfold(0, block_size, 1).all(-1), as_tuple=False).flatten()
        if mask.numel() >= block_size else torch.empty(0, dtype=torch.long)
    )
    generator = torch.Generator().manual_seed(_stable_seed(sample_id, seed))
    if valid.numel():
        valid = valid[torch.randperm(valid.numel(), generator=generator)]
    if valid.numel() > count:
        offset = (int(epoch) * count) % valid.numel()
        selected = valid[(torch.arange(count) + offset) % valid.numel()]
    else:
        selected = valid
    positions = torch.full((count,), -1, dtype=torch.long)
    keep = torch.zeros(count, dtype=torch.bool)
    positions[:selected.numel()], keep[:selected.numel()] = selected, True
    return AnchorPositions(positions, keep)


def build_training_batch(sample: dict[str, torch.Tensor | str], anchors: AnchorPositions,
                         *, block_size: int, mask_token_id: int,
                         device: torch.device) -> "TrainingBatch":
    """Build one physical-block batch while preserving exact cached mRoPE."""

    from .offline_trainer import TrainingBatch

    ids = torch.as_tensor(sample["input_ids"]).long()
    aux = torch.as_tensor(sample["auxiliary_hidden"])
    final = torch.as_tensor(sample["target_final_hidden"])
    mrope = torch.as_tensor(sample["position_ids"]).long()
    if mrope.shape != (ids.numel(), 3):
        raise ValueError("cached position_ids must be [tokens,3]")
    safe = anchors.positions.clamp_min(0)
    offsets = torch.arange(block_size)
    indices = safe[:, None] + offsets
    keep = anchors.keep_mask
    if keep.any() and int(indices[keep].max()) >= ids.numel():
        raise ValueError("sampled block exceeds cached trajectory")
    indices[~keep] = 0
    targets = ids[indices]
    model_ids = targets.clone()
    model_ids[:, 1:] = int(mask_token_id)
    model_ids[~keep], targets[~keep] = 0, 0
    block_final = final[indices]
    block_final[~keep] = 0
    draft_positions = mrope[indices]

    tokens, anchor_count = ids.numel(), anchors.positions.numel()
    visible = torch.zeros(1, anchor_count, block_size, tokens + block_size, dtype=torch.bool)
    for block in range(anchor_count):
        visible[:, block, :, tokens:] = True
        if bool(keep[block]):
            visible[:, block, :, :tokens] = torch.arange(tokens) < safe[block]
    full_positions = torch.cat((mrope, draft_positions.flatten(0, 1)), dim=0)
    return TrainingBatch(
        input_ids=model_ids[None].to(device),
        target_ids=targets[None].to(device),
        position_ids=full_positions.T[:, None].to(device),
        auxiliary_hidden=aux[None].to(device),
        target_final_hidden=block_final[None].to(device),
        keep_mask=keep[None].to(device),
        attention_mask=visible.to(device),
    )
