from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChunkedLmProjection:
    nll: torch.Tensor
    topk_scores: torch.Tensor
    topk_ids: torch.Tensor


def chunked_lm_projection(
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    top_k: int,
    token_chunk_size: int = 256,
    vocab_chunk_size: int = 4096,
    token_mask: torch.Tensor | None = None,
) -> ChunkedLmProjection:
    """Compute exact full-vocabulary NLL and top-k with bounded activations.

    The function chunks both flattened token positions and vocabulary rows.
    Log-normalizers are merged with ``logaddexp`` and top-k candidates are
    merged after every vocabulary chunk.  No ``[tokens, vocabulary]`` tensor is
    ever materialized, while gradients still reach ``hidden_states``.
    """

    if hidden_states.ndim < 2 or target_ids.shape != hidden_states.shape[:-1]:
        raise ValueError("target_ids must match every non-hidden dimension")
    if lm_head_weight.ndim != 2 or lm_head_weight.shape[1] != hidden_states.shape[-1]:
        raise ValueError("lm_head_weight must have shape [vocabulary, hidden]")
    vocabulary = int(lm_head_weight.shape[0])
    if top_k < 1 or top_k > vocabulary:
        raise ValueError("top_k must be in [1, vocabulary]")
    if token_chunk_size < 1 or vocab_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    if token_mask is not None and token_mask.shape != target_ids.shape:
        raise ValueError("token_mask shape differs from target_ids")
    if target_ids.numel() and bool(((target_ids < 0) | (target_ids >= vocabulary)).any()):
        raise ValueError("target_ids contain an out-of-vocabulary ID")

    prefix = target_ids.shape
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_targets = target_ids.reshape(-1).to(device=hidden_states.device, dtype=torch.long)
    if token_mask is None:
        valid = torch.ones_like(flat_targets, dtype=torch.bool)
    else:
        valid = token_mask.reshape(-1).to(device=hidden_states.device, dtype=torch.bool)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    reduction_dtype = torch.float64 if hidden_states.dtype == torch.float64 else torch.float32

    nll_output = torch.zeros(flat_targets.shape, device=hidden_states.device, dtype=reduction_dtype)
    score_output = torch.full(
        (flat_targets.numel(), top_k),
        float("-inf"),
        device=hidden_states.device,
        dtype=reduction_dtype,
    )
    id_output = torch.zeros(
        (flat_targets.numel(), top_k), device=hidden_states.device, dtype=torch.long
    )

    for token_start in range(0, int(valid_indices.numel()), token_chunk_size):
        token_index = valid_indices[token_start : token_start + token_chunk_size]
        hidden = flat_hidden.index_select(0, token_index)
        targets = flat_targets.index_select(0, token_index)
        log_normalizer: torch.Tensor | None = None
        target_scores = torch.zeros(
            targets.shape, device=hidden_states.device, dtype=reduction_dtype
        )
        running_scores = torch.empty(
            (hidden.shape[0], 0), device=hidden_states.device, dtype=reduction_dtype
        )
        running_ids = torch.empty(
            (hidden.shape[0], 0), device=hidden_states.device, dtype=torch.long
        )
        for vocab_start in range(0, vocabulary, vocab_chunk_size):
            vocab_end = min(vocabulary, vocab_start + vocab_chunk_size)
            weight = lm_head_weight[vocab_start:vocab_end]
            logits = (hidden.to(weight.dtype) @ weight.T).to(reduction_dtype)
            chunk_lse = torch.logsumexp(logits, dim=-1)
            log_normalizer = (
                chunk_lse
                if log_normalizer is None
                else torch.logaddexp(log_normalizer, chunk_lse)
            )

            in_chunk = (targets >= vocab_start) & (targets < vocab_end)
            local_target = (targets - vocab_start).clamp(0, vocab_end - vocab_start - 1)
            gathered = logits.gather(1, local_target[:, None]).squeeze(1)
            target_scores = target_scores + torch.where(
                in_chunk, gathered, torch.zeros_like(gathered)
            )

            local_k = min(top_k, vocab_end - vocab_start)
            chunk_scores, chunk_ids = logits.topk(local_k, dim=-1)
            chunk_ids = chunk_ids + vocab_start
            merged_scores = torch.cat((running_scores, chunk_scores), dim=-1)
            merged_ids = torch.cat((running_ids, chunk_ids), dim=-1)
            keep_k = min(top_k, merged_scores.shape[-1])
            running_scores, slots = merged_scores.topk(keep_k, dim=-1)
            running_ids = merged_ids.gather(1, slots)

        if log_normalizer is None:  # pragma: no cover - vocabulary is validated non-empty
            raise RuntimeError("empty vocabulary")
        nll = log_normalizer - target_scores
        nll_output = nll_output.index_copy(0, token_index, nll)
        score_output = score_output.index_copy(0, token_index, running_scores)
        id_output = id_output.index_copy(0, token_index, running_ids)

    return ChunkedLmProjection(
        nll=nll_output.reshape(prefix),
        topk_scores=score_output.reshape(prefix + (top_k,)),
        topk_ids=id_output.reshape(prefix + (top_k,)),
    )
