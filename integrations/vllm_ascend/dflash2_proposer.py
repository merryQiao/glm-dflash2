from __future__ import annotations

import torch


def rerank_topk_candidates(
    candidate_ids: torch.Tensor,
    base_scores: torch.Tensor,
    selector_residual: torch.Tensor,
) -> torch.Tensor:
    """Choose DFlash2 tokens from an existing top-k without another vocab projection."""

    if candidate_ids.shape != base_scores.shape or base_scores.shape != selector_residual.shape:
        raise ValueError("candidate IDs, base scores, and selector residual must have the same shape")
    if candidate_ids.ndim < 2 or candidate_ids.shape[-1] < 1:
        raise ValueError("DFlash2 requires a non-empty candidate axis")
    winner = (base_scores + selector_residual.to(base_scores.dtype)).argmax(dim=-1)
    return candidate_ids.gather(-1, winner.unsqueeze(-1)).squeeze(-1)


class DFlash2Proposer:
    """Small custom-class boundary used by the pinned vLLM-Ascend adapter.

    Runtime-specific request/metadata conversion intentionally lives outside
    the training package. The candidate must be runtime-attested before use.
    """

    def __init__(self, draft_model, embed_tokens, lm_head, *, top_k: int) -> None:
        self.draft_model = draft_model
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError("top_k must be positive")

    @torch.no_grad()
    def propose(self, *, input_ids, predecessor_ids, position_ids, target_hidden, attention_mask=None):
        noise = self.embed_tokens(input_ids)
        hidden = self.draft_model(
            position_ids=position_ids,
            noise_embedding=noise,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
        )
        base_scores, candidate_ids = self.lm_head(hidden).topk(self.top_k, dim=-1)
        selector_scores = self.draft_model.candidate_selector.pair_scores(
            hidden, base_scores, candidate_ids, predecessor_ids
        )
        residual = selector_scores - base_scores
        return rerank_topk_candidates(candidate_ids, base_scores, residual)
