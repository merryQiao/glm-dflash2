"""Exact Qwen3-Omni Thinker mRoPE positions for frozen trajectories."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch


def validate_position_ids(position_ids: torch.Tensor, *, tokens: int) -> None:
    if position_ids.dtype != torch.int64:
        raise ValueError("position IDs must use int64")
    if tuple(position_ids.shape) != (int(tokens), 3):
        raise ValueError(f"position IDs must have shape ({int(tokens)},3)")
    if position_ids.numel() and int(position_ids.min()) < 0:
        raise ValueError("position IDs must be non-negative")


def extend_response_position_ids(
    prompt_position_ids: torch.Tensor, *, response_tokens: int
) -> torch.Tensor:
    """Append 1-D text positions after the largest multimodal prompt index."""

    if prompt_position_ids.dtype != torch.int64 or prompt_position_ids.ndim != 2:
        raise ValueError("prompt position IDs must be int64 [3,prompt_tokens]")
    if prompt_position_ids.shape[0] != 3 or prompt_position_ids.shape[1] < 1:
        raise ValueError("prompt position IDs must have three non-empty axes")
    if response_tokens < 1:
        raise ValueError("response_tokens must be positive")
    start = int(prompt_position_ids.max().item()) + 1
    response = torch.arange(
        start,
        start + int(response_tokens),
        dtype=torch.int64,
        device=prompt_position_ids.device,
    ).unsqueeze(-1).expand(-1, 3)
    result = torch.cat((prompt_position_ids.T.contiguous(), response), dim=0)
    validate_position_ids(result, tokens=result.shape[0])
    return result


class _OfficialRopeProvider:
    """Bind the official Transformers index routine without allocating weights."""

    def __init__(self, thinker_config: Any) -> None:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoePreTrainedModelForConditionalGeneration,
        )

        self.config = thinker_config
        self.spatial_merge_size = int(thinker_config.vision_config.spatial_merge_size)
        self.position_id_per_seconds = int(thinker_config.position_id_per_seconds)
        base = Qwen3OmniMoePreTrainedModelForConditionalGeneration
        self.get_llm_pos_ids_for_vision = MethodType(
            base.get_llm_pos_ids_for_vision, self
        )
        self.get_rope_index = MethodType(base.get_rope_index, self)


def exact_trajectory_position_ids(
    *,
    processor: Any,
    request: dict[str, Any],
    prompt_token_ids: list[int],
    response_tokens: int,
    thinker_config: Any,
) -> torch.Tensor:
    """Re-run the official processor and Thinker index routine on one request.

    The saved vLLM prompt IDs are the authority. A processor/runtime tokenizer
    drift aborts before positions are archived.
    """

    media = dict(request.get("multi_modal_data") or {})
    kwargs: dict[str, Any] = {
        "text": str(request["prompt"]),
        "audio": media.get("audio"),
        "images": media.get("image"),
        "videos": media.get("video"),
        "return_tensors": "pt",
        "padding": True,
        **dict(request.get("mm_processor_kwargs") or {}),
    }
    processed = processor(**kwargs)
    encoded = processed["input_ids"].to(dtype=torch.int64)
    expected = torch.tensor(prompt_token_ids, dtype=torch.int64)
    if encoded.shape != (1, expected.numel()) or not torch.equal(encoded[0], expected):
        raise ValueError("official processor token IDs differ from saved engine prompt IDs")
    provider = _OfficialRopeProvider(thinker_config)
    feature_mask = processed.get("feature_attention_mask")
    audio_seqlens = feature_mask.sum(-1) if feature_mask is not None else None
    prompt_positions, _ = provider.get_rope_index(
        input_ids=encoded,
        image_grid_thw=processed.get("image_grid_thw"),
        video_grid_thw=processed.get("video_grid_thw"),
        attention_mask=processed.get("attention_mask", torch.ones_like(encoded)),
        use_audio_in_video=bool(
            dict(request.get("mm_processor_kwargs") or {}).get(
                "use_audio_in_video", False
            )
        ),
        audio_seqlens=audio_seqlens,
        second_per_grids=processed.get("video_second_per_grid"),
    )
    return extend_response_position_ids(
        prompt_positions[:, 0].to(dtype=torch.int64),
        response_tokens=int(response_tokens),
    ).cpu()
