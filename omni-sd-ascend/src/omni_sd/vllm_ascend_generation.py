"""vLLM-Ascend provider for deterministic, resumable Thinker generation.

No vLLM or torch-npu module is imported eagerly, so contract checks remain
runnable away from an Ascend host.
"""

from __future__ import annotations

import json
from typing import Any

from omni_sd.thinker_data import stable_int
from omni_sd.thinker_generation import model_conversation


def _one_or_many(items: list[Any] | None) -> Any:
    if not items:
        return None
    return items[0] if len(items) == 1 else items


def prepare_request(
    row: dict[str, Any], config: dict[str, Any], processor: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Render chat and decode local media through the official Qwen utility."""

    from qwen_omni_utils import process_mm_info

    generation = config["generation"]
    conversation = model_conversation(row, generation)
    tools = json.loads(str(row["tools_json"]))
    prompt = processor.apply_chat_template(
        conversation,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos, video_kwargs = process_mm_info(
        conversation,
        use_audio_in_video=bool(generation["use_audio_in_video"]),
        return_video_kwargs=True,
    )
    request: dict[str, Any] = {"prompt": prompt}
    media = {
        name: value
        for name, value in (
            ("audio", _one_or_many(audios)),
            ("image", _one_or_many(images)),
            ("video", _one_or_many(videos)),
        )
        if value is not None
    }
    if media:
        request["multi_modal_data"] = media
    if videos:
        request["mm_processor_kwargs"] = {
            **video_kwargs,
            "use_audio_in_video": bool(generation["use_audio_in_video"]),
        }
    return request, conversation


def engine_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Translate a validated config into one TP/EP-owned vLLM engine."""

    model = config["model"]
    runtime = config["runtime"]
    generation = config["generation"]
    result: dict[str, Any] = {
        "model": str(model["path"]),
        "revision": str(model["revision"]),
        "tokenizer_revision": str(model["processor_revision"]),
        "dtype": str(model["dtype"]),
        "trust_remote_code": True,
        "max_model_len": int(generation["max_model_tokens"]),
        "max_num_seqs": int(runtime["max_num_seqs"]),
        "max_num_batched_tokens": int(runtime["max_num_batched_tokens"]),
        "limit_mm_per_prompt": {
            key: int(value)
            for key, value in runtime["limit_mm_per_prompt"].items()
        },
        "gpu_memory_utilization": float(runtime["gpu_memory_utilization"]),
        "tensor_parallel_size": int(runtime["tensor_parallel_size"]),
        "enable_expert_parallel": bool(runtime.get("enable_expert_parallel", False)),
        "distributed_executor_backend": str(
            runtime.get("distributed_executor_backend", "mp")
        ),
        "enforce_eager": bool(runtime.get("enforce_eager", False)),
        "enable_prefix_caching": bool(runtime.get("enable_prefix_caching", False)),
        "enable_chunked_prefill": False,
        "seed": int(generation["master_seed"]),
    }
    if runtime.get("quantization") not in (None, "", "none"):
        result["quantization"] = str(runtime["quantization"])
    return result


def condition_seed(config: dict[str, Any], condition_id: str) -> int:
    """Seed one condition independently of shard and batch composition."""

    generation = config["generation"]
    return stable_int(
        "omni-thinker-generation-v1",
        generation["sampling_profile"],
        int(generation["master_seed"]),
        str(condition_id),
    )


def sampling_kwargs(config: dict[str, Any], condition_id: str) -> dict[str, Any]:
    """Return explicit sampling arguments for one condition."""

    generation = config["generation"]
    do_sample = bool(generation["do_sample"])
    return {
        "n": 1,
        "seed": condition_seed(config, condition_id),
        "max_tokens": int(generation["max_new_tokens"]),
        "temperature": float(generation["temperature"]) if do_sample else 0.0,
        "top_p": float(generation["top_p"]) if do_sample else 1.0,
        "top_k": int(generation["top_k"]) if do_sample else -1,
        "repetition_penalty": float(generation["repetition_penalty"]),
        "stop_token_ids": [int(value) for value in generation["stop_token_ids"]],
    }


def completion_payload(request_output: Any, eos_token_id: int) -> dict[str, Any]:
    """Preserve exact engine token IDs and reject unusable responses."""

    if len(request_output.outputs) != 1:
        raise ValueError("expected exactly one Thinker completion")
    completion = request_output.outputs[0]
    prompt_ids = [int(value) for value in request_output.prompt_token_ids]
    response_ids = [int(value) for value in completion.token_ids]
    response_text = str(completion.text)
    if not prompt_ids:
        raise ValueError("empty Thinker prompt token output")
    if not response_ids or not response_text.strip():
        raise ValueError("empty Thinker completion")
    finish_reason = (
        "eos"
        if response_ids[-1] == int(eos_token_id)
        else str(completion.finish_reason or "unknown")
    )
    return {
        "prompt_token_ids": prompt_ids,
        "response_token_ids": response_ids,
        "response_text": response_text,
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "finish_reason": finish_reason,
    }


def load_engine(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Create the production engine lazily inside the Ascend environment."""

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    model = config["model"]
    processor = AutoProcessor.from_pretrained(
        str(model["path"]),
        revision=str(model["processor_revision"]),
        trust_remote_code=True,
    )
    return LLM(**engine_kwargs(config)), processor, SamplingParams
