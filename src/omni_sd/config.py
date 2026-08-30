"""Strict configuration contract for the Ascend Thinker data pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


MODALITY_KEYS = ("text", "image", "multi_image", "audio", "video", "other")


class ConfigError(ValueError):
    """Raised before model loading when a pipeline configuration is unsafe."""


def _mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str, section: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{section}.{key} must be a positive integer")
    return value


def _batch_sizes(config: Mapping[str, Any], section: str) -> None:
    sizes = _mapping(config, section)
    for key in MODALITY_KEYS:
        _positive_int(sizes, key, section)


def validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a defensive copy after validating every production contract."""

    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a mapping")
    config = deepcopy(dict(raw))
    for section in ("input", "output", "model", "generation", "runtime", "hidden_states"):
        _mapping(config, section)

    model = config["model"]
    for key in ("id", "path", "revision", "processor_revision", "dtype"):
        if model.get(key) in (None, ""):
            raise ConfigError(f"model.{key} is required")
    for key in ("revision", "processor_revision"):
        if str(model[key]).startswith("REPLACE_"):
            raise ConfigError(f"model.{key} must be resolved to an immutable commit")
    if str(model["dtype"]) != "bfloat16":
        raise ConfigError("the production target must use BF16")

    runtime = config["runtime"]
    if runtime.get("backend") != "vllm_ascend":
        raise ConfigError("runtime.backend must be vllm_ascend")
    if str(runtime.get("hardware", "")).lower() not in {"a2", "a3"}:
        raise ConfigError("runtime.hardware must be a2 or a3")
    _positive_int(runtime, "tensor_parallel_size", "runtime")
    _positive_int(runtime, "max_num_seqs", "runtime")
    _positive_int(runtime, "max_num_batched_tokens", "runtime")
    limits = _mapping(runtime, "limit_mm_per_prompt")
    for key in ("image", "video", "audio"):
        _positive_int(limits, key, "runtime.limit_mm_per_prompt")
    utilization = runtime.get("gpu_memory_utilization")
    if not isinstance(utilization, (int, float)) or not 0 < float(utilization) < 1:
        raise ConfigError("runtime.gpu_memory_utilization must be between zero and one")
    if runtime.get("quantization") not in (None, "", "none"):
        raise ConfigError("BF16 target weights must not enable Ascend quantization")

    generation = config["generation"]
    _positive_int(generation, "max_new_tokens", "generation")
    _positive_int(generation, "max_model_tokens", "generation")
    _positive_int(generation, "eos_token_id", "generation")
    stop_ids = generation.get("stop_token_ids")
    if not isinstance(stop_ids, list) or not stop_ids or not all(
        isinstance(value, int) and value >= 0 for value in stop_ids
    ):
        raise ConfigError("generation.stop_token_ids must be a non-empty integer list")
    if int(generation["eos_token_id"]) not in stop_ids:
        raise ConfigError("generation.stop_token_ids must include eos_token_id")
    if int(runtime["max_num_batched_tokens"]) < int(generation["max_model_tokens"]):
        raise ConfigError(
            "runtime.max_num_batched_tokens must cover max_model_tokens because "
            "hidden extraction disables chunked prefill"
        )

    _batch_sizes(config, "vllm_batch_size")
    hidden = config["hidden_states"]
    if hidden.get("backend") != "vllm_ascend":
        raise ConfigError("hidden_states.backend must be vllm_ascend")
    for key in ("output_root", "scratch_root", "dtype", "final_norm_weight_key"):
        if hidden.get(key) in (None, ""):
            raise ConfigError(f"hidden_states.{key} is required")
    if hidden["dtype"] != "bfloat16":
        raise ConfigError("hidden_states.dtype must be bfloat16")
    layer_count = _positive_int(hidden, "num_target_layers", "hidden_states")
    _positive_int(hidden, "hidden_size", "hidden_states")
    _positive_int(hidden, "max_shard_bytes", "hidden_states")
    layer_ids = hidden.get("layer_ids")
    if (
        not isinstance(layer_ids, list)
        or not layer_ids
        or not all(isinstance(value, int) for value in layer_ids)
        or len(set(layer_ids)) != len(layer_ids)
        or min(layer_ids) < 0
        or max(layer_ids) >= layer_count
    ):
        raise ConfigError(
            f"hidden_states.layer_ids must be unique indices in [0, {layer_count})"
        )
    if hidden.get("require_final_normalized_hidden") is not True:
        raise ConfigError(
            "hidden_states.require_final_normalized_hidden must stay true; "
            "a decoder output is not a valid substitute"
        )
    epsilon = hidden.get("final_norm_epsilon")
    if not isinstance(epsilon, (int, float)) or float(epsilon) <= 0:
        raise ConfigError("hidden_states.final_norm_epsilon must be positive")
    _batch_sizes(hidden, "batch_size")

    output = config["output"]
    _positive_int(output, "conditions_per_shard", "output")
    _positive_int(config["input"], "expected_conditions", "input")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return validate_config(value)
