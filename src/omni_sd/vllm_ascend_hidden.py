"""Native vLLM hidden-state extraction contracts for Ascend.

Upstream currently archives intermediate decoder outputs only. This module
therefore refuses to label an unnormalised last-layer output as the final
normalised hidden state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

HIDDEN_CACHE_SCHEMA = "omni-thinker-hidden-cache-v3"
from safetensors import safe_open
from safetensors.torch import load_file

from omni_sd.vllm_ascend_generation import engine_kwargs


class HiddenContractError(ValueError):
    """Raised when a hidden artifact cannot safely supervise a drafter."""


class FinalRMSNorm:
    """Exact Qwen3-Omni final RMSNorm evaluated outside the target engine."""

    def __init__(self, weight: torch.Tensor, epsilon: float) -> None:
        if weight.ndim != 1 or not torch.isfinite(weight).all():
            raise HiddenContractError("final norm weight must be a finite vector")
        self.weight = weight.detach().cpu()
        self.epsilon = float(epsilon)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.weight.numel():
            raise HiddenContractError("raw final hidden does not match norm weight")
        dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        # Match Transformers ordering: FP32 normalize -> cast to input dtype ->
        # multiply checkpoint weight in its model dtype.
        normalized = (values * torch.rsqrt(variance + self.epsilon)).to(dtype)
        return (self.weight.to(dtype) * normalized).contiguous()


def load_final_normalizer(config: dict[str, Any]) -> FinalRMSNorm:
    """Load only the final RMSNorm vector, never the full target model."""

    model_root = Path(str(config["model"]["path"]))
    hidden = config["hidden_states"]
    key = str(hidden["final_norm_weight_key"])
    index_path = model_root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get(
            "weight_map", {}
        )
        filename = weight_map.get(key)
        if not filename:
            raise HiddenContractError(f"final norm key {key!r} is absent from index")
        tensor_path = model_root / str(filename)
    else:
        tensor_path = model_root / "model.safetensors"
    if not tensor_path.is_file():
        raise FileNotFoundError(tensor_path)
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise HiddenContractError(f"final norm key {key!r} is absent from checkpoint")
        weight = handle.get_tensor(key)
    return FinalRMSNorm(weight, float(hidden["final_norm_epsilon"]))


def extractor_engine_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    result = engine_kwargs(config)
    hidden = config["hidden_states"]
    result["speculative_config"] = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {
                # The synthetic ``num_hidden_layers`` ID is upstream's raw
                # post-decoder output. We normalize it offline with the exact
                # checkpoint final-norm weight.
                "eagle_aux_hidden_state_layer_ids": [
                    *[int(value) for value in hidden["layer_ids"]],
                    int(hidden["num_target_layers"]),
                ]
            }
        },
    }
    result["kv_transfer_config"] = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "shared_storage_path": str(hidden["scratch_root"]),
            "allow_custom_save_path": True,
            "use_synchronization_lock": True,
        },
    }
    result["enable_chunked_prefill"] = False
    return result


def response_loss_mask(tokens: int, prompt_tokens: int) -> torch.Tensor:
    if tokens <= 0 or prompt_tokens < 0 or prompt_tokens > tokens:
        raise HiddenContractError("invalid prompt/response token lengths")
    mask = torch.zeros(tokens, dtype=torch.bool)
    mask[prompt_tokens:] = True
    return mask


def validate_hidden_tensors(
    hidden_states: torch.Tensor,
    final_hidden_states: torch.Tensor,
    *,
    tokens: int,
    layers: int,
    hidden_size: int,
) -> None:
    expected_hidden = (tokens, layers, hidden_size)
    expected_final = (tokens, hidden_size)
    if tuple(hidden_states.shape) != expected_hidden:
        raise HiddenContractError(
            f"hidden shape {tuple(hidden_states.shape)} != {expected_hidden}"
        )
    if tuple(final_hidden_states.shape) != expected_final:
        raise HiddenContractError(
            f"final hidden shape {tuple(final_hidden_states.shape)} != {expected_final}"
        )
    if not torch.is_floating_point(hidden_states) or not torch.is_floating_point(
        final_hidden_states
    ):
        raise HiddenContractError("hidden tensors must be floating point")
    if not torch.isfinite(hidden_states).all() or not torch.isfinite(
        final_hidden_states
    ).all():
        raise HiddenContractError("hidden tensors must be finite")


def load_connector_tensors(
    path: str | Path,
    expected_token_ids: Sequence[int],
    config: dict[str, Any],
    normalizer: FinalRMSNorm | None = None,
    tensor_loader: Callable[[str], Mapping[str, torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    tensors = (
        load_file(str(path), device="cpu")
        if tensor_loader is None
        else dict(tensor_loader(str(path)))
    )
    missing = {"token_ids", "hidden_states"}.difference(tensors)
    if missing:
        raise HiddenContractError(f"connector record missing {sorted(missing)}")
    token_ids = tensors["token_ids"].to(torch.int64)
    expected = torch.tensor([int(value) for value in expected_token_ids], dtype=torch.int64)
    if not torch.equal(token_ids, expected):
        raise HiddenContractError("connector token IDs do not exactly match trajectory")
    hidden_states = tensors["hidden_states"]
    hidden = config["hidden_states"]
    configured_layers = len(hidden["layer_ids"])
    if "final_hidden_states" in tensors:
        final_hidden_states = tensors["final_hidden_states"]
    elif hidden_states.ndim == 3 and hidden_states.shape[1] == configured_layers + 1:
        if normalizer is None:
            raise HiddenContractError(
                "final normalized hidden is absent and no validated RMSNorm was provided"
            )
        raw_final = hidden_states[:, -1]
        hidden_states = hidden_states[:, :-1]
        final_hidden_states = normalizer(raw_final)
    else:
        raise HiddenContractError(
            "final normalized hidden is absent; upstream last-layer output is "
            "unnormalized and cannot be substituted"
        )
    validate_hidden_tensors(
        hidden_states,
        final_hidden_states,
        tokens=len(expected_token_ids),
        layers=configured_layers,
        hidden_size=int(hidden["hidden_size"]),
    )
    expected_dtype = getattr(torch, str(hidden["dtype"]))
    if hidden_states.dtype != expected_dtype or final_hidden_states.dtype != expected_dtype:
        raise HiddenContractError(
            f"hidden dtype must be {expected_dtype}, got "
            f"{hidden_states.dtype}/{final_hidden_states.dtype}"
        )
    return {
        "token_ids": token_ids,
        "hidden_states": hidden_states,
        "final_hidden_states": final_hidden_states,
    }


def native_connector_loader(path: str) -> Mapping[str, torch.Tensor]:
    """Use upstream's lock-aware loader for asynchronous connector writes."""

    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        example_hidden_states_connector,
    )

    return example_hidden_states_connector.load_hidden_states(path)


def cleanup_connector_artifact(path: str | Path) -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        example_hidden_states_connector,
    )

    example_hidden_states_connector.cleanup_hidden_states(str(path))


def load_engine(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Create the extractor lazily inside the pinned Ascend image."""

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    model = config["model"]
    processor = AutoProcessor.from_pretrained(
        str(model["path"]),
        revision=str(model["processor_revision"]),
        trust_remote_code=True,
    )
    return LLM(**extractor_engine_kwargs(config)), processor, SamplingParams
