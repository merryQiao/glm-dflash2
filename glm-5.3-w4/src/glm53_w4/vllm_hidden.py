from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .contracts import TARGET_CONTRACT


class HiddenContractError(ValueError):
    pass


class FinalRMSNorm:
    def __init__(self, weight: torch.Tensor, epsilon: float) -> None:
        if weight.ndim != 1 or not bool(torch.isfinite(weight.float()).all()):
            raise HiddenContractError("final norm weight must be a finite vector")
        self.weight = weight.detach().cpu().contiguous()
        self.epsilon = float(epsilon)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.weight.numel():
            raise HiddenContractError("raw final hidden does not match final RMSNorm")
        dtype = hidden_states.dtype
        values = hidden_states.float()
        normalized = values * torch.rsqrt(
            values.square().mean(-1, keepdim=True) + self.epsilon
        )
        return (normalized * self.weight.float()).to(dtype).contiguous()


def build_engine_kwargs(
    *,
    model_path: str | Path,
    tensor_parallel_size: int,
    scratch_root: str | Path,
    max_model_len: int = 131072,
    gpu_memory_utilization: float = 0.90,
) -> dict[str, Any]:
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    return {
        "model": str(model_path),
        "tensor_parallel_size": int(tensor_parallel_size),
        "dtype": "bfloat16",
        "quantization": "ascend",
        "trust_remote_code": True,
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "enable_chunked_prefill": False,
        # Hidden-state extraction must see every requested prompt token. Prefix
        # cache reuse can otherwise return a shorter hidden stream than the
        # exact token-ID replay (vLLM issue #44485).
        "enable_prefix_caching": False,
        "speculative_config": {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [
                        *TARGET_CONTRACT.layer_ids,
                        TARGET_CONTRACT.num_hidden_layers,
                    ]
                }
            },
        },
        "kv_transfer_config": {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": str(scratch_root),
                "allow_custom_save_path": True,
                "use_synchronization_lock": True,
            },
        },
    }


def trajectory_tokens(
    record: Mapping[str, Any],
) -> tuple[list[int], list[bool], str]:
    sample_id = str(record.get("sample_id") or record.get("id") or "")
    if not sample_id:
        raise HiddenContractError("trajectory has no stable sample_id")
    if "input_ids" in record and "loss_mask" in record:
        ids = [int(value) for value in record["input_ids"]]
        mask = [bool(value) for value in record["loss_mask"]]
    elif "prompt_token_ids" in record and "response_token_ids" in record:
        prompt = [int(value) for value in record["prompt_token_ids"]]
        response = [int(value) for value in record["response_token_ids"]]
        ids = prompt + response
        mask = [False] * len(prompt) + [True] * len(response)
    else:
        raise HiddenContractError(
            "Stage B requires exact input_ids/loss_mask or prompt/response token IDs"
        )
    if not ids or len(ids) != len(mask) or not any(mask):
        raise HiddenContractError("trajectory token IDs/loss mask are invalid")
    if any(value < 0 or value >= TARGET_CONTRACT.vocab_size for value in ids):
        raise HiddenContractError(
            "trajectory contains a token ID outside the formal GLM vocabulary"
        )
    return ids, mask, sample_id


def load_connector_tensors(
    path: str | Path,
    expected_token_ids: Sequence[int],
    *,
    normalizer: FinalRMSNorm,
    hidden_size: int = 6144,
    tensor_loader: Callable[[str], Mapping[str, torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    tensors = (
        load_file(str(path), device="cpu")
        if tensor_loader is None
        else dict(tensor_loader(str(path)))
    )
    if "token_ids" not in tensors or "hidden_states" not in tensors:
        raise HiddenContractError("connector must return token_ids and hidden_states")
    token_ids = tensors["token_ids"].to(torch.int64).reshape(-1)
    expected = torch.tensor(list(expected_token_ids), dtype=torch.int64)
    if not torch.equal(token_ids.cpu(), expected):
        raise HiddenContractError("connector token IDs do not exactly match Stage A")
    hidden = tensors["hidden_states"]
    # vLLM's extract_hidden_states connector returns the requested streams in
    # one tensor.  The final requested layer is the raw post-block state; it
    # must be normalized here with the real checkpoint RMSNorm.  Accepting a
    # pre-normalized side channel would make Stage B semantics connector
    # dependent, so fail closed instead.
    if hidden.ndim == 3 and hidden.shape[1] == len(TARGET_CONTRACT.layer_ids) + 1:
        aux = hidden[:, :-1]
        final = normalizer(hidden[:, -1])
    else:
        raise HiddenContractError(
            "connector must return five auxiliary streams and raw layer-78 output"
        )
    expected_aux = (len(expected_token_ids), len(TARGET_CONTRACT.layer_ids), hidden_size)
    expected_final = (len(expected_token_ids), hidden_size)
    if tuple(aux.shape) != expected_aux or tuple(final.shape) != expected_final:
        raise HiddenContractError(
            f"hidden shape mismatch: {tuple(aux.shape)}/{tuple(final.shape)}"
        )
    if aux.dtype != torch.bfloat16 or final.dtype != torch.bfloat16:
        raise HiddenContractError("Stage-B hidden streams must be BF16")
    if not bool(torch.isfinite(aux.float()).all()) or not bool(
        torch.isfinite(final.float()).all()
    ):
        raise HiddenContractError("hidden streams contain NaN or Inf")
    return {
        "token_ids": token_ids,
        "aux_hidden_states": aux.contiguous(),
        "target_final_hidden": final.contiguous(),
    }


def load_final_normalizer(
    model_root: str | Path,
    *,
    weight_key: str = "model.norm.weight",
    epsilon: float | None = None,
) -> FinalRMSNorm:
    root = Path(model_root)
    if epsilon is None:
        config_path = root / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            raw_epsilon = config.get("rms_norm_eps", 1e-5)
            if not isinstance(raw_epsilon, (int, float)) or float(raw_epsilon) <= 0:
                raise HiddenContractError("config.rms_norm_eps must be positive")
            epsilon = float(raw_epsilon)
        else:
            epsilon = 1e-5
    indexes = sorted(root.rglob("*.safetensors.index.json"))
    if indexes:
        mapping = json.loads(indexes[0].read_text(encoding="utf-8")).get("weight_map", {})
        filename = mapping.get(weight_key)
        if not filename:
            raise HiddenContractError(f"final norm tensor {weight_key!r} is absent")
        path = (root / str(filename)).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise HiddenContractError("final norm shard is outside the model directory")
        with safe_open(path, framework="pt", device="cpu") as handle:
            if weight_key not in handle.keys():
                raise HiddenContractError(f"final norm tensor {weight_key!r} is absent")
            weight = handle.get_tensor(weight_key)
    else:
        matches: list[tuple[Path, torch.Tensor]] = []
        for path in sorted(root.rglob("*.safetensors")):
            resolved = path.resolve()
            if root.resolve() not in resolved.parents:
                raise HiddenContractError("final norm shard is outside the model directory")
            with safe_open(resolved, framework="pt", device="cpu") as handle:
                if weight_key in handle.keys():
                    matches.append((resolved, handle.get_tensor(weight_key)))
        if len(matches) != 1:
            raise HiddenContractError(
                f"expected exactly one {weight_key!r} tensor, found {len(matches)}"
            )
        weight = matches[0][1]
    return FinalRMSNorm(weight, float(epsilon))


def native_connector_loader(path: str) -> Mapping[str, torch.Tensor]:
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        example_hidden_states_connector,
    )

    return example_hidden_states_connector.load_hidden_states(path)


def cleanup_connector_artifact(path: str | Path) -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        example_hidden_states_connector,
    )

    example_hidden_states_connector.cleanup_hidden_states(str(path))
