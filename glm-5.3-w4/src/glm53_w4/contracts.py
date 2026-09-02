from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


DFLASH2_METHOD = "dflash2"
DSPARK_METHOD = "dspark"


@dataclass(frozen=True)
class TargetContract:
    architecture: str = "GlmMoeDsaForCausalLM"
    model_type: str = "glm_moe_dsa"
    layer_ids: tuple[int, ...] = (1, 20, 38, 56, 75)
    num_hidden_layers: int = 78
    hidden_size: int = 6144
    intermediate_size: int = 12288
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    head_dim: int = 192
    vocab_size: int = 154880
    tie_word_embeddings: bool = False


@dataclass(frozen=True)
class DraftContract:
    num_hidden_layers: int = 5
    hidden_size: int = 6144
    intermediate_size: int = 12288
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    head_dim: int = 64
    sliding_window: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 8_000_000.0
    selector_top_k: int = 16
    selector_rank: int = 256
    markov_rank: int = 256


TARGET_CONTRACT = TargetContract()
DRAFT_CONTRACT = DraftContract()


def _scalar_values(value: Any) -> list[Any]:
    """Flatten nested ModelSlim metadata without assuming one schema version.

    ModelSlim stores the actual per-weight quantization types in
    ``quant_model_description.json``; depending on the release, the values may
    be nested under layer/processor dictionaries.  Looking only at one top
    level key silently rejects valid W4A8 exports.
    """

    if isinstance(value, Mapping):
        result: list[Any] = []
        for item in value.values():
            result.extend(_scalar_values(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_scalar_values(item))
        return result
    return [] if value is None else [value]


def _normalized_quantization(
    config: Mapping[str, Any], quant_description: Mapping[str, Any] | None = None
) -> set[str]:
    values: list[Any] = [
        config.get("quantize"),
        config.get("quantization"),
        config.get("quant_method"),
        config.get("quantization_config"),
    ]
    if quant_description is not None:
        values.append(quant_description)
    result: set[str] = set()
    for value in _scalar_values(values):
        result.add(str(value).strip().lower().replace("-", "").replace("_", ""))
    return result


def validate_w4a8_target_config(
    config: Mapping[str, Any],
    *,
    expected_hidden_size: int | None = None,
    expected_vocab_size: int | None = None,
    quant_description: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate formal GLM-5.3 plus an explicit ModelSlim W4A8 marker.

    ``quant_description`` should be the parsed ModelSlim
    ``quant_model_description.json`` when validating a production export;
    ModelSlim generally does not put the W4A8 marker in ``config.json``.
    Tiny shape overrides exist only for unit tests and smoke checkpoints. All
    production launchers call this function without overrides.
    """

    target = TARGET_CONTRACT
    architectures = tuple(str(value) for value in config.get("architectures", ()))
    expected_hidden = target.hidden_size if expected_hidden_size is None else int(
        expected_hidden_size
    )
    expected_vocab = target.vocab_size if expected_vocab_size is None else int(
        expected_vocab_size
    )
    checks = {
        "architecture": target.architecture in architectures,
        "model_type": str(config.get("model_type", "")) == target.model_type,
        "hidden_size": int(config.get("hidden_size", -1)) == expected_hidden,
        "intermediate_size": int(config.get("intermediate_size", -1))
        == (target.intermediate_size if expected_hidden_size is None else expected_hidden * 2),
        "num_hidden_layers": int(config.get("num_hidden_layers", -1))
        == target.num_hidden_layers,
        "num_attention_heads": int(config.get("num_attention_heads", -1))
        == target.num_attention_heads,
        "num_key_value_heads": int(config.get("num_key_value_heads", -1))
        == target.num_key_value_heads,
        "head_dim": int(config.get("head_dim", -1)) == target.head_dim,
        "vocab_size": int(config.get("vocab_size", -1)) == expected_vocab,
        "untied_io": bool(config.get("tie_word_embeddings", True)) is False,
    }
    failures = [name for name, valid in checks.items() if not valid]
    if failures:
        raise ValueError(
            "checkpoint is not the formal GLM-5.3 target contract: "
            + ", ".join(failures)
        )
    quantization = _normalized_quantization(config, quant_description)
    # ModelSlim may suffix the marker with the granularity (for example
    # ``W4A8_DYNAMIC_PER_GROUP``), but W4A8C8/W8A8/FP8 are different runtime
    # contracts and must never pass this check by substring accident.
    compatible = {
        value
        for value in quantization
        if "w4a8" in value and not any(
            token in value for token in ("w4a8c8", "w8a8", "fp8")
        )
    }
    incompatible = {
        value
        for value in quantization
        if any(token in value for token in ("w4a8c8", "w8a8", "fp8"))
    }
    if incompatible or not compatible:
        raise ValueError(
            "formal GLM-5.3 checkpoint must carry ModelSlim W4A8 metadata; "
            f"found {sorted(quantization)}"
        )
    return {
        "schema": "formal-glm53-w4a8-v2",
        "model_type": target.model_type,
        "quantization": "W4A8",
        "target_contract": asdict(target),
    }


def validate_method_contract(method: str, *, block_size: int) -> None:
    method = str(method).lower()
    block_size = int(block_size)
    if method == DSPARK_METHOD:
        if block_size != 8:
            raise ValueError("DSpark uses physical block size 8 only")
        return
    if method == DFLASH2_METHOD:
        if block_size not in (8, 16):
            raise ValueError("DFlash2 block size must be 8 or 16")
        return
    raise ValueError(f"unsupported Stage-C method: {method}")
