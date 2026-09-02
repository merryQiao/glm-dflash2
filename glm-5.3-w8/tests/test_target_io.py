from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from glm53_w8.target_io import extract_w8a8_target_io, load_frozen_target_io


def _checkpoint(
    root: Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    nested_description: bool = False,
) -> Path:
    root.mkdir()
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 192,
        "vocab_size": 7,
        "tie_word_embeddings": False,
        "quantize": "w8a8",
        "quantization_config": {"model_quant_type": "W8A8"},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if nested_description:
        # Real ModelSlim exports put the authoritative per-weight ABI in this
        # sidecar rather than in config.json.
        (root / "quant_model_description.json").write_text(
            json.dumps(
                {
                    "model.layers.0.self_attn.q_proj.weight": {
                        "quant_type": "W8A8_DYNAMIC",
                        "params": {"dtype": "int8"},
                    },
                    "optional": {"format": "ascend_v1"},
                }
            ),
            encoding="utf-8",
        )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    save_file(
        {
            "model.embed_tokens.weight": torch.randn(7, 4, dtype=dtype),
            "lm_head.weight": torch.randn(7, 4, dtype=dtype),
        },
        root / "model.safetensors",
    )
    return root


def test_extracts_one_dense_bf16_artifact(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "source")
    output = tmp_path / "io"
    manifest = extract_w8a8_target_io(
        source,
        output,
        expected_hidden_size=4,
        expected_vocab_size=7,
    )
    assert manifest["target_quantization"] == "W8A8"
    assert manifest["storage_dtype"] == "bfloat16"
    frozen = load_frozen_target_io(output)
    assert frozen.embed_tokens.weight.dtype == torch.bfloat16
    assert frozen.lm_head.weight.dtype == torch.bfloat16
    assert not frozen.embed_tokens.weight.requires_grad
    assert not frozen.lm_head.weight.requires_grad


def test_extracts_from_nested_modelslim_description(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "source", nested_description=True)
    # Remove the smoke-only marker from config.json; the sidecar alone must
    # identify this as a ModelSlim W8A8 export.
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["quantize"] = None
    config["quantization_config"] = {}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = extract_w8a8_target_io(
        source,
        tmp_path / "io",
        expected_hidden_size=4,
        expected_vocab_size=7,
    )
    assert manifest["target_quantization"] == "W8A8"


def test_refuses_quantized_or_fp32_io_tensors(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "source", dtype=torch.float32)
    with pytest.raises(ValueError, match="BF16"):
        extract_w8a8_target_io(
            source,
            tmp_path / "io",
            expected_hidden_size=4,
            expected_vocab_size=7,
        )


def test_rejects_target_io_manifest_without_identity_provenance(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "source")
    output = tmp_path / "io"
    extract_w8a8_target_io(source, output, expected_hidden_size=4, expected_vocab_size=7)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("source_model_fingerprint")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity provenance"):
        load_frozen_target_io(output)
