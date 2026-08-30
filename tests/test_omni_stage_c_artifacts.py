from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import save_file

from omni_stage_c.blocks import AnchorPositions, build_training_batch
from omni_stage_c.hidden_cache import PackedThinkerHiddenCache
from omni_stage_c.target_io import validate_official_model_config
from omni_sd.provenance import artifact_record


def official_config() -> dict:
    return {
        "thinker_config": {
            "text_config": {
                "num_hidden_layers": 48,
                "hidden_size": 2048,
                "vocab_size": 152064,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "tie_word_embeddings": False,
            }
        }
    }


def test_target_io_rejects_quantized_biased_or_scaled_head():
    validate_official_model_config(official_config())
    for field, value in (("lm_head_bias", True), ("logit_scale", 0.5),
                         ("final_logit_softcapping", 30.0)):
        config = official_config()
        config["thinker_config"]["text_config"][field] = value
        with pytest.raises(ValueError):
            validate_official_model_config(config)
    config = official_config()
    config["quantization_config"] = {"quant_method": "w8a8"}
    with pytest.raises(ValueError, match="non-quantized"):
        validate_official_model_config(config)


def test_physical_blocks_preserve_exact_cached_mrope():
    tokens, hidden = 7, 4
    mrope = torch.stack((torch.arange(tokens), torch.arange(tokens) + 10,
                         torch.arange(tokens) + 20), dim=-1)
    sample = {
        "sample_id": "x",
        "input_ids": torch.arange(tokens),
        "loss_mask": torch.ones(tokens, dtype=torch.bool),
        "auxiliary_hidden": torch.randn(tokens, 5, hidden),
        "target_final_hidden": torch.randn(tokens, hidden),
        "position_ids": mrope,
    }
    anchors = AnchorPositions(torch.tensor([2]), torch.tensor([True]))
    batch = build_training_batch(sample, anchors, block_size=3,
                                 mask_token_id=31, device=torch.device("cpu"))
    assert batch.position_ids.shape == (3, 1, 10)
    assert torch.equal(batch.position_ids[:, 0, :tokens].T, mrope)
    assert torch.equal(batch.position_ids[:, 0, tokens:].T, mrope[2:5])


def test_stage_b_v3_cache_is_random_access_and_checksum_bound(tmp_path):
    shard = tmp_path / "shards" / "train-00000-000.safetensors"
    index = tmp_path / "shards" / "train-00000-000.index.parquet"
    shard.parent.mkdir()
    tokens = 2
    save_file({
        "offsets": torch.tensor([0, tokens]),
        "input_ids": torch.tensor([3, 4], dtype=torch.int32),
        "loss_mask": torch.tensor([False, True]),
        "target_hidden_states": torch.zeros(tokens, 5, 2048, dtype=torch.bfloat16),
        "target_last_hidden_states": torch.zeros(tokens, 2048, dtype=torch.bfloat16),
        "position_ids": torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int64),
    }, str(shard))
    pq.write_table(pa.Table.from_pylist([{
        "condition_id": "sample", "start": 0, "end": tokens,
        "prompt_tokens": 1, "response_tokens": 1,
    }]), index)
    manifest = {
        "status": "PASS", "schema": "omni-thinker-hidden-cache-v3",
        "hidden_size": 2048, "target_layer_ids": [1, 12, 24, 36, 47],
        "dtype": "bfloat16", "position_layout": "tokens,axes",
        "position_axes": ["temporal", "height", "width"],
        "position_ids_source": "official_transformers_get_rope_index",
        "cache_fingerprint": "cache", "trajectory_generation_fingerprint": "trajectory",
        "files": [{
            "data": artifact_record(shard, relative_to=tmp_path),
            "index": artifact_record(index, relative_to=tmp_path),
        }],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    cache = PackedThinkerHiddenCache(tmp_path)
    row = cache[0]
    assert row["sample_id"] == "sample"
    assert row["position_ids"].shape == (2, 3)
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        PackedThinkerHiddenCache(tmp_path)
