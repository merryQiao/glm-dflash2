from __future__ import annotations

from pathlib import Path

import torch

from glm53_w4.hidden_cache import HiddenCacheDataset, HiddenShardWriter


def test_hidden_cache_roundtrip_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    provenance = {
        "target_quantization": "W4A8",
        "runtime_backend": "vllm-ascend",
        "source_model_fingerprint": "model-x",
        "tokenizer_fingerprint": "tok-x",
    }
    with HiddenShardWriter(
        root,
        hidden_size=4,
        layer_ids=(1, 20, 38, 56, 75),
        provenance=provenance,
        max_shard_bytes=1 << 20,
    ) as writer:
        writer.append(
            sample_id="a",
            input_ids=torch.tensor([1, 2, 3]),
            loss_mask=torch.tensor([False, True, True]),
            aux_hidden_states=torch.randn(3, 5, 4, dtype=torch.bfloat16),
            target_final_hidden=torch.randn(3, 4, dtype=torch.bfloat16),
        )
        writer.freeze()
    dataset = HiddenCacheDataset(root)
    assert len(dataset) == 1
    row = dataset[0]
    assert row["sample_id"] == "a"
    assert row["aux_hidden_states"].shape == (3, 5, 4)
    assert row["target_final_hidden"].shape == (3, 4)
    assert dataset.manifest["provenance"] == provenance


def test_window_read_does_not_materialize_full_hidden(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    provenance = {
        "target_quantization": "W4A8",
        "runtime_backend": "vllm-ascend",
    }
    with HiddenShardWriter(
        root,
        hidden_size=4,
        layer_ids=(1, 20, 38, 56, 75),
        provenance=provenance,
    ) as writer:
        writer.append(
            sample_id="long",
            input_ids=torch.arange(20),
            loss_mask=torch.ones(20, dtype=torch.bool),
            aux_hidden_states=torch.randn(20, 5, 4, dtype=torch.bfloat16),
            target_final_hidden=torch.randn(20, 4, dtype=torch.bfloat16),
        )
        writer.freeze()
    dataset = HiddenCacheDataset(root)
    tokens = dataset.token_fields(0)
    assert tokens["input_ids"].shape == (20,)
    row = dataset.get_window(0, 7, 12)
    assert row["input_ids"].tolist() == list(range(7, 12))
    assert row["aux_hidden_states"].shape == (5, 5, 4)
    assert row["position_offset"] == 7
