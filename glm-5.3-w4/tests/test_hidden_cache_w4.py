from __future__ import annotations

from pathlib import Path

import pytest
import torch

from glm53_w4.hidden_cache import HiddenShardWriter


def test_w4a8_cache_provenance_is_independent(tmp_path: Path) -> None:
    with HiddenShardWriter(
        tmp_path / "cache",
        hidden_size=4,
        layer_ids=(1, 20, 38, 56, 75),
        provenance={"target_quantization": "W4A8", "runtime_backend": "vllm-ascend"},
    ) as writer:
        writer.append(
            sample_id="x",
            input_ids=torch.tensor([1, 2]),
            loss_mask=torch.tensor([False, True]),
            aux_hidden_states=torch.randn(2, 5, 4, dtype=torch.bfloat16),
            target_final_hidden=torch.randn(2, 4, dtype=torch.bfloat16),
        )
        writer.freeze()
    manifest = (tmp_path / "cache" / "manifest.json").read_text()
    assert '"target_quantization":"W4A8"' in manifest
    assert "W8A8" not in manifest
