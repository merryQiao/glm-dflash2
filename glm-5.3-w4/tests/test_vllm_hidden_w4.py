from __future__ import annotations

from pathlib import Path

from glm53_w4.vllm_hidden import build_engine_kwargs


def test_w4a8_vllm_ascend_extractor_uses_generic_ascend_backend(tmp_path: Path) -> None:
    kwargs = build_engine_kwargs(
        model_path="/models/glm53-w4a8",
        tensor_parallel_size=8,
        scratch_root=tmp_path,
    )
    assert kwargs["quantization"] == "ascend"
    assert kwargs["speculative_config"]["method"] == "extract_hidden_states"
