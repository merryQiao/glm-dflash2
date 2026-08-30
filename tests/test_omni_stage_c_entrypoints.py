from __future__ import annotations

from pathlib import Path

from tools.train_thinker_drafter import recipe_for, run_tiny_smoke


ROOT = Path(__file__).resolve().parents[1]


def test_recipe_freezes_512_anchors_seed42_window4096():
    recipe = recipe_for("dflash2", 16)
    assert recipe.anchors_per_sample == 512
    assert recipe.seed == 42
    assert recipe.max_window_tokens == 4096
    assert recipe.epochs == 3
    assert recipe.learning_rate == 6e-4
    assert recipe.gradient_accumulation == 8


def test_tiny_smoke_all_five_routes():
    for method, block_size in (
        ("dflash", 8),
        ("dflash", 16),
        ("dflash2", 8),
        ("dflash2", 16),
        ("dspark", 8),
    ):
        result = run_tiny_smoke(method=method, block_size=block_size)
        assert result["finite_loss"]
        assert result["optimizer_steps"] == 1


def test_launcher_uses_npu_hccl_fsdp2():
    launcher = (ROOT / "scripts/train_thinker_drafter.sh").read_text()
    assert "common_ascend.sh" in launcher
    assert "--device npu" in launcher
    assert "--backend hccl" in launcher
    assert "--strategy fsdp2" in launcher
