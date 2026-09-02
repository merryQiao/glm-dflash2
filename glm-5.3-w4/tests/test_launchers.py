from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_all_ascend_launchers_are_strict_and_executable() -> None:
    scripts = sorted((ROOT / "scripts").glob("*.sh"))
    assert {path.name for path in scripts} == {
        "extract_target_io.sh",
        "run_stage_b_hidden.sh",
        "smoke_no_npu.sh",
        "train_drafter.sh",
    }
    for script in scripts:
        assert script.stat().st_mode & 0o111
        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


def test_stage_b_launcher_contains_w4a8_hidden_contract() -> None:
    text = (ROOT / "scripts/run_stage_b_hidden.sh").read_text()
    assert "extract_hidden_vllm_ascend.py" in text
    assert "--tp-size" in text
    assert "TARGET_IO_DIR" in text
    assert "TRAJECTORY_JSONL" in text


def test_training_launcher_exposes_all_fixed_stage_c_knobs() -> None:
    text = (ROOT / "scripts/train_drafter.sh").read_text()
    for value in (
        "METHOD",
        "BLOCK_SIZE",
        "MASK_TOKEN_ID",
        "HIDDEN_CACHE",
        "TARGET_IO_DIR",
        "NPROC_PER_NODE",
        "fsdp2",
        "--gradient-accumulation",
        "--checkpoint-every",
    ):
        assert value in text
