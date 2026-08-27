from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from tests.helpers import complete_config
from omni_sd.inference_profile import (
    ProfileContractError,
    aggregate_performance,
    normalize_input_record,
    profile_batch_kind,
    request_latency_seconds,
)
from omni_sd.vllm_ascend_generation import engine_kwargs, sampling_kwargs


def test_simple_record_uses_stage_a_condition_contract(tmp_path: Path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")

    row = normalize_input_record(
        {"id": "example", "text": "Describe it", "image": "image.jpg"},
        index=0,
        base_dir=tmp_path,
    )

    assert row["condition_id"] == "example"
    assert row["modality"] == "image"
    assert json.loads(row["messages_json"]) == [
        {"role": "user", "content": "Describe it"}
    ]
    assert json.loads(row["media_json"]) == [
        {"type": "image", "path": str(image.resolve())}
    ]
    assert json.loads(row["tools_json"]) == []


def test_native_messages_resolve_relative_multimodal_paths(tmp_path: Path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    row = normalize_input_record(
        {
            "id": "native",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": "sample.wav"},
                        {"type": "text", "text": "Transcribe"},
                    ],
                }
            ],
        },
        index=1,
        base_dir=tmp_path,
    )

    content = json.loads(row["messages_json"])[0]["content"]
    assert content[0]["audio"] == str(audio.resolve())
    assert row["modality"] == "audio"
    assert json.loads(row["media_json"]) == []
    assert profile_batch_kind(row) == "audio"


def test_multiple_native_images_use_conservative_multi_image_batch(tmp_path: Path):
    for name in ("one.jpg", "two.jpg"):
        (tmp_path / name).write_bytes(b"image")
    row = normalize_input_record(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "one.jpg"},
                        {"type": "image", "image": "two.jpg"},
                        {"type": "text", "text": "Compare"},
                    ],
                }
            ]
        },
        index=0,
        base_dir=tmp_path,
    )

    assert profile_batch_kind(row) == "multi_image"


def test_aggregate_performance_excludes_load_and_warmup():
    summary = aggregate_performance(
        prompt_tokens=[10, 20],
        completion_tokens=[5, 15],
        batch_seconds=[1.0, 3.0],
        request_seconds=[0.5, 2.5],
        model_load_seconds=99.0,
        warmup_seconds=11.0,
    )

    assert summary["requests"] == 2
    assert summary["prompt_tokens"] == 30
    assert summary["completion_tokens"] == 20
    assert summary["measured_seconds"] == 4.0
    assert summary["requests_per_second"] == pytest.approx(0.5)
    assert summary["completion_tokens_per_second"] == pytest.approx(5.0)
    assert summary["total_tokens_per_second"] == pytest.approx(12.5)
    assert summary["model_load_seconds"] == 99.0
    assert summary["warmup_seconds"] == 11.0
    assert summary["batch_latency_ms"]["p50"] == pytest.approx(2000.0)
    assert summary["request_latency_ms"]["p95"] == pytest.approx(2400.0)


def test_empty_measurement_is_rejected():
    with pytest.raises(ProfileContractError, match="no measured requests"):
        aggregate_performance(
            prompt_tokens=[],
            completion_tokens=[],
            batch_seconds=[],
            request_seconds=[],
            model_load_seconds=1.0,
            warmup_seconds=0.0,
        )


def test_request_latency_prefers_vllm_metrics_and_falls_back():
    class Metrics:
        arrival_time = 10.0
        finished_time = 10.75

    class Output:
        metrics = Metrics()

    assert request_latency_seconds(Output(), 2.0) == pytest.approx(0.75)
    assert request_latency_seconds(object(), 2.0) == pytest.approx(2.0)


def test_cli_dry_run_reuses_production_provider_without_importing_vllm(
    tmp_path: Path,
):
    root = Path(__file__).resolve().parents[1]
    config = complete_config()
    config_path = tmp_path / "config.yaml"
    profile_path = tmp_path / "profile.json"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "inference_qwen3-omni.py"),
            "--config",
            str(config_path),
            "--text",
            "hello",
            "--dry-run",
            "--profile-json",
            str(profile_path),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["status"] == "DRY_RUN"
    assert profile["requests"] == 1
    assert profile["engine"] == engine_kwargs(config)
    assert profile["sampling"] == sampling_kwargs(config, "command-line")
    assert profile["component"] == "thinker"


def test_measured_profile_excludes_warmup_and_keeps_exact_ids(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "omni_inference_cli", root / "inference_qwen3-omni.py"
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    class Metrics:
        arrival_time = 1.0
        finished_time = 1.25

    class Completion:
        token_ids = [20, 21, 151645]
        text = "ok"
        finish_reason = "stop"

    class Output:
        prompt_token_ids = [10, 11]
        outputs = [Completion()]
        metrics = Metrics()

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class LLM:
        def __init__(self):
            self.calls = []

        def generate(self, requests, sampling_params, use_tqdm):
            self.calls.append((requests, sampling_params, use_tqdm))
            return [Output() for _ in requests]

    llm = LLM()
    monkeypatch.setattr(
        cli,
        "runtime_identity",
        lambda **_: {"visible_devices": [0, 1, 2, 3], "backend": "vllm_ascend"},
    )
    monkeypatch.setattr(cli, "load_engine", lambda _: (llm, object(), SamplingParams))
    monkeypatch.setattr(
        cli,
        "prepare_request",
        lambda row, _config, _processor: ({"prompt": row["condition_id"]}, []),
    )

    rows = [
        normalize_input_record(
            {"id": f"row-{index}", "text": "hello"},
            index=index,
            base_dir=root,
        )
        for index in range(2)
    ]
    records, profile = cli.run_profile(
        complete_config(), rows, complete_config()["vllm_batch_size"], warmup=1
    )

    assert len(llm.calls) == 2  # one warmup plus one measured batch
    assert profile["performance"]["requests"] == 2
    assert profile["performance"]["completion_tokens"] == 6
    assert [record["prompt_token_ids"] for record in records] == [[10, 11], [10, 11]]
    assert [record["response_token_ids"] for record in records] == [
        [20, 21, 151645],
        [20, 21, 151645],
    ]
    measured_params = llm.calls[1][1]
    assert measured_params[0].kwargs["seed"] != measured_params[1].kwargs["seed"]
