from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import types

import pytest
import yaml

from tests.helpers import complete_config
from omni_sd.inference_profile import (
    ProfileContractError,
    aggregate_evaluation,
    aggregate_profile_performance,
    component_availability,
    normalize_input_record,
    profile_batch_kind,
    reduce_hbm_measurements,
    request_latency_seconds,
    score_prediction,
    validate_evaluation_contract,
    worker_reset_npu_peak,
    worker_snapshot_npu_memory,
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


def test_evaluation_metadata_survives_normalization_without_entering_messages(
    tmp_path: Path,
):
    row = normalize_input_record(
        {
            "id": "evaluated",
            "text": "Answer the question",
            "evaluation": {
                "metric": "normalized_exact_match",
                "reference": "Café!",
            },
        },
        index=0,
        base_dir=tmp_path,
    )

    assert json.loads(row["evaluation_json"]) == {
        "metric": "normalized_exact_match",
        "reference": "Café!",
    }
    assert "evaluation" not in json.loads(row["messages_json"])[0]


@pytest.mark.parametrize(
    ("metric", "prediction", "reference", "expected"),
    [
        ("exact_match", "Answer ", "Answer", False),
        ("exact_match", "Answer", "Answer", True),
        ("normalized_exact_match", "  CAFÉ—test! ", "cafe\u0301 test", True),
        ("multiple_choice_accuracy", "B", "B", True),
        ("multiple_choice_accuracy", "Answer: b.", "B", True),
        ("multiple_choice_accuracy", "I choose B", "B", False),
    ],
)
def test_omni_eval_v1_scoring(metric, prediction, reference, expected):
    assert score_prediction(metric, prediction, reference) is expected


def test_multiple_choice_reference_must_be_one_ascii_letter():
    with pytest.raises(ProfileContractError, match="ASCII A-Z"):
        score_prediction("multiple_choice_accuracy", "A", "AA")


def test_evaluation_aggregates_partial_references_and_unavailable_modality():
    summary = aggregate_evaluation(
        [
            {
                "modality": "text",
                "response_text": "yes",
                "evaluation_json": json.dumps(
                    {"metric": "exact_match", "reference": "yes"}
                ),
            },
            {
                "modality": "text",
                "response_text": "ignored",
                "evaluation_json": None,
            },
            {
                "modality": "image",
                "response_text": "ignored",
                "evaluation_json": None,
            },
        ]
    )

    assert summary["available"] is True
    assert summary["metric"] == "exact_match"
    assert summary["overall"] == {
        "available": True,
        "evaluated": 1,
        "skipped": 2,
        "correct": 1,
        "accuracy": 1.0,
    }
    assert summary["by_modality"]["text"]["evaluated"] == 1
    assert summary["by_modality"]["text"]["skipped"] == 1
    assert summary["by_modality"]["image"] == {
        "available": False,
        "evaluated": 0,
        "skipped": 1,
        "reason": "no evaluation references",
    }


def test_evaluation_rejects_mixed_metrics():
    rows = [
        {
            "modality": "text",
            "response_text": "x",
            "evaluation_json": json.dumps({"metric": metric, "reference": "x"}),
        }
        for metric in ("exact_match", "normalized_exact_match")
    ]
    with pytest.raises(ProfileContractError, match="mixed evaluation metrics"):
        aggregate_evaluation(rows)


def test_evaluation_contract_rejects_mixed_metrics_before_inference():
    with pytest.raises(ProfileContractError, match="mixed evaluation metrics"):
        validate_evaluation_contract(rows=[
            {
                "evaluation_json": json.dumps(
                    {"metric": "exact_match", "reference": "yes"}
                )
            },
            {
                "evaluation_json": json.dumps(
                    {"metric": "normalized_exact_match", "reference": "yes"}
                )
            },
        ])


def test_evaluation_is_unavailable_when_all_references_are_missing():
    summary = aggregate_evaluation(
        [{"modality": "text", "response_text": "x", "evaluation_json": None}]
    )
    assert summary == {
        "available": False,
        "scorer_version": "omni_eval_v1",
        "evaluated": 0,
        "skipped": 1,
        "reason": "no evaluation references",
    }


def test_profile_performance_uses_weighted_engine_and_end_to_end_denominators():
    summary = aggregate_profile_performance(
        requests=[
            {
                "modality": "text",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "preprocess_seconds": 0.1,
                "engine_request_seconds": 0.5,
            },
            {
                "modality": "audio",
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "preprocess_seconds": 0.3,
                "engine_request_seconds": None,
            },
        ],
        batches=[
            {
                "modality": "text",
                "requests": 1,
                "engine_seconds": 1.0,
                "end_to_end_seconds": 1.2,
            },
            {
                "modality": "audio",
                "requests": 1,
                "engine_seconds": 3.0,
                "end_to_end_seconds": 3.5,
            },
        ],
        model_load_seconds=99.0,
        warmup={
            "preprocess_seconds": 1.0,
            "engine_seconds": 9.0,
            "end_to_end_seconds": 11.0,
        },
    )

    overall = summary["overall"]
    assert overall["requests"] == 2
    assert overall["prompt_tokens"] == 30
    assert overall["completion_tokens"] == 20
    assert overall["engine_seconds"] == 4.0
    assert overall["end_to_end_seconds"] == 4.7
    assert overall["requests_per_second"] == pytest.approx(0.5)
    assert overall["completion_tokens_per_second"] == pytest.approx(5.0)
    assert overall["total_tokens_per_second"] == pytest.approx(12.5)
    assert overall["engine"]["completion_tokens_per_second"] == pytest.approx(5.0)
    assert overall["end_to_end"]["completion_tokens_per_second"] == pytest.approx(
        20 / 4.7
    )
    assert overall["request_preprocess_latency_ms"]["mean"] == pytest.approx(200)
    assert overall["request_engine_latency_ms"]["available"] is True
    assert overall["request_engine_latency_ms"]["observed"] == 1
    assert overall["request_engine_latency_ms"]["missing"] == 1
    assert summary["by_modality"]["text"]["completion_tokens_per_second"] == 5.0
    assert summary["by_modality"]["audio"][
        "completion_tokens_per_second"
    ] == 5.0
    assert summary["model_load_seconds"] == 99.0
    assert summary["warmup"]["end_to_end_seconds"] == 11.0
    assert overall["batch_engine_latency_ms"]["p50"] == pytest.approx(2000.0)
    assert overall["batch_end_to_end_latency_ms"]["p50"] == pytest.approx(2350.0)


def test_empty_measurement_is_rejected():
    with pytest.raises(ProfileContractError, match="no measured requests"):
        aggregate_profile_performance(
            requests=[],
            batches=[],
            model_load_seconds=1.0,
            warmup={
                "preprocess_seconds": 0.0,
                "engine_seconds": 0.0,
                "end_to_end_seconds": 0.0,
            },
        )


def test_actual_payload_classifies_every_supported_batch_kind(tmp_path: Path):
    paths = {}
    for media_type, suffix in (("image", "jpg"), ("audio", "wav"), ("video", "mp4")):
        path = tmp_path / f"sample.{suffix}"
        path.write_bytes(media_type.encode())
        paths[media_type] = str(path)

    def row(**media):
        return normalize_input_record(
            {"text": "inspect", **media}, index=0, base_dir=tmp_path
        )

    assert profile_batch_kind(row()) == "text"
    assert profile_batch_kind(row(image=paths["image"])) == "image"
    assert profile_batch_kind(row(image=[paths["image"], paths["image"]])) == "multi_image"
    assert profile_batch_kind(row(audio=paths["audio"])) == "audio"
    assert profile_batch_kind(row(video=paths["video"])) == "video"
    assert profile_batch_kind(
        row(image=paths["image"], audio=paths["audio"])
    ) == "other"


def test_component_availability_distinguishes_loaded_from_timed():
    components = component_availability(["text", "audio", "image", "video"])
    assert components["audio_encoder"] == {
        "loaded": True,
        "executed": True,
        "timing_available": False,
        "reason": "vLLM-Ascend does not expose request-scoped audio encoder events",
    }
    assert components["vision_encoder"]["executed"] is True
    assert components["thinker"]["executed"] is True
    for name in ("talker", "mtp", "code2wav"):
        assert components[name]["loaded"] is False
        assert components[name]["executed"] is False
        assert components[name]["timing_available"] is False
        assert components[name]["reason"]


def test_hbm_reduction_preserves_per_rank_current_and_batch_peak_semantics():
    batches = [
        [
            {
                "rank": 0,
                "logical_device": 0,
                "physical_device": 4,
                "allocated_current_bytes": 100,
                "reserved_current_bytes": 120,
                "allocated_peak_bytes": 150,
                "reserved_peak_bytes": 180,
            },
            {
                "rank": 1,
                "logical_device": 1,
                "physical_device": 5,
                "allocated_current_bytes": 200,
                "reserved_current_bytes": 240,
                "allocated_peak_bytes": 260,
                "reserved_peak_bytes": 300,
            },
        ],
        [
            {
                "rank": 0,
                "logical_device": 0,
                "physical_device": 4,
                "allocated_current_bytes": 110,
                "reserved_current_bytes": 130,
                "allocated_peak_bytes": 170,
                "reserved_peak_bytes": 190,
            },
            {
                "rank": 1,
                "logical_device": 1,
                "physical_device": 5,
                "allocated_current_bytes": 190,
                "reserved_current_bytes": 230,
                "allocated_peak_bytes": 250,
                "reserved_peak_bytes": 290,
            },
        ],
    ]

    summary = reduce_hbm_measurements(batches, tensor_parallel_size=2)

    assert summary["available"] is True
    assert summary["source"] == "torch_npu_allocator"
    assert summary["per_rank"]["0"]["final_current"]["allocated_bytes"] == 110
    assert summary["per_rank"]["1"]["max_post_batch_current"][
        "allocated_bytes"
    ] == 200
    assert summary["per_rank"]["0"]["max_batch_peak"]["allocated_bytes"] == 170
    assert summary["max_rank_peak"]["allocated_bytes"] == 260
    assert summary["max_batch_sum_of_rank_peaks"]["allocated_bytes"] == 420
    assert summary["max_batch_sum_of_rank_peaks"]["reserved_bytes"] == 480


@pytest.mark.parametrize(
    ("batches", "message"),
    [
        (
            [[{"rank": 0, "physical_device": 4}]],
            "exactly 2 worker ranks",
        ),
        (
            [
                [
                    {"rank": 0, "physical_device": 4},
                    {"rank": 1, "physical_device": 4},
                ]
            ],
            "unique physical NPU",
        ),
    ],
)
def test_hbm_reduction_rejects_partial_or_duplicate_workers(batches, message):
    with pytest.raises(ProfileContractError, match=message):
        reduce_hbm_measurements(batches, tensor_parallel_size=2)


def test_worker_hbm_rpc_uses_rank_local_torch_npu_allocator(monkeypatch):
    class FakeNPU:
        def __init__(self):
            self.reset_calls = []
            self.sync_calls = []

        def current_device(self):
            return 1

        def synchronize(self, device=None):
            self.sync_calls.append(device)

        def reset_peak_memory_stats(self, device=None):
            self.reset_calls.append(device)

        def memory_allocated(self, device=None):
            return 101

        def memory_reserved(self, device=None):
            return 202

        def max_memory_allocated(self, device=None):
            return 303

        def max_memory_reserved(self, device=None):
            return 404

    fake_npu = FakeNPU()
    fake_dist = types.SimpleNamespace(is_initialized=lambda: True, get_rank=lambda: 1)
    fake_torch = types.SimpleNamespace(npu=fake_npu, distributed=fake_dist)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,7")

    reset = worker_reset_npu_peak(object())
    snapshot = worker_snapshot_npu_memory(object())

    assert reset == {"rank": 1, "logical_device": 1, "physical_device": 7}
    assert fake_npu.reset_calls == [1]
    assert snapshot == {
        "rank": 1,
        "logical_device": 1,
        "physical_device": 7,
        "allocated_current_bytes": 101,
        "reserved_current_bytes": 202,
        "allocated_peak_bytes": 303,
        "reserved_peak_bytes": 404,
    }


def test_request_latency_uses_only_vllm_metrics_without_batch_fallback():
    class Metrics:
        arrival_time = 10.0
        finished_time = 10.75

    class Output:
        metrics = Metrics()

    assert request_latency_seconds(Output()) == pytest.approx(0.75)
    assert request_latency_seconds(object()) is None


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
            self.rpc_calls = []

        def generate(self, requests, sampling_params, use_tqdm):
            self.calls.append((requests, sampling_params, use_tqdm))
            return [Output() for _ in requests]

        def collective_rpc(self, function):
            self.rpc_calls.append(function.__name__)
            if function.__name__ == "worker_reset_npu_peak":
                return [
                    {"rank": rank, "logical_device": rank, "physical_device": rank}
                    for rank in range(4)
                ]
            return [
                {
                    "rank": rank,
                    "logical_device": rank,
                    "physical_device": rank,
                    "allocated_current_bytes": 100 + rank,
                    "reserved_current_bytes": 200 + rank,
                    "allocated_peak_bytes": 300 + rank,
                    "reserved_peak_bytes": 400 + rank,
                }
                for rank in range(4)
            ]

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
    ticks = iter(index / 10 for index in range(100))
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(ticks))

    rows = [
        normalize_input_record(
            {
                "id": f"row-{index}",
                "text": "hello",
                "evaluation": {
                    "metric": "exact_match",
                    "reference": "ok",
                },
            },
            index=index,
            base_dir=root,
        )
        for index in range(2)
    ]
    records, profile = cli.run_profile(
        complete_config(), rows, complete_config()["vllm_batch_size"], warmup=1
    )

    assert len(llm.calls) == 2  # one warmup plus one measured batch
    overall = profile["performance"]["overall"]
    assert overall["requests"] == 2
    assert overall["completion_tokens"] == 6
    assert overall["preprocess_seconds"] == pytest.approx(0.2)
    assert overall["engine_seconds"] == pytest.approx(0.1)
    assert overall["end_to_end_seconds"] == pytest.approx(0.7)
    assert profile["performance"]["warmup"]["engine_seconds"] == pytest.approx(0.1)
    assert profile["performance"]["by_modality"]["text"]["requests"] == 2
    assert profile["evaluation"]["overall"]["accuracy"] == 1.0
    assert profile["components"]["talker"]["loaded"] is False
    assert profile["memory"]["source"] == "torch_npu_allocator"
    assert profile["memory"]["max_rank_peak"]["allocated_bytes"] == 303
    assert profile["variant_identity"] == {"kind": "target_only"}
    assert profile["comparison_identity"]["fingerprint"]
    assert [record["prompt_token_ids"] for record in records] == [[10, 11], [10, 11]]
    assert [record["response_token_ids"] for record in records] == [
        [20, 21, 151645],
        [20, 21, 151645],
    ]
    assert all(record["evaluation_result"]["correct"] for record in records)
    assert records[0]["preprocess_latency_ms"] == pytest.approx(100.0)
    assert records[0]["request_engine_latency_ms"] == pytest.approx(250.0)
    assert records[0]["batch_engine_latency_ms"] == pytest.approx(100.0)
    assert records[0]["batch_end_to_end_latency_ms"] == pytest.approx(700.0)
    assert llm.rpc_calls == ["worker_reset_npu_peak", "worker_snapshot_npu_memory"]
    measured_params = llm.calls[1][1]
    assert measured_params[0].kwargs["seed"] != measured_params[1].kwargs["seed"]

    def unavailable_rpc(_function):
        raise AttributeError("collective_rpc unavailable")

    monkeypatch.setattr(llm, "collective_rpc", unavailable_rpc)
    _, missing_hbm_profile = cli.run_profile(
        complete_config(),
        rows,
        complete_config()["vllm_batch_size"],
        warmup=0,
        allow_missing_hbm=True,
    )
    assert missing_hbm_profile["memory"]["available"] is False
    assert missing_hbm_profile["memory"]["source"] == "torch_npu_allocator"


def _load_inference_cli():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        f"omni_inference_cli_{id(root)}", root / "inference_qwen3-omni.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_identity_binds_local_media_bytes(tmp_path: Path):
    cli = _load_inference_cli()
    image = tmp_path / "image.bin"
    image.write_bytes(b"first")
    row = normalize_input_record(
        {"id": "media", "text": "describe", "image": str(image)},
        index=0,
        base_dir=tmp_path,
    )
    arguments = {
        "config": complete_config(),
        "rows": [row],
        "sizes": complete_config()["vllm_batch_size"],
        "warmup": 1,
        "runtime": {"visible_devices": [0, 1, 2, 3]},
    }

    first = cli._comparison_identity(**arguments)
    image.write_bytes(b"second")
    second = cli._comparison_identity(**arguments)

    assert first["fingerprint"] != second["fingerprint"]


def test_profile_bundle_publishes_checksum_marker_last(tmp_path: Path):
    cli = _load_inference_cli()
    output = tmp_path / "records.jsonl"
    profile = tmp_path / "profile.json"
    marker = tmp_path / "SUCCESS.json"

    cli.write_profile_bundle(
        output_jsonl=output,
        profile_json=profile,
        success_marker=marker,
        records=[{"condition_id": "one"}],
        profile={"status": "PASS"},
    )

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["output_jsonl"]["sha256"] == cli._sha256_file(output)
    assert payload["profile_json"]["sha256"] == cli._sha256_file(profile)


def test_profile_bundle_failure_removes_all_final_artifacts(
    tmp_path: Path, monkeypatch
):
    cli = _load_inference_cli()
    output = tmp_path / "records.jsonl"
    profile = tmp_path / "profile.json"
    marker = tmp_path / "SUCCESS.json"
    real_replace = cli.os.replace
    calls = 0

    def fail_second_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(cli.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected"):
        cli.write_profile_bundle(
            output_jsonl=output,
            profile_json=profile,
            success_marker=marker,
            records=[{"condition_id": "one"}],
            profile={"status": "PASS"},
        )

    assert not output.exists()
    assert not profile.exists()
    assert not marker.exists()
