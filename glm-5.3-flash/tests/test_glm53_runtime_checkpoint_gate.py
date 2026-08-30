from __future__ import annotations

import importlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def _distributed_module():
    return importlib.import_module("glm53_drafters.distributed")


def _checkpointing_module():
    return importlib.import_module("glm53_drafters.checkpointing")


def _capability_module():
    return importlib.import_module("glm53_drafters.capability")


def _seed_everything(rank: int) -> None:
    random.seed(701 + rank)
    np.random.seed(811 + rank)
    torch.manual_seed(919 + rank)


def _new_training_state(rank: int):
    _seed_everything(rank)
    model = nn.parallel.DistributedDataParallel(
        nn.Sequential(nn.Linear(3, 5), nn.SiLU(), nn.Linear(5, 2))
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 / (step + 1)
    )
    anchor = torch.Generator().manual_seed(1031 + rank)
    return model, optimizer, scheduler, anchor


def _train_steps(model, optimizer, scheduler, anchor, *, rank: int, start: int, stop: int):
    for cursor in range(start, stop):
        optimizer.zero_grad(set_to_none=True)
        stochastic = (
            random.random()
            + float(np.random.random())
            + float(torch.rand(()))
            + float(torch.rand((), generator=anchor))
        )
        value = torch.full((4, 3), float(rank + cursor + 1)) + stochastic * 0.01
        target = torch.full((4, 2), float(cursor % 2))
        (model(value) - target).square().mean().backward()
        optimizer.step()
        scheduler.step()


def _nested_close(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and bool(torch.equal(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _nested_close(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _nested_close(a, b) for a, b in zip(left, right)
        )
    return left == right


def _resume_parity_worker(rank: int, rendezvous: str, checkpoint: str, reports: str):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        checkpointing = _checkpointing_module()

        baseline = _new_training_state(rank)
        _train_steps(*baseline, rank=rank, start=0, stop=4)
        baseline_model, baseline_optimizer, baseline_scheduler, baseline_anchor = baseline
        baseline_state = {
            "model": {key: value.detach().clone() for key, value in baseline_model.state_dict().items()},
            "optimizer": baseline_optimizer.state_dict(),
            "scheduler": baseline_scheduler.state_dict(),
            "python_next": random.random(),
            "numpy_next": float(np.random.random()),
            "torch_next": torch.rand(3),
            "anchor_next": torch.rand(3, generator=baseline_anchor),
        }

        interrupted = _new_training_state(rank)
        _train_steps(*interrupted, rank=rank, start=0, stop=2)
        model, optimizer, scheduler, anchor = interrupted
        progress = checkpointing.TrainingProgress(
            global_step=2, micro_step=0, epoch=0, sampler_cursor=2
        )
        checkpointing.save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
            anchor_generator=anchor,
            at_optimizer_boundary=True,
            semantic_config={"cache_identity": "cache-a", "recipe": "dflash-b8"},
        )

        resumed = _new_training_state(rank)
        model, optimizer, scheduler, anchor = resumed
        restored = checkpointing.load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            anchor_generator=anchor,
            expected_semantic_config={
                "cache_identity": "cache-a",
                "recipe": "dflash-b8",
            },
        )
        _train_steps(
            model,
            optimizer,
            scheduler,
            anchor,
            rank=rank,
            start=restored.sampler_cursor,
            stop=4,
        )
        actual_state = {
            "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_next": random.random(),
            "numpy_next": float(np.random.random()),
            "torch_next": torch.rand(3),
            "anchor_next": torch.rand(3, generator=anchor),
        }
        report = {
            "passed": _nested_close(actual_state, baseline_state),
            "restored_progress": {
                "global_step": restored.global_step,
                "micro_step": restored.micro_step,
                "epoch": restored.epoch,
                "sampler_cursor": restored.sampler_cursor,
            },
        }
        Path(reports, f"rank-{rank}.json").write_text(json.dumps(report))
    finally:
        dist.destroy_process_group()


def test_device_resolution_is_cpu_gloo_or_lazy_npu_hccl(monkeypatch):
    distributed = _distributed_module()
    device, backend = distributed.resolve_device_backend("cpu", local_rank=0)
    assert device == torch.device("cpu")
    assert backend == "gloo"

    imported = []
    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        set_device=lambda rank: imported.append(("set", rank)),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    with mock.patch.object(
        distributed.importlib, "import_module", side_effect=lambda name: imported.append(name)
    ):
        device, backend = distributed.resolve_device_backend("npu", local_rank=3)
    assert device.index == 3
    assert device.type in {"npu", "privateuseone"}
    assert backend == "hccl"
    assert imported == ["torch_npu", ("set", 3)]


def test_fsdp2_wrap_policy_covers_layers_and_method_heads():
    distributed = _distributed_module()

    class Draft(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
            self.candidate_selector = nn.Linear(2, 2)
            self.markov_head = nn.Linear(2, 2)
            self.confidence_head = nn.Linear(2, 1)

    targets = distributed.fsdp2_wrap_policy(Draft())
    assert [(item.name, item.reshard_after_forward) for item in targets] == [
        ("layers.0", True),
        ("layers.1", True),
        ("candidate_selector", True),
        ("markov_head", False),
        ("confidence_head", True),
        ("<root>", True),
    ]


def test_additive_accumulation_scaling_matches_concatenated_global_mean():
    distributed = _distributed_module()
    accumulated = torch.tensor(0.7, requires_grad=True)
    reference = accumulated.detach().clone().requires_grad_(True)
    total_denominator = torch.zeros(())
    batches = (
        (torch.tensor([1.0]), torch.tensor([0.0])),
        (torch.tensor([2.0, 3.0, 4.0]), torch.tensor([1.0, 1.0, 2.0])),
    )
    for inputs, targets in batches:
        per_item = (accumulated * inputs - targets).square()
        numerator_loss, global_denominator = (
            distributed.scale_additive_loss_for_accumulation(
                per_item.sum(), torch.tensor(per_item.numel(), dtype=torch.float32)
            )
        )
        numerator_loss.backward()
        total_denominator += global_denominator
    accumulated.grad.div_(total_denominator)

    all_inputs = torch.cat([inputs for inputs, _ in batches])
    all_targets = torch.cat([targets for _, targets in batches])
    (reference * all_inputs - all_targets).square().mean().backward()
    torch.testing.assert_close(accumulated.grad, reference.grad)


def test_npu_rng_is_captured_and_restored_when_available(monkeypatch):
    checkpointing = _checkpointing_module()
    calls = []
    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        get_rng_state_all=lambda: [torch.tensor([9], dtype=torch.uint8)],
        set_rng_state_all=lambda state: calls.append(state),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    anchor = torch.Generator().manual_seed(7)
    state = checkpointing.capture_rng_state(anchor)
    assert "npu_rng" in state
    checkpointing.restore_rng_state(state, anchor)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], torch.tensor([9], dtype=torch.uint8))


def test_checkpoint_is_complete_atomic_semantic_and_boundary_guarded(tmp_path):
    checkpointing = _checkpointing_module()
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    anchor = torch.Generator().manual_seed(11)
    progress = checkpointing.TrainingProgress(3, 0, 1, 17)

    with pytest.raises(ValueError, match="optimizer-step boundary"):
        checkpointing.save_training_checkpoint(
            tmp_path / "mid-step",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
            anchor_generator=anchor,
            at_optimizer_boundary=False,
        )

    root = tmp_path / "complete"
    checkpointing.save_training_checkpoint(
        root,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
        anchor_generator=anchor,
        at_optimizer_boundary=True,
        semantic_config={"method": "dflash2", "block_size": 8},
    )
    assert (root / "COMPLETE").read_text() == "complete\n"
    with pytest.raises(ValueError, match="semantic configuration"):
        checkpointing.load_training_checkpoint(
            root,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            anchor_generator=anchor,
            expected_semantic_config={"method": "dflash2", "block_size": 16},
        )

    incomplete = tmp_path / "incomplete"
    with mock.patch.object(
        checkpointing, "_atomic_torch_save", side_effect=OSError("disk full")
    ), pytest.raises(OSError, match="disk full"):
        checkpointing.save_training_checkpoint(
            incomplete,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
            anchor_generator=anchor,
            at_optimizer_boundary=True,
        )
    assert not (incomplete / "COMPLETE").exists()


def test_two_rank_gloo_interrupted_resume_matches_uninterrupted(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    mp.spawn(
        _resume_parity_worker,
        args=(str(tmp_path / "rdzv"), str(tmp_path / "checkpoint"), str(reports)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        report = json.loads((reports / f"rank-{rank}.json").read_text())
        assert report == {
            "passed": True,
            "restored_progress": {
                "global_step": 2,
                "micro_step": 0,
                "epoch": 0,
                "sampler_cursor": 2,
            },
        }


def test_candidate_capability_is_immutable_and_runtime_needs_all_34_states(tmp_path):
    capability = _capability_module()
    path = tmp_path / "capability.json"
    record = capability.write_candidate_capability(
        path, artifact_identity="sha256:candidate"
    )
    assert record["runtime_attested"] is False
    assert record["deployable_export"] is False
    with pytest.raises(FileExistsError):
        capability.write_candidate_capability(path, artifact_identity="sha256:other")
    with pytest.raises(RuntimeError, match="rollback attestation"):
        capability.assert_runtime_usable(record, rollback_attestation=None)
    with pytest.raises(RuntimeError, match="34"):
        capability.assert_runtime_usable(
            record,
            rollback_attestation={
                "schema": "glm53-runtime-rollback-attestation-v1",
                "artifact_identity": "sha256:candidate",
                "strategy": "per_step_state_snapshots",
                "state_count": 33,
                "all_state_parity": True,
                "recurrent_state_parity": True,
                "short_convolution_state_parity": True,
            },
        )
    with pytest.raises(RuntimeError, match="artifact identity"):
        capability.assert_runtime_usable(
            record,
            rollback_attestation={
                "schema": "glm53-runtime-rollback-attestation-v1",
                "artifact_identity": "sha256:different-candidate",
                "strategy": "recompute_from_committed_checkpoint",
                "state_count": 34,
                "all_state_parity": True,
                "recurrent_state_parity": True,
                "short_convolution_state_parity": True,
            },
        )
    capability.assert_runtime_usable(
        record,
        rollback_attestation={
            "schema": "glm53-runtime-rollback-attestation-v1",
            "artifact_identity": "sha256:candidate",
            "strategy": "recompute_from_committed_checkpoint",
            "state_count": 34,
            "all_state_parity": True,
            "recurrent_state_parity": True,
            "short_convolution_state_parity": True,
        },
    )


def _valid_gate_evidence(tmp_path: Path) -> dict[str, Path]:
    values = {
        "tap-mapping": {
            "passed": True,
            "logical_layer_ids": [1, 11, 22, 32, 42],
            "concrete_hidden_state_indices": [2, 12, 23, 33, 43],
        },
        "final-logit": {
            "passed": True,
            "max_abs_error": 0.0,
            "tolerance": 1e-5,
        },
        "fsdp-resume": {
            "passed": True,
            "backend": "hccl",
            "fsdp2": True,
            "model": True,
            "optimizer": True,
            "scheduler": True,
            "rng": True,
            "sampler_cursor": True,
            "global_step": True,
            "dflash_bool_sdpa": True,
            "training_window_tokens": 4096,
        },
        "hbm": {
            "passed": True,
            "representative": True,
            "peak_bytes": 80,
            "capacity_bytes": 100,
        },
    }
    paths = {}
    for name, value in values.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    return paths


def _run_gate(paths: dict[str, Path], output: Path):
    command = [
        sys.executable,
        str(ROOT / "tools" / "run_ascend_training_gate.py"),
    ]
    for name in ("tap-mapping", "final-logit", "fsdp-resume", "hbm"):
        command.extend((f"--{name}-evidence", str(paths[name])))
    command.extend(("--output", str(output)))
    return subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )


def test_ascend_gate_records_all_evidence_and_fails_closed(tmp_path):
    paths = _valid_gate_evidence(tmp_path)
    output = tmp_path / "gate.json"
    result = _run_gate(paths, output)
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text())
    assert record["schema"] == "glm53-ascend-training-gate-v1"
    assert record["production_eligible"] is True
    assert record["runtime_attested"] is False
    assert set(record["evidence"]) == {
        "tap_mapping",
        "final_logit",
        "fsdp_resume",
        "hbm",
    }

    paths["hbm"].write_text(
        json.dumps(
            {
                "passed": False,
                "representative": True,
                "peak_bytes": 101,
                "capacity_bytes": 100,
            }
        )
    )
    failed_output = tmp_path / "failed-gate.json"
    result = _run_gate(paths, failed_output)
    assert result.returncode != 0
    failed = json.loads(failed_output.read_text())
    assert failed["production_eligible"] is False
    assert failed["runtime_attested"] is False
    assert failed["evidence"]["hbm"]["passed"] is False

    paths = _valid_gate_evidence(tmp_path)
    invalid_sdpa = json.loads(paths["fsdp-resume"].read_text())
    invalid_sdpa["dflash_bool_sdpa"] = False
    paths["fsdp-resume"].write_text(json.dumps(invalid_sdpa))
    sdpa_output = tmp_path / "failed-sdpa-gate.json"
    result = _run_gate(paths, sdpa_output)
    assert result.returncode != 0
    failed = json.loads(sdpa_output.read_text())
    assert failed["production_eligible"] is False
    assert failed["evidence"]["fsdp_resume"]["passed"] is False


def test_owned_production_files_have_no_forbidden_accelerator_paths():
    forbidden = ("cu" + "da", "nc" + "cl", "flash" + "attention")
    paths = [
        ROOT / "src/glm53_drafters/distributed.py",
        ROOT / "src/glm53_drafters/checkpointing.py",
        ROOT / "src/glm53_drafters/capability.py",
        ROOT / "tools/run_ascend_training_gate.py",
        ROOT / "scripts/run_ascend_training_gate.sh",
    ]
    for path in paths:
        lowered = path.read_text().lower()
        for token in forbidden:
            assert token not in lowered, f"{token} found in {path}"
