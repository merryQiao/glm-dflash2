# GLM-5.3 Drafter Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and execute each task with red-green-refactor.

**Goal:** Repair confirmed official-config, mask, DFlash2, initialization, data-coverage, and observability defects while preserving the approved DSpark TV loss.

**Architecture:** Keep the existing Stage A → hidden cache → offline trainer boundaries. Add small pure helpers at those boundaries so production contracts can be unit-tested without loading GLM-5.3 or requiring an NPU.

**Tech Stack:** Python, PyTorch, pytest/unittest, JSON/JSONL, FSDP2/HCCL production path.

---

### Task 1: Official target config and mask contract

**Files:** `src/glm53_drafters/target_io.py`, `tools/train_drafter_offline.py`, `tests/test_glm53_target_io.py`, `tests/test_glm53_training_entrypoints.py`

- [x] Add failing official root/text schema, missing/conflicting field, wrong-mask, and legacy-artifact tests.
- [x] Run focused tests and confirm expected failures.
- [x] Implement exact root `glm5_next`, nested `glm5_next_text`, nested `dtype` resolution and tokenizer `[MASK]` validation.
- [x] Bump relevant artifact schemas and bind literal mask token, tokenizer identity, and canonical target-I/O identity into semantic config.
- [x] Run focused tests green.

### Task 2: DFlash2 official top-k-hit selector supervision

**Files:** `src/glm53_drafters/dflash2_model.py`, `src/glm53_drafters/objectives.py`, `src/glm53_drafters/offline_trainer.py`, `tests/test_glm53_training_models.py`, `tests/test_glm53_training_objectives.py`

- [x] Add a DSpark characterization test locking `0.1 CE + 0.9 TV + 1.0 confidence` before touching the trainer.
- [x] Add hit/miss candidate tests, unary-recall assertions, keep/depth weighting, hit-only denominator, zero-miss gradient, and inference-parity tests.
- [x] Confirm the legacy target-injection behavior fails the official parity tests.
- [x] Keep the real base top-k unchanged and mask selector loss/denominator on misses.
- [x] Optimize independently normalized `base.mean + selector.mean`.
- [x] Run focused tests green.

### Task 3: Official-style initialization

**Files:** `src/glm53_drafters/modeling_common.py`, model files, `tools/train_drafter_offline.py`, `tests/test_glm53_training_models.py`

- [x] Add failing initialization-distribution and preserved-zero/identity tests.
- [x] Add centralized FP32 initialization with `initializer_range=0.02`.
- [x] Apply before BF16/FSDP; preserve explicit stabilizing parameters.
- [x] Run focused tests green.

### Task 4: Route statistics and long-row coverage

**Files:** `tools/generate_trajectories.py`, `src/glm53_drafters/blocks.py`, `tools/train_drafter_offline.py`, relevant tests.

- [x] Add failing committed-JSONL route-summary tests covering resume, partial/error, and no-resume behavior.
- [x] Add failing deterministic full-cycle window/anchor coverage and resume-stability tests.
- [x] Persist canonical per-row routes, recompute summaries from committed rows, and implement cycling windows.
- [x] Record policies in semantic manifests.
- [x] Run focused tests green.

### Task 5: Training and held-out metrics

**Files:** `src/glm53_drafters/offline_trainer.py`, `tools/train_drafter_offline.py`, training tests.

- [x] Add failing JSONL idempotency and held-out tests for tiny split counts, disjointness, route distribution, fixed windows, and resume.
- [x] Bind split identity/policy into semantics before step calculation; add all-rank unbiased reduced metrics and deterministic validation under `no_grad`, restoring train/eval mode.
- [x] Include DFlash2 unary recall and existing DSpark confidence metrics.
- [x] Run focused tests green.

### Task 6: Documentation and verification

**Files:** `README.md`, `docs/ASCEND_910B_A2_RUNBOOK.md`

- [x] Document exact mask ID, mixed Stage A routes, window policy, and experimental DFlash2 objective.
- [x] Run `PYTHONPATH=src:. pytest -q`.
- [x] Run tiny smoke for DFlash/DFlash2/DSpark B8/B16-supported combinations.
- [x] Confirm the DSpark coefficient/denominator characterization test remains green.
