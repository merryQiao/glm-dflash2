# GLM-5.2 BF16 Sampling Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct and streamline the BF16 sampling trajectory, hidden-cache, and offline DFlash2 training path.

**Architecture:** Preserve the two-pass pipeline and reference routing semantics. Tighten each boundary independently: immutable sampled trajectories, exact hidden capture, read-only packed cache, globally normalized distributed objective, durable checkpoints, and an explicit serving ABI.

**Tech Stack:** Python, PyTorch/FSDP2, Transformers 4.57.3, SGLang, safetensors, unittest.

---

### Task 1: Sampling and trajectory integrity

**Files:** `tools/generate_trajectories.py`, `scripts/run_stage_a_trajectories.sh`, `tests/test_sglang_stage_a.py`, `README.md`

- [ ] Add failing tests for official sampling defaults and duplicate output IDs.
- [ ] Implement temperature 1.0/top-p 0.95 defaults, strict duplicate detection, and exception-safe locking.
- [ ] Run the focused Stage-A tests.

### Task 2: Hidden capture and read-only cache

**Files:** `src/glm_dflash2/sglang_hidden_runner.py`, `src/glm_dflash2/hidden_cache.py`, `tools/validate_hidden_cache.py`, associated tests

- [ ] Add failing tests for the GLM Eagle3 hook fallback and immutable dataset open.
- [ ] Add explicit checksum policy so training does not hash every hidden sample.
- [ ] Run focused hidden-cache and runner tests.

### Task 3: Anchor and distributed objective correctness

**Files:** `src/glm_dflash2/dflash2_blocks.py`, `src/glm_dflash2/distributed.py`, `src/glm_dflash2/offline_trainer.py`, associated tests

- [ ] Add failing tests for zero-supervision suffix anchors and unequal rank denominators.
- [ ] Filter invalid anchors and implement global per-microbatch weighted means.
- [ ] Run focused single- and two-rank tests.

### Task 4: Durable checkpoints and serving ABI

**Files:** `src/glm_dflash2/checkpointing.py`, `src/glm_dflash2/dflash2_model.py`, associated tests and docs

- [ ] Add failing tests for the official architecture name and durable marker helper.
- [ ] Implement fsync-backed atomic metadata/marker writes and ABI correction.
- [ ] Run checkpoint/model tests.

### Task 5: Remove obsolete code and verify

**Files:** historical vLLM response-only modules/scripts/tests and unused helpers

- [x] Remove only code unreachable from the SGLang two-pass production path.
- [ ] Update README and dependency-facing examples for BF16 sampling.
- [ ] Run all 129+ tests and `scripts/smoke_train_no_npu.sh`.
