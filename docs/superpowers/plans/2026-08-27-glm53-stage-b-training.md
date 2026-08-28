# GLM-5.3 Stage B and Offline Drafter Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Ascend 910B A2 Stage B hidden-cache path and aligned offline DFlash, DFlash2, and DSpark training paths for GLM-5.3-Flash.

**Architecture:** Keep the proven Stage A package unchanged and add a separate `glm53_drafters` package. Stage B teacher-forces frozen Stage A tokens into an Ascend SGLang internal runner and writes one common schema-v2 cache; three offline methods consume that cache and frozen target token I/O through a shared five-layer dense draft backbone. Runtime export is capability-gated because GLM-5.3 linear-attention rollback is not solved by training.

**Tech Stack:** Python 3.11/3.12, PyTorch/torch_npu, HCCL/FSDP2, SGLang Ascend, Transformers Qwen3 config primitives, pytest/unittest, packed binary cache.

---

### Task 1: Fixed GLM-5.3 contracts

**Files:**
- Create: `src/glm53_drafters/__init__.py`
- Create: `src/glm53_drafters/contracts.py`
- Test: `tests/test_glm53_drafter_contracts.py`

- [ ] Write failing tests for target depth/width/vocabulary, official layer
  selection `[1,11,22,32,42]`, logical-to-hidden index mapping, method/block
  matrix, and cache shape/storage estimates.
- [ ] Run `PYTHONPATH=src python -m pytest -q tests/test_glm53_drafter_contracts.py`
  and verify failure is caused by the missing package.
- [ ] Implement immutable target/draft/cache contracts and strict validators.
- [ ] Re-run the focused test and full existing Stage A suite.

### Task 2: Capture and packed hidden cache

**Files:**
- Create: `src/glm53_drafters/hidden_capture.py`
- Create: `src/glm53_drafters/hidden_cache.py`
- Create: `src/glm53_drafters/hidden_extraction.py`
- Create: `tests/test_glm53_hidden_cache.py`
- Create: `tests/test_glm53_hidden_extraction.py`

- [ ] Write failing tests for `[T,5,4096]` plus final `[T,4096]` semantics,
  ordered capture metadata, append/read/resume, stream checksums, partial
  artifact rejection, and frozen-manifest identity binding.
- [ ] Add failing tests that `smoke_unverified` input requires an explicit
  bounded flag, propagates `production_eligible=false`, never freezes a
  production cache, and that `smoke_failed` is always rejected.
- [ ] Verify focused RED failures.
- [ ] Implement atomic packed writer, mmap dataset/collator, cache validator,
  source trajectory reader, storage preflight, and resumable extraction loop.
- [ ] Verify focused GREEN tests and Stage A regression tests.

### Task 3: Ascend SGLang Stage B adapter

**Files:**
- Create: `src/glm53_drafters/sglang_hidden_runner.py`
- Create: `tools/extract_hidden_sglang.py`
- Create: `scripts/run_stage_b_hidden.sh`
- Create: `tests/test_glm53_sglang_hidden_runner.py`
- Modify: `tests/test_launchers.py`

- [ ] Write failing tests for NPU/Ascend-only production arguments, DP/PP/cache
  restrictions, lazy imports, GLM-5.3 wrapper discovery, capture hook
  negotiation, physical tap verification, final-norm discovery, BF16 CPU
  transfer, and launcher construction.
- [ ] Verify focused RED failures.
- [ ] Implement the strict runner and CLI without importing SGLang/torch_npu at
  module import time.
- [ ] Verify focused GREEN tests, `bash -n`, and import-without-SGLang smoke.

### Task 4: Frozen target embedding and LM head

**Files:**
- Create: `src/glm53_drafters/target_io.py`
- Create: `tools/extract_target_io.py`
- Create: `scripts/extract_target_io.sh`
- Create: `tests/test_glm53_target_io.py`

- [ ] Write failing tests for nested `text_config`, key discovery, untied
  `[154880,4096]` tensors, BF16 preservation, bias rejection, fingerprints,
  checksums, and hidden-cache parity.
- [ ] Verify RED, implement minimal extraction/load/parity code, then verify
  GREEN and regression tests.

### Task 5: Shared backbone and DFlash/DFlash2 models

**Files:**
- Create: `src/glm53_drafters/modeling_common.py`
- Create: `src/glm53_drafters/dflash_model.py`
- Create: `src/glm53_drafters/dflash2_model.py`
- Create: `tests/test_glm53_dflash_models.py`

- [ ] Write failing tests for the fixed five-layer config, target projection,
  full non-causal block attention shape, no sliding/recurrent state, DFlash2
  identity initialization, block-local convolution, selector shapes, and finite
  gradients on tiny configs.
- [ ] Verify RED failures.
- [ ] Implement Qwen3-shaped primitives with device-generic SDPA and no CUDA
  APIs, then DFlash and DFlash2 wrappers.
- [ ] Verify GREEN and full regression tests.

### Task 6: Block construction and chunked objectives

**Files:**
- Create: `src/glm53_drafters/blocks.py`
- Create: `src/glm53_drafters/chunked_lm_head.py`
- Create: `src/glm53_drafters/objectives.py`
- Create: `tests/test_glm53_objectives.py`

- [ ] Write failing tests for deterministic valid anchors, one-anchor physical
  blocks, absolute positions, assistant-mask boundaries, depth weights,
  chunked CE equality to dense CE, selector hit masking, exact full-vocabulary
  TV, and confidence target `1-TV`.
- [ ] Verify RED, implement the minimal primitives, and verify GREEN.

### Task 7: DSpark model and all offline trainers

**Files:**
- Create: `src/glm53_drafters/dspark_model.py`
- Create: `src/glm53_drafters/offline_trainer.py`
- Create: `tests/test_glm53_offline_trainers.py`

- [ ] Write failing tests for rank-256 Markov/confidence interfaces, predecessor
  alignment, frozen target I/O, DFlash/DFlash2/DSpark additive metrics,
  acceptance statistics, zero-valid-token differentiability, and one finite CPU
  optimizer step per method with tiny dimensions.
- [ ] Verify RED, implement method trainers, and verify GREEN.

### Task 8: Ascend distribution and durable resume

**Files:**
- Create: `src/glm53_drafters/distributed.py`
- Create: `src/glm53_drafters/checkpointing.py`
- Create: `tests/test_glm53_distributed.py`
- Create: `tests/test_glm53_checkpointing.py`

- [ ] Write failing tests for CPU/gloo and NPU/HCCL resolution, lazy torch_npu,
  FSDP2 wrap policy, exact accumulation scaling, NPU RNG state, atomic complete
  markers, semantic-config mismatch rejection, and optimizer-boundary-only
  checkpoints.
- [ ] Add a two-rank CPU/gloo interrupted-versus-uninterrupted parity test for
  model, optimizer, scheduler, RNG, distributed sampler cursor, global step,
  and optimizer-boundary resume. Record the same FSDP2/HCCL parity as an A2
  hardware gate.
- [ ] Verify RED, implement without NCCL/CUDA branches, and verify GREEN.

### Task 9: Unified training CLI and Ascend launchers

**Files:**
- Create: `tools/train_drafter_offline.py`
- Create: `scripts/train_drafter.sh`
- Create: `scripts/smoke_stage_b_training.sh`
- Create: `tools/run_ascend_training_gate.py`
- Create: `scripts/run_ascend_training_gate.sh`
- Create: `src/glm53_drafters/capability.py`
- Create: `tests/test_glm53_training_cli.py`
- Modify: `tests/test_launchers.py`

- [ ] Write failing tests for the exact method/block matrix, fixed production
  dimensions, recipes, FSDP2 default, scheduler/resume semantics, semantic
  fingerprints, capability metadata, and NPU launcher arguments.
- [ ] Write failing tests that every training artifact is immutable
  `runtime_attested=false`, no deployable export is produced, the Ascend gate
  records tap/final-logit/FSDP-resume/HBM evidence, and runtime use hard-fails
  without a separate rollback attestation with no override.
- [ ] Verify RED, implement CLI/launchers, and verify GREEN.
- [ ] Run a CPU tiny-config smoke for all three methods and `bash -n` on every
  script.

### Task 10: Documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`
- Create: `docs/ASCEND_910B_A2_STAGE_B_TRAINING.md`

- [ ] Document the two-stage workflow, exact taps and off-by-one mapping,
  storage estimate, target-I/O step, all five training settings, resume, and
  the real Ascend gates.
- [ ] Document that linear/KDA rollback remains an evaluation/runtime blocker,
  not a Stage B/training blocker.
- [ ] Run `PYTHONPATH=src python -m pytest -q`, `python -m compileall -q src tools`,
  all launcher syntax checks, CLI help checks, no-parent-import search, and
  no-CUDA/NCCL/FlashAttention search.
- [ ] Request an independent code review, fix all Critical/Important findings,
  and re-run the complete verification suite.
