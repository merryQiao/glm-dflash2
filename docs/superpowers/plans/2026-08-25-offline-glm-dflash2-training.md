# Offline GLM-5.2 DFlash2 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete offline, Ascend-ready DFlash2 trainer that consumes the existing packed GLM-5.2 hidden cache without loading the target backbone.

**Architecture:** Extract and freeze only target token I/O weights, construct exact DFlash blocks from cached token-position masks, train a five-layer Qwen3-style DFlash2 draft with dense SDPA on NPU, and export resumable plus serving-compatible checkpoints. Production code is introduced test-first in focused modules.

**Tech Stack:** Python 3.12, PyTorch 2.x, torch-npu/HCCL, Transformers Qwen3 components, safetensors, NumPy mmap, unittest.

---

### Task 1: Target I/O extraction and loading

**Files:**
- Create: `src/glm_dflash2/target_io.py`
- Create: `tools/extract_target_io.py`
- Test: `tests/test_target_io.py`

- [ ] Write failing tests for sharded-index key resolution, dense dtype/shape validation, tied/untied heads, manifest fingerprints, frozen load, and cache/I-O provenance mismatch.
- [ ] Run the focused tests and confirm failure is due to missing implementation.
- [ ] Implement selective safetensors extraction without model construction and a frozen embedding/head loader.
- [ ] Run focused tests and keep all existing tests green.

### Task 2: Exact block preparation and objective

**Files:**
- Create: `src/glm_dflash2/dflash2_blocks.py`
- Create: `src/glm_dflash2/dflash2_objective.py`
- Test: `tests/test_dflash2_blocks.py`
- Test: `tests/test_dflash2_objective.py`

- [ ] Write failing tests for anchor eligibility, uniform-without-replacement sampling capped at 64, sorted selection and per-sample padding, seed/rank/epoch RNG behavior, clean-anchor/mask construction, unshifted labels, absolute RoPE, `j<a and (a+k)-j<2048` context visibility, invalid/no-anchor batches, and per-position/per-layer sentinels that detect +/-1 or layer-order errors.
- [ ] Implement the minimal block builder and verify the exact tensor shapes.
- [ ] Write failing tests for the exact base/selector numerators and denominators, ground-truth predecessor teacher forcing, empty candidate hits, candidate recall, base/selector acceptance length, and distributed additive metric reduction.
- [ ] Implement numerically stable FP32 CE reductions and metrics; run focused tests.

### Task 3: Exact chunked LM projection

**Files:**
- Create: `src/glm_dflash2/chunked_lm_head.py`
- Test: `tests/test_chunked_lm_head.py`

- [ ] Write dense-reference tests for vocabulary/token chunking, online log-sum-exp CE, exact running top-k, ties, masks, and gradients into hidden states.
- [ ] Implement exact token- and vocab-chunked projection without a full `[tokens,vocab]` allocation.
- [ ] Run focused tests across multiple chunk sizes and prove values/gradients match dense projection.

### Task 4: Portable DFlash2 model

**Files:**
- Create: `src/glm_dflash2/dflash2_model.py`
- Test: `tests/test_dflash2_model.py`

- [ ] Write failing tiny-model tests for the complete HF/SGLang config contract (`model_type`, architecture, five sliding layers, non-causal flag and every `dflash_config` key), target projection, `[context|noise]` attention shapes, absolute RoPE positions, block-local two-tap convolution, one prepare/finish projection, identity initialization, selector no-op initialization, and output shapes.
- [ ] Implement the Qwen3-style draft, identity-initialized DFlash2 convolution, and published pairwise selector.
- [ ] Verify finite forward/backward, then compare every exported key/shape (including no `.weight` on codebooks) against the SGLang loader contract.

### Task 5: Offline trainer forward contract

**Files:**
- Create: `src/glm_dflash2/offline_trainer.py`
- Test: `tests/test_offline_trainer.py`

- [ ] Write a failing end-to-end tiny-cache test covering cache row -> provenance gate -> collator -> blocks -> model -> chunked frozen LM head -> total loss.
- [ ] Implement the trainer module and its metric dataclass with separate base/selector values.
- [ ] Verify one optimizer step decreases deterministic tiny-batch loss and leaves frozen I/O byte-identical.

### Task 6: Ascend training loop and exact resume

**Files:**
- Create: `src/glm_dflash2/distributed.py`
- Create: `src/glm_dflash2/checkpointing.py`
- Create: `tools/train_dflash2_offline.py`
- Test: `tests/test_checkpointing.py`
- Test: `tests/test_train_cli.py`

- [ ] Write failing tests for device/backend selection, FSDP2 wrap/ignored-I-O plan, gradient-sync toggling, scheduler continuity, per-rank/data/anchor RNG restoration, optimizer-step-only checkpointing, rank-zero logging, and consolidated export metadata.
- [ ] Implement CPU/CUDA/NPU device abstraction, HCCL initialization, bottom-up FSDP2 wrapping, BF16 autocast, FSDP2-native accumulation sync toggling, clipping, cosine schedule, logging, save and resume.
- [ ] Run a mandatory two-process CPU resume test spanning accumulation and epoch boundaries; verify uninterrupted/resumed weights, LR, data cursor and sampled anchors match.

### Task 7: Launchers, smoke, hardware gates, and documentation

**Files:**
- Create: `requirements-train.txt`
- Create: `scripts/extract_glm52_io.sh`
- Create: `scripts/train_glm52_dflash2_910b.sh`
- Create: `scripts/smoke_train_no_npu.sh`
- Create: `scripts/gate_train_2rank_910b.sh`
- Create: `tools/compare_sglang_runtime.py`
- Modify: `README.md`
- Modify: `docs/design.md`
- Modify: `docs/implementation-plan.md`
- Test: `tests/test_training_launchers.py`

- [ ] Write failing launcher tests for required mask ID/cache/I-O paths, every fixed GLM draft dimension, HCCL/FSDP2, B16, exactly 64 anchors, gamma 7, chunk sizes, selector settings, and no target model argument in training.
- [ ] Implement launchers and a no-NPU tiny-cache smoke.
- [ ] Implement executable two-rank HCCL/FSDP2 and SGLang parity gates that write JSON artifacts; compare logits, top-k, pair scores, final path and frozen-I-O sharing.
- [ ] Document extraction, single-node and multi-node 910B commands, storage/memory caveats, pinned vendor stack, resume/export, and the rule that serving compatibility is not claimed before both gates pass.

### Task 8: Full verification

- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `bash scripts/smoke_no_model.sh` to protect the existing data pipeline.
- [ ] Run `bash scripts/smoke_train_no_npu.sh`.
- [ ] Run `python -m compileall -q src tools` and `bash -n scripts/*.sh`.
- [ ] Inspect exported config/safetensors and prove that no full target model is instantiated by the training path.
- [ ] Record that the real 910B HCCL/FSDP2 and SGLang serving parity gates remain hardware-only until run on the deployment server.
