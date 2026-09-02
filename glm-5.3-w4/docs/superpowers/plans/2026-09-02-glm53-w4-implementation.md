# GLM-5.3 W4A8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this plan task-by-task.

**Goal:** Build an isolated W4A8 Stage B/Stage C pipeline for GLM-5.3 with strict provenance and the existing DSpark/DFlash2 training objectives.

**Architecture:** Copy the tested W8A8 pipeline into `glm-5.3-w4`, rename the package to `glm53_w4`, and parameterize only the target quantization contract, schema, provenance, and user-facing commands. Keep the drafter architecture, physical 2048 window, hidden alignment, and losses unchanged.

**Tech Stack:** Python 3.11+, PyTorch, safetensors, pytest, vLLM-Ascend/ModelSlim on 910B.

---

### Task 1: Add failing W4A8 contract tests

**Files:**
- Create: `tests/test_contracts_w4.py`
- Create: `tests/test_target_io_w4.py`
- Create: `tests/test_hidden_cache_w4.py`
- Create: `tests/test_vllm_hidden_w4.py`

- [x] Write tests that import `glm53_w4`, accept W4A8 metadata, reject W8A8/W4A8C8, emit W4A8 manifests, reject W8A8 cache provenance, and preserve the `quantization=ascend` vLLM engine argument.
- [x] Run the new tests and verify they fail because the W4 package did not yet exist.

### Task 2: Copy the tested pipeline into an isolated package

**Files:**
- Create: all runtime files under `src/glm53_w4/`
- Create: `tools/`, `scripts/`, `tests/`, `README.md`, `requirements.txt`, `.gitignore`

- [x] Copy the W8A8 implementation and rename imports/module paths to `glm53_w4`.
- [x] Keep only Stage B extraction and Stage C DSpark/DFlash2 entry points; do not add an alternate target-loading path.

### Task 3: Implement strict W4A8 provenance

**Files:**
- Modify: `src/glm53_w4/contracts.py`
- Modify: `src/glm53_w4/target_io.py`
- Modify: `src/glm53_w4/hidden_cache.py`
- Modify: `src/glm53_w4/vllm_hidden.py`
- Modify: `tools/extract_hidden_vllm_ascend.py`
- Modify: `tools/train_offline.py`

- [x] Require W4A8 in recursively scanned ModelSlim metadata.
- [x] Use W4A8-specific schema names, manifest values, and error messages.
- [x] Preserve dense BF16 target I/O checks and exact six-stream hidden alignment.
- [x] Reject cross-quantization trajectory/cache/I/O reuse.

### Task 4: Update launchers and documentation

**Files:**
- Modify: `scripts/extract_target_io.sh`
- Modify: `scripts/run_stage_b_hidden.sh`
- Modify: `scripts/train_drafter.sh`
- Modify: `scripts/smoke_no_npu.sh`
- Modify: `README.md`

- [x] Make all defaults and help text explicitly W4A8.
- [x] Document Stage B and Stage C commands, 910B smoke gate, and the BF16 I/O requirement.
- [x] Ensure DSpark B8 and DFlash2 B8/B16 commands are available.

### Task 5: Run verification and review

- [x] Run the W4A8 tests and the complete W4 suite.
- [x] Run `PYTHONPATH=src python -m compileall -q src tools tests`.
- [x] Run `bash -n scripts/*.sh`.
- [x] Run `PY=python bash scripts/smoke_no_npu.sh`.
- [x] Inspect the diff to confirm `glm-5.3-w8` and unrelated Omni files are unchanged.
