# GLM-5.3 Stage B Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task.

**Goal:** Close the independent Stage B Tasks 1–4 review findings without importing or modifying the adjacent GLM-5.2 implementation.

**Architecture:** Central strict validators bind Stage A records and the local GLM5Next checkpoint before model allocation. The SGLang adapter validates resolved runtime objects, independently hooks concrete decoder blocks and final norm, and emits per-row numeric evidence comparing packed auxiliary slices and globally comparable native logits. The cache writer freezes production only when every committed row has valid evidence; otherwise it seals as incomplete, while bounded smoke remains explicitly non-production.

**Tech Stack:** Python 3, PyTorch CPU tests, safetensors, unittest/pytest, lazy SGLang integration, Ascend/NPU production contract.

---

### Task 1: Strict Stage A identity and token contract

**Files:**
- Modify: `tests/test_glm53_hidden_extraction.py`
- Modify: `tests/test_glm53_hidden_cache.py`
- Modify: `src/glm53_drafters/contracts.py`
- Modify: `src/glm53_drafters/hidden_extraction.py`
- Modify: `src/glm53_drafters/hidden_cache.py`
- Modify: `tools/extract_hidden_sglang.py`

- [ ] Add tests proving frozen production input rejects missing/false identity or eligibility fields and rejects every mismatch with the local target identity.
- [ ] Add adversarial tests for bool, float, string, negative, and `vocab_size` token IDs, plus float/string/non-binary masks.
- [ ] Run the focused tests and confirm failures are caused by permissive defaults/coercion.
- [ ] Add reusable non-coercive sequence validation, required identity-field validation, and strict equality binding.
- [ ] Preserve explicit `smoke_unverified`, `production_eligible=false`, and the total 50-record bound while requiring the same exact source identity.
- [ ] Record validated source identity in cache provenance rather than replacing it with an unverified target claim.
- [ ] Re-run focused tests to GREEN.

### Task 2: Exact GLM5Next production checkpoint contract

**Files:**
- Modify: `tests/test_glm53_target_io.py`
- Modify: `src/glm53_drafters/target_io.py`

- [ ] Add tiny-checkpoint tests that exercise an explicitly test-only contract object while the production entrypoint remains hard-wired to `model_type=glm5_next`, `num_hidden_layers=45`, `hidden_size=4096`, `vocab_size=154880`, RMS epsilon exactly `1e-5`, BF16 config, untied embeddings, and no quantization metadata.
- [ ] Add off-by-one layer/vocab/hidden tests and reject wrong architecture, epsilon, dtype, tied/bias, and all quantization declarations.
- [ ] Run tests and confirm RED.
- [ ] Implement a nested-text-config-aware exact contract validator used by both identity and extraction before reading tensor payloads. Production callers cannot pass dimension/architecture overrides; only the validator's direct unit-test API accepts a separate immutable tiny contract.
- [ ] Include validated architecture fields in the target-I/O manifest and cache compatibility checks.
- [ ] Re-run tests to GREEN.

### Task 3: Resolved SGLang runtime validation

**Files:**
- Modify: `tests/test_glm53_sglang_hidden_runner.py`
- Modify: `src/glm53_drafters/sglang_hidden_runner.py`
- Modify: `tools/extract_hidden_sglang.py`

- [ ] Add tests where requested `model_runner=torch` resolves to anything other than an actual SGLang PyTorch/torch runner object, a resolved NPU device (`npu` or `npu:<rank>`), or the resolved Ascend attention backend (`ascend`) and must fail after `load_model`.
- [ ] Add tests that provenance contains post-load `runner_class`, `model_class`, resolved runner implementation=`pytorch`, resolved device type=`npu`, resolved attention backend=`ascend`, resolved dtype=`bfloat16`, DP=1, PP=1, one-request/no-radix/no-chunk/no-graph settings, TP/EP topology, SGLang/torch-npu/CANN versions, and requested-versus-resolved contract separation.
- [ ] Run tests and confirm RED.
- [ ] Validate the resolved ServerArgs and loaded runner/model/device/backend objects against the literal production values above; reject ambiguity rather than trusting raw `model_runner` or a wrapper merely named `torch_runner`.
- [ ] Record requested and resolved contracts separately in immutable backend metadata.
- [ ] Re-run tests to GREEN.

### Task 4: Per-row numeric tap and final-hidden attestation

**Files:**
- Modify: `tests/test_glm53_hidden_cache.py`
- Modify: `tests/test_glm53_hidden_extraction.py`
- Modify: `tests/test_glm53_sglang_hidden_runner.py`
- Modify: `src/glm53_drafters/hidden_capture.py`
- Modify: `src/glm53_drafters/hidden_cache.py`
- Modify: `src/glm53_drafters/hidden_extraction.py`
- Modify: `src/glm53_drafters/sglang_hidden_runner.py`
- Modify: `tools/extract_hidden_sglang.py`

- [ ] Define immutable attestation evidence containing schema/version, ordered logical/physical taps, independently hooked tensor parity metrics, final-logit parity metrics, native-logit API identity, tolerances, token count, and pass/failure reason.
- [ ] Add tests for missing evidence, one swapped/off-by-one tap, numeric mismatch, unavailable concrete layers, unavailable globally comparable TP logits, and a fully passing tiny runner.
- [ ] Add writer tests showing any missing/failed row prevents production freeze and seals `status=incomplete`, `production_eligible=false`; smoke remains `smoke_unverified` regardless of evidence.
- [ ] Run tests and confirm RED.
- [ ] Install independent forward hooks on exact concrete decoder blocks and final norm; compare every returned packed auxiliary slice in order.
- [ ] Obtain native logits only through the loaded runner/model's supported globally comparable logits path. Never compare a raw TP-sharded LM-head output; if global logits or concrete blocks cannot be proven, return actionable failed evidence.
- [ ] Compare the captured last-token final hidden through the same supported global logits path against native last-token logits under recorded tolerances.
- [ ] Attach evidence to every capture and row; aggregate passed row count and attestation digest into the cache seal.
- [ ] Make `freeze()` fail closed to an incomplete seal when production evidence is absent or failed.
- [ ] Re-run focused tests to GREEN.

### Task 5: Regression verification

**Files:** No production edits.

- [ ] Run all Tasks 1–4 focused tests.
- [ ] Run the pre-existing Stage A regression suite.
- [ ] Compile modified Python files and parse modified shell launchers.
- [ ] Verify no `glm_dflash2` import and no CUDA/NCCL/FlashAttention2 API was added.
- [ ] Compare repository status/path inventory captured before and after this task and prove every edited path is below `glm-5.3-flash`; do not modify any adjacent parent implementation file.
- [ ] Report exact RED/GREEN commands and note that real production eligibility still requires a successful run on the locked Ascend SGLang runtime.
