# Unified GLM Drafters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one schema-v2 GLM-5.2 hidden cache and three aligned offline training consumers—DFlash, DFlash2, and DSpark—using the exact common five-layer GLM DFlash backbone defined in the approved specification.

**Architecture:** Stage B returns a typed capture containing five auxiliary residual streams and the post-final-norm LM-head input from one target forward, then writes four independently checksummed streams. A shared block builder and backbone implement the common token, mask, absolute-position, and attention contract; method-specific trainers add only DFlash2 convolution/selector or DSpark Markov/confidence modules and objectives. Legacy schema-v1 reading remains opt-in and is rejected by aligned training CLIs.

**Tech Stack:** Python 3.12, PyTorch/torch-npu, Transformers 4.57.3, SGLang standalone runner, safetensors, unittest, shell launchers.

All local test commands use:

```bash
PY=/inspire/ssd/project/sais-bio/public/chenbaoyou/miniconda3/envs/nemo/bin/python
```

---

## File map

- `src/glm_dflash2/glm_draft_config.py`: immutable production constants and validation for the shared GLM-5.2 draft experiment.
- `src/glm_dflash2/dflash_blocks.py`: deterministic sample-ID anchor selection and common block/mask/absolute-position construction.
- `src/glm_dflash2/draft_backbone.py`: full-attention five-layer GLM DFlash backbone plus DFlash2 dynamic-convolution variant.
- `src/glm_dflash2/dflash2_model.py`: compatibility exports and DFlash2 selector/model assembly.
- `src/glm_dflash2/dspark_model.py`: DSpark Markov and confidence heads on the shared backbone.
- `src/glm_dflash2/method_objectives.py`: globally reducible DFlash, DFlash2, and DSpark numerators/denominators.
- `src/glm_dflash2/hidden_capture.py`: typed one-forward auxiliary/final-hidden contract and backend mapping metadata.
- `src/glm_dflash2/hidden_cache.py`: schema-v2 storage/reader/collator plus explicit schema-v1 legacy adapter.
- `src/glm_dflash2/sglang_hidden_runner.py`: one-forward SGLang capture adapter.
- `src/glm_dflash2/hidden_extraction.py`: production cache extraction with strict v2 validation.
- `src/glm_dflash2/target_io.py`: frozen embedding/LM-head artifact validation used by every method.
- `src/glm_dflash2/offline_trainer.py`: shared batch preparation and three method-specific offline trainer modules.
- `tools/extract_hidden_sglang.py`: schema-v2 Stage-B CLI.
- `tools/validate_hidden_cache.py`: schema, numerical parity, and negative-control gate.
- `tools/train_drafter_offline.py`: unified `--method dflash|dflash2|dspark` training/export CLI.
- `scripts/run_stage_b_hidden.sh`, `scripts/train_glm52_drafter_910b.sh`: Ascend launchers.
- `tests/test_*`: unit, integration, provenance, launcher, and regression coverage.

### Task 1: Lock the shared GLM architecture

**Files:**
- Create: `src/glm_dflash2/glm_draft_config.py`
- Create: `src/glm_dflash2/draft_backbone.py`
- Modify: `src/glm_dflash2/dflash2_model.py`
- Test: `tests/test_glm_draft_config.py`
- Modify test: `tests/test_dflash2_model.py`

- [ ] **Step 1: Write failing architecture tests**

Assert the production config has logical layers `(1, 20, 38, 56, 75)`, depth 78, hidden/intermediate `6144/12288`, five draft layers, `64/64` heads, explicit `head_dim=64`, Q/K/V width 4096, O projection `4096 -> 6144`, full attention, no sliding window, RoPE theta `8e6`, and RMS epsilon `1e-5`. Add a tiny-config test proving projection widths use `heads * head_dim` rather than `hidden_size`.

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=src $PY -m unittest tests.test_glm_draft_config tests.test_dflash2_model -v`

Expected: failures on the old `32/8/head_dim128/sliding_window2048` production configuration.

- [ ] **Step 3: Implement the immutable config and shared backbone**

Expose `GLM52_DRAFT_SPEC`, `build_glm52_draft_config(method=...)`, `DFlashDecoderLayer`, `DFlash2DecoderLayer`, and `GLMDraftBackbone`. Validate:

```python
q_width = config.num_attention_heads * config.head_dim
kv_width = config.num_key_value_heads * config.head_dim
assert q_width == kv_width == 4096  # production
assert config.sliding_window is None
```

Keep DFlash2's grouped two-tap convolution identity initialization; plain DFlash uses no convolution kernels.

- [ ] **Step 4: Preserve compatibility imports**

Re-export existing public classes/builders from `dflash2_model.py` so old tests/checkpoints can be diagnosed, but make the GLM production builder return the corrected architecture.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the same unittest command and require zero failures.

- [ ] **Step 6: Commit**

`git add src/glm_dflash2/glm_draft_config.py src/glm_dflash2/draft_backbone.py src/glm_dflash2/dflash2_model.py tests/test_glm_draft_config.py tests/test_dflash2_model.py && git commit -m "feat: align GLM draft backbone with published shape"`

### Task 2: Enforce the common block and anchor contract

**Files:**
- Create: `src/glm_dflash2/dflash_blocks.py`
- Modify: `src/glm_dflash2/dflash2_blocks.py`
- Test: `tests/test_dflash_blocks.py`
- Modify test: `tests/test_dflash2_blocks.py`

- [ ] **Step 1: Write failing block-contract tests**

Cover: one anchor plus 15 masks; labels `a+1..a+15`; anchor eligibility requires `loss_mask[a] && loss_mask[a+1]`; validity is cumulative and never reopens after a hole; context keys are exactly absolute positions `<a`; local positions are `a..a+15`; full attention is independent of context length; and deterministic anchors depend only on `(seed, epoch, sample_id)`.

Also assert every one of the 16 local queries can attend non-causally to every
one of the 16 local slots, while a block that reaches the physical sequence end
retains only the in-range cumulative-valid successor positions.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src $PY -m unittest tests.test_dflash_blocks tests.test_dflash2_blocks -v`

Expected: failures from sliding-window masking, non-cumulative validity, and generator-state/rank-dependent sampling.

- [ ] **Step 3: Implement pure deterministic sampling**

Hash the UTF-8 tuple `(global_seed, epoch, sample_id)` to seed a local CPU generator, sample eligible anchors uniformly without replacement, sort only for stable packing, and never mutate a trainer RNG.

- [ ] **Step 4: Implement full-attention blocks and compatibility wrapper**

Build `DFlashBlocks` with explicit `context_position_ids`, `draft_position_ids`, `target_ids`, and cumulative `target_mask`. Retain `dflash2_blocks.py` as a thin deprecated import wrapper.

- [ ] **Step 5: Verify GREEN and commit**

Run the focused tests, then commit as `feat: unify GLM draft block contract`.

### Task 3: Add cache schema v2 with mandatory final hidden

**Files:**
- Create: `src/glm_dflash2/hidden_capture.py`
- Modify: `src/glm_dflash2/hidden_cache.py`
- Test: `tests/test_hidden_capture.py`
- Modify test: `tests/test_hidden_cache.py`

- [ ] **Step 1: Write failing schema-v2 tests**

Round-trip a sample with:

```python
aux = torch.randn(T, 5, H, dtype=torch.bfloat16)
final = torch.randn(T, H, dtype=torch.bfloat16)
```

Assert separate files/checksums, exact layer order, post-final-norm semantics, target depth 78, `[T,5,H]` auxiliary view, `[T,5*H]` compatibility view, `[T,H]` final view, strict corruption detection, and collator padding. Assert missing final stream fails for v2. Assert v1 opens only with `allow_legacy_v1=True` and reports `aligned_methods_allowed=False`.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src $PY -m unittest tests.test_hidden_capture tests.test_hidden_cache -v`.

- [ ] **Step 3: Implement the typed capture and v2 streams**

Add `TargetHiddenCapture(aux_hidden_states, target_final_hidden, capture_mapping)` validation. Change writer append to require both tensors for v2 and persist `aux_hidden_states.bin` and `target_final_hidden.bin`; derive flattened `hidden_states` in the reader only.

- [ ] **Step 4: Implement strict metadata and explicit legacy adapter**

Require `schema_version=2`, logical layer order, backend tap namespace/semantics, dtype/width, final semantics, target/tokenizer/generation provenance, and all stream checksums. Do not synthesize final hidden for v1.

- [ ] **Step 5: Verify GREEN and commit**

Run focused cache tests and commit as `feat: add unified hidden cache schema v2`.

### Task 4: Capture auxiliary and final hidden in one SGLang forward

**Files:**
- Modify: `src/glm_dflash2/sglang_hidden_runner.py`
- Modify: `src/glm_dflash2/hidden_extraction.py`
- Modify: `tools/extract_hidden_sglang.py`
- Modify: `scripts/run_stage_b_hidden.sh`
- Test: `tests/test_sglang_hidden_runner.py`
- Modify test: `tests/test_hidden_extraction.py`
- Modify test: `tests/test_ascend_launchers.py`

- [ ] **Step 1: Write failing one-forward tests**

Mock SGLang returning packed auxiliary taps plus a separately identified post-final-norm tensor. Assert one runner call per sample, ordered mapping metadata, rank-zero CPU BF16 return, empty nonzero-rank return, and failure when final semantics/tensor is absent or is merely decoder-layer 77 output.

- [ ] **Step 2: Verify RED**

Run the three focused test modules and require the new assertions to fail.

- [ ] **Step 3: Implement the capture bridge**

Configure five auxiliary GLM taps plus the backend's post-final-norm LM-head-input hook before model load. Normalize backend output into `TargetHiddenCapture`; never infer semantics from shape alone. Keep compatibility with supported SGLang runner signatures without issuing a second forward.

- [ ] **Step 4: Wire Stage B and fail closed**

Make extraction write schema v2 only by default, record the concrete capture mapping, and reject legacy/ambiguous SGLang builds with an actionable runtime error.

- [ ] **Step 5: Verify GREEN and commit**

Run focused tests, then commit as `feat: capture GLM auxiliary and final hidden together`.

### Task 4A: Harden the frozen target I/O artifact

**Files:**
- Modify: `src/glm_dflash2/target_io.py`
- Modify: `tools/extract_target_io.py`
- Modify test: `tests/test_target_io.py`

- [ ] **Step 1: Write failing target-I/O contract tests**

Assert embedding and LM-head contents are independently checksummed, their
shapes are exactly `[vocab_size, 6144]`, vocabulary size/dtype/model revision
match the cache provenance, and the artifact rejects bias, logit scaling, or
soft-cap unless an explicitly versioned reconstruction transform reproduces
them. Assert cache manifest and target-I/O fingerprints must match before any
trainer forward.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src $PY -m unittest tests.test_target_io -v`.

- [ ] **Step 3: Implement strict extraction/loading checks**

Persist source keys, shapes, dtypes, per-tensor checksums, target/tokenizer
fingerprints, vocabulary size, and `logit_transform = "identity"`. Reject
unsupported target heads rather than silently treating them as linear.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused test and commit as `feat: harden frozen target IO contract`.

### Task 5: Implement aligned DFlash and DFlash2 consumers

**Files:**
- Create: `src/glm_dflash2/method_objectives.py`
- Modify: `src/glm_dflash2/dflash2_model.py`
- Modify: `src/glm_dflash2/offline_trainer.py`
- Test: `tests/test_method_objectives.py`
- Modify test: `tests/test_offline_trainer.py`
- Modify test: `tests/test_dflash2_objective.py`

- [ ] **Step 1: Write failing objective/trainer tests**

For both methods assert DFlash CE uses `exp(-d/7)` and globally reducible numerator/denominator. For DFlash2 assert predecessor is anchor at depth 0 and teacher previous target afterward, selector CE applies only when target is in top-16, zero global selector denominator creates differentiable zero, and neither method reads/changes `target_final_hidden`.

Assert the production selector is fixed at rank 256/top-16 and cannot be
mutated by the aligned CLI.

- [ ] **Step 2: Verify RED**

Run focused objective/trainer tests.

- [ ] **Step 3: Implement shared preparation and two trainers**

Add `OfflineDFlashTrainer` and refactor `OfflineDFlash2Trainer` to consume the shared block builder/backbone. Preserve frozen target embedding and LM head; expose additive metrics only.

- [ ] **Step 4: Verify GREEN and commit**

Run focused tests and commit as `feat: add aligned DFlash and DFlash2 trainers`.

### Task 6: Implement the aligned DSpark consumer

**Files:**
- Create: `src/glm_dflash2/dspark_model.py`
- Modify: `src/glm_dflash2/method_objectives.py`
- Modify: `src/glm_dflash2/offline_trainer.py`
- Test: `tests/test_dspark_model.py`
- Test: `tests/test_dspark_objective.py`
- Test: `tests/test_dspark_trainer.py`

- [ ] **Step 1: Write failing DSpark tests**

Use tiny vocab fixtures to assert: target token at `a+d+1` uses final hidden at `a+d`; frozen LM-head reconstruction equals dense logits; Markov bias is applied before draft softmax; full-vocab FP32 TV equals `0.5*L1`; confidence target is `clamp(1 - TV, 0, 1)` and detached; and total loss is `0.1 CE + 0.9 TV + 1.0 BCE` with common depth weights.

Assert DSpark predecessor IDs use the clean anchor at depth 0 and the
teacher-forced previous target thereafter. Instrument dtype flow to prove the
LM-head matmul is BF16 and only its logits are promoted to FP32 for softmax and
loss.

- [ ] **Step 2: Verify RED**

Run the three new test modules and require import/behavior failures.

- [ ] **Step 3: Implement DSpark modules and exact chunked distribution loss**

Implement the low-rank Markov head and confidence head on the common backbone. Reconstruct target logits from `target_final_hidden[a+d] @ lm_head.T`; compute numerically exact full-vocabulary softmax/TV in bounded chunks using FP32 log-sum-exp passes, without top-k approximation.

- [ ] **Step 4: Add `OfflineDSparkTrainer` and strict v2 requirement**

Reject legacy v1 or missing/incorrect final semantics before training. Keep target probabilities and LM head frozen/detached.

- [ ] **Step 5: Verify GREEN and commit**

Run focused tests and commit as `feat: add aligned offline DSpark training`.

### Task 7: Unify training/export CLIs and Ascend launchers

**Files:**
- Create: `tools/train_drafter_offline.py`
- Create: `scripts/train_glm52_drafter_910b.sh`
- Modify: `tools/train_dflash2_offline.py`
- Modify: `scripts/train_glm52_dflash2_910b.sh`
- Modify: `src/glm_dflash2/__init__.py`
- Test: `tests/test_unified_train_cli.py`
- Modify test: `tests/test_training_launchers.py`

- [ ] **Step 1: Write failing CLI/launcher tests**

Assert `--method` selects only method-specific modules/loss coefficients, all aligned methods reject v1, architecture flags cannot mutate the approved shape, the same cache manifest/sample IDs are recorded in each run, Ascend uses torch-npu/FSDP2, and DSpark export includes frozen-I/O provenance but not target backbone weights.

Assert all three exported configs contain the identical ordered logical layer
IDs, target/draft dimensions, 64/64/head-dim-64 full-attention fields and
absolute-position contract. Add exact-load fixtures for the Ascend serving
fork's DFlash, DFlash2, and DSpark class dispatch.

- [ ] **Step 2: Verify RED**

Run CLI and launcher test modules.

- [ ] **Step 3: Implement unified CLI and compatibility wrappers**

Share data loader, sampler, optimizer, scheduler, distributed reductions, checkpoint/resume, and export. Keep old DFlash2 entrypoints as warning-emitting wrappers to `--method dflash2`.

- [ ] **Step 4: Verify GREEN and commit**

Run focused tests and commit as `feat: add unified GLM drafter training entrypoint`.

### Task 8: Add numerical parity gate and end-to-end smoke

**Files:**
- Modify: `tools/validate_hidden_cache.py`
- Create: `tools/calibrate_hidden_capture_gate.py`
- Create: `tests/test_hidden_parity_gate.py`
- Create: `tests/test_unified_pipeline.py`
- Modify: `scripts/smoke_no_model.sh`
- Modify: `README.md`
- Modify: `docs/ASCEND_910B_RUNBOOK.md`

- [ ] **Step 1: Write failing parity/smoke tests**

Test `cosine_error=1-cosine_similarity`, max/mean absolute errors, version/fingerprint matching, strict checked-in bounds, three-run calibration, shifted-layer and pre-norm negative controls, one-forward accounting, identical sample IDs across three tiny method runs, and one optimizer step per method. For every metric assert the stored bound is exactly `max(explicit_floor, 2 * worst_direct_vs_direct_variation)`, remains strictly below both negative-control errors, and the artifact persists the floor, observed variation, final bound, target/runtime fingerprints, CANN, torch-npu, and SGLang versions.

- [ ] **Step 2: Verify RED**

Run the new parity and unified pipeline tests.

- [ ] **Step 3: Implement gate tooling and documentation**

Calibration emits an artifact but production validation never recalibrates silently. Document the one real-910B hardware gate that remains external to CPU tests, exact Stage A/B/train commands, storage cost, and legacy-v1 restrictions.

The hardware gate must strictly load each exported checkpoint in the exact
Ascend SGLang/vLLM fork and compare a fixed batch's offline and runtime base
logits. It additionally compares DFlash2 selector logits and DSpark Markov and
confidence outputs, keyed by the same token IDs, anchors, positions, and cache
fingerprint.

- [ ] **Step 4: Verify focused GREEN**

Run the new tests and `bash scripts/smoke_no_model.sh`.

- [ ] **Step 5: Run full verification**

Run:

```bash
PY=/inspire/ssd/project/sais-bio/public/chenbaoyou/miniconda3/envs/nemo/bin/python
PYTHONPATH=src "$PY" -m unittest discover -s tests -v
bash scripts/smoke_no_model.sh
git diff --check
```

Require zero test failures, smoke exit code 0, and no whitespace errors.

- [ ] **Step 6: Commit**

`git add tools/validate_hidden_cache.py tools/calibrate_hidden_capture_gate.py tests/test_hidden_parity_gate.py tests/test_unified_pipeline.py scripts/smoke_no_model.sh README.md docs/ASCEND_910B_RUNBOOK.md && git commit -m "test: verify unified GLM drafter pipeline"`

## Hardware acceptance remaining after local completion

The local implementation is not allowed to claim real GLM-5.2 capture parity without Ascend evidence. On the 910B server, run the versioned parity gate first; only after all five auxiliary tensors, final post-norm hidden, reconstructed logits, and both negative controls pass may a full cache or full training run begin.
