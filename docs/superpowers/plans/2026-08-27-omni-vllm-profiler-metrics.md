# Omni vLLM Profiler Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add faithful preprocessing, per-modality, benchmark-evaluation, workload-identity, component-availability, and TP-worker HBM metrics to the existing vLLM-Ascend Thinker profiler without changing generation behavior.

**Architecture:** Keep `inference_qwen3-omni.py` as the only runtime entry point and extend the existing pure `omni_sd.inference_profile` helper for deterministic contracts and arithmetic. Query torch-npu allocator statistics inside vLLM workers via public `LLM.collective_rpc`; never use parent-process memory or a native model fallback. Tests fake clocks, vLLM outputs, and RPC results so all metric semantics are verified without NPU hardware.

**Tech Stack:** Python 3.12, vLLM/vLLM-Ascend offline `LLM`, torch-npu worker RPC, pytest, PyArrow, YAML.

---

## File map

- Modify `omni-sd-ascend/src/omni_sd/inference_profile.py`: input evaluation metadata, pure scorers, batch/request aggregation, modality summaries, component metadata, HBM worker callables/reduction, fingerprints.
- Modify `omni-sd-ascend/inference_qwen3-omni.py`: precise timing boundaries, RPC lifecycle, atomic output/success marker, CLI flags, report rendering.
- Modify `omni-sd-ascend/tests/test_inference_profile.py`: contract, scorer, arithmetic, fake-engine integration, failure tests.
- Modify `omni-sd-ascend/README.md` and `omni-sd-ascend/AI_HANDOFF.md`: command and metric semantics.

### Task 1: Pure request, scoring, aggregation, and component contracts

**Files:**
- Modify: `omni-sd-ascend/tests/test_inference_profile.py`
- Modify: `omni-sd-ascend/src/omni_sd/inference_profile.py`

- [ ] **Step 1: Write failing tests for evaluation metadata and scorer v1**

Add focused tests that prove `evaluation` survives normalization, exact match is literal, normalized exact match uses NFKC/casefold/punctuation/whitespace rules, and multiple-choice accepts only the frozen full-string formats.

- [ ] **Step 2: Run scorer tests and verify RED**

Run:

```bash
cd omni-sd-ascend
python -m pytest -q tests/test_inference_profile.py -k 'evaluation or score'
```

Expected: failures because scorer/evaluation functions do not exist.

- [ ] **Step 3: Implement minimal scorer and evaluation aggregation**

Implement `SCORE_VERSION = "omni_eval_v1"`, strict metadata validation, `score_prediction`, and `aggregate_evaluation`. Preserve metadata as canonical JSON and never include it in `prepare_request` inputs.

- [ ] **Step 4: Write failing tests for batch/request and modality arithmetic**

Use two modalities and unequal batch times. Assert engine/e2e TPS use sums, preprocessing has its own distribution, missing vLLM request timestamps remain unavailable, and zero-reference modalities are unavailable rather than zero.

- [ ] **Step 5: Run aggregation tests and verify RED**

Run the focused test names and confirm failures come from the missing new aggregation API.

- [ ] **Step 6: Implement minimal aggregation and component report**

Return `performance.overall` and `performance.by_modality`; keep request preprocessing, optional request engine latency, batch engine latency, and outer batch e2e latency separate. Add explicit loaded/executed/timing fields for all six components.

- [ ] **Step 7: Run the complete profile unit test file**

Run:

```bash
python -m pytest -q tests/test_inference_profile.py
```

Expected: all tests in the file pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add omni-sd-ascend/src/omni_sd/inference_profile.py \
        omni-sd-ascend/tests/test_inference_profile.py
git commit -m "feat: add omni profile metric contracts"
```

### Task 2: vLLM worker HBM telemetry

**Files:**
- Modify: `omni-sd-ascend/tests/test_inference_profile.py`
- Modify: `omni-sd-ascend/src/omni_sd/inference_profile.py`

- [ ] **Step 1: Write failing HBM worker/reduction tests**

Test exact TP rank count, unique physical devices, per-batch peak reset/snapshot records, `final_current`, `max_post_batch_current`, per-rank maxima, and `max_batch_sum_of_rank_peaks`. Test missing/duplicate ranks fail and `allow_missing` makes the entire section unavailable.

- [ ] **Step 2: Run HBM tests and verify RED**

Expected: failures because RPC callables/reduction do not exist.

- [ ] **Step 3: Implement torch-npu worker callables and pure reducer**

The reset callable synchronizes and resets each worker allocator peak. The snapshot callable synchronizes and returns rank, logical/physical device, current allocated/reserved, and peak allocated/reserved bytes. Imports stay lazy so CPU tests remain dependency-light.

- [ ] **Step 4: Run HBM tests and full profile tests**

Expected: all pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add omni-sd-ascend/src/omni_sd/inference_profile.py \
        omni-sd-ascend/tests/test_inference_profile.py
git commit -m "feat: collect vllm worker hbm metrics"
```

### Task 3: Integrate precise timing, identity, and atomic outputs

**Files:**
- Modify: `omni-sd-ascend/tests/test_inference_profile.py`
- Modify: `omni-sd-ascend/inference_qwen3-omni.py`
- Modify: `omni-sd-ascend/src/omni_sd/inference_profile.py`

- [ ] **Step 1: Write failing fake-engine integration tests**

Use an injected monotonic clock and fake `collective_rpc`. Assert warmup is excluded; measured preprocessing/engine/e2e clocks are distinct; exact IDs remain intact; modality, evaluation result, batch timings, HBM, components, and workload/variant identities appear in the final report.

- [ ] **Step 2: Run integration tests and verify RED**

Expected: failures from the old `_generate` tuple and old flat performance report.

- [ ] **Step 3: Implement the measured batch object and profile integration**

Time each `prepare_request`, the enclosing `LLM.generate`, and the outer batch independently. Reset/snapshot HBM around measured batches only. Add `--allow-missing-hbm` and `--variant-id` without changing engine or sampling kwargs.

- [ ] **Step 4: Add deterministic workload identity tests and implementation**

Hash normalized rows/order/batches, actual local media bytes, evaluation metadata, engine/sampling/runtime identity, and model/processor artifact manifests. Reject unfrozen remote media for strict identity.

- [ ] **Step 5: Add atomic-output failure test and implementation**

Prove a failure leaves no final JSONL/profile/success marker. On success atomically rename both files and write a checksum-bound success marker last.

- [ ] **Step 6: Run complete profiler tests**

```bash
python -m pytest -q tests/test_inference_profile.py tests/test_ascend_launchers.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add omni-sd-ascend/inference_qwen3-omni.py \
        omni-sd-ascend/src/omni_sd/inference_profile.py \
        omni-sd-ascend/tests/test_inference_profile.py
git commit -m "feat: extend vllm ascend omni profiler"
```

### Task 4: Documentation and complete verification

**Files:**
- Modify: `omni-sd-ascend/README.md`
- Modify: `omni-sd-ascend/AI_HANDOFF.md`

- [ ] **Step 1: Document command, fields, and unavailable stages**

Document strict vLLM-only execution, the HBM RPC requirement/opt-out, evaluation JSONL schema, modality summaries, success marker, and why Talker/MTP/Code2Wav remain unavailable.

- [ ] **Step 2: Run focused and repository test suites**

```bash
cd omni-sd-ascend
python -m pytest -q
python -m py_compile inference_qwen3-omni.py $(find src scripts/data -name '*.py' -type f)
for script in scripts/*.sh; do bash -n "$script"; done
```

Then from repository root:

```bash
python -m pytest -q
git diff --check
git status --short
```

- [ ] **Step 3: Inspect the final diff against the approved scope**

Confirm no data-generation, hidden-extraction, training, model-loading, engine,
or sampling behavior changed.

- [ ] **Step 4: Commit documentation**

```bash
git add omni-sd-ascend/README.md omni-sd-ascend/AI_HANDOFF.md
git commit -m "docs: explain omni profiler metrics"
```
