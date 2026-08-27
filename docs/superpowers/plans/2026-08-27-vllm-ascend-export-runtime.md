# GLM-5.2 vLLM-Ascend Export and Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the existing two-stage GLM-5.2 data/training pipeline while producing separate DFlash, DFlash2, and DSpark candidate exports, binding successful candidates to the actual vLLM-Ascend runtime, and evaluating them through one lossless serial benchmark.

**Architecture:** SGLang remains responsible for exact sampled token trajectories and post-final-norm hidden replay. The offline trainer dispatches to three method-specific vLLM-Ascend exporters through a compatibility facade. Exports are non-deployable candidates until a runtime parity command creates an immutable attestation bound to the candidate files and current Ascend environment; the evaluation launcher validates that attestation before starting servers.

**Tech Stack:** Python 3.12, PyTorch, safetensors, Hugging Face config objects, SGLang Stage A/B, vLLM/vLLM-Ascend runtime, Ascend CANN/torch-npu, unittest/pytest, Bash.

---

All commands below run from the repository root and assume:

```bash
export PY=${PY:-/inspire/ssd/project/sais-bio/public/chenbaoyou/miniconda3/envs/nemo/bin/python}
export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH}
```

## File map

### Preserve

- `src/glm_dflash2/sglang_stage_a.py`: exact response-token capability and Stage A bookkeeping.
- `src/glm_dflash2/sglang_hidden_runner.py`: SGLang post-final-norm Stage B replay.
- `src/glm_dflash2/hidden_cache.py`: cache-v2 storage and checksums.
- `src/glm_dflash2/offline_trainer.py`: framework-neutral training loop.
- `src/glm_dflash2/draft_backbone.py`, `dflash2_model.py`, `dspark_model.py`: training models.

### Create

- `src/glm_dflash2/vllm_ascend/__init__.py`: stable export/attestation public API.
- `src/glm_dflash2/vllm_ascend/export_common.py`: shared validation, candidate writer, hashes, and loader.
- `src/glm_dflash2/vllm_ascend/export_dflash.py`: DFlash config and key mapping.
- `src/glm_dflash2/vllm_ascend/export_dflash2.py`: DFlash2 config and key mapping.
- `src/glm_dflash2/vllm_ascend/export_dspark.py`: DSpark config and key mapping.
- `src/glm_dflash2/vllm_ascend/capability.py`: pinned-runtime identity and method capability contract.
- `src/glm_dflash2/vllm_ascend/parity.py`: candidate-bound deploy attestation.
- `integrations/vllm_ascend/dflash2_proposer.py`: version-pinned custom DFlash2 proposer.
- `integrations/vllm_ascend/dflash2_model_loader.py`: strict DFlash2 checkpoint loader.
- `integrations/vllm_ascend/apply_patch.sh`: idempotent, version-gated integration installer.
- `integrations/vllm_ascend/VERSION`: supported vLLM/vLLM-Ascend commits and patch revision.
- `tools/attest_vllm_ascend_export.py`: turn a tested candidate into an attested export.
- `scripts/generate_trajectories.sh`, `extract_hidden_sglang.sh`, `train_drafter.sh`: canonical stage launchers.
- `tests/fixtures/vllm_speculators_dspark_abi.json`: version-pinned DSpark config/state/runtime ABI witness.
- Focused tests under `tests/test_vllm_export_*.py`, `tests/test_vllm_capability.py`, and `tests/test_dflash2_vllm_adapter.py`.

### Modify

- `src/glm_dflash2/speculator_export.py`: compatibility facade only.
- `src/glm_dflash2/sglang_hidden_runner.py` and `tools/extract_hidden_sglang.py`: enforce and record the Stage B runtime contract.
- `src/glm_dflash2/vllm_eval.py`, `tools/benchmark_vllm_ascend.py`, `scripts/eval_vllm_ascend.sh`: attestation, token IDs, counters, and rejection-mode gates.
- `tools/train_drafter_offline.py`: call the facade without changing training behavior.
- `README.md`, `AI_HANDOFF.md`, `docs/ASCEND_910B_RUNBOOK.md`: document the canonical two-stage and runtime flow.

---

### Task 1: Freeze the two independent data stages

**Files:**
- Create: `scripts/generate_trajectories.sh`
- Create: `scripts/extract_hidden_sglang.sh`
- Create: `scripts/train_drafter.sh`
- Modify: `src/glm_dflash2/sglang_hidden_runner.py`
- Modify: `tools/extract_hidden_sglang.py`
- Test: `tests/test_ascend_launchers.py`
- Test: `tests/test_sglang_hidden_runner.py`
- Test: `tests/test_sglang_stage_a.py`

- [ ] **Step 1: Write failing launcher tests**

Assert the three canonical scripts exist and do not invoke the next stage. Stage A
must produce only a frozen trajectory JSONL/manifest; Stage B must require an
explicit frozen trajectory path; training must require an explicit frozen cache.
Keep `build_two_pass_cache.sh` as a clearly deprecated compatibility wrapper.

- [ ] **Step 2: Run the focused tests and verify the expected missing-script failures**

Run:

```bash
$PY -m unittest tests.test_ascend_launchers tests.test_sglang_stage_a -v
```

Expected: FAIL because the canonical independent launchers do not exist.

- [ ] **Step 3: Add Stage B contract tests**

Test that runner metadata contains post-final-norm semantics, exact logical and
physical layer mapping, SGLang/runtime identity, Model Runner identity, CANN,
torch-npu, dtype, topology, graph/cache/chunked-prefill settings, and device type.
Test that chunked prefill, prefix/radix cache, graphs, concurrent requests, an
unknown Model Runner, DP other than one, or PP other than one fail before model
allocation.

- [ ] **Step 4: Run and verify the contract tests fail for missing identity/gates**

```bash
$PY -m unittest tests.test_sglang_hidden_runner tests.test_ascend_launchers -v
```

Expected: FAIL on absent runtime fields and unsupported-setting validation.

- [ ] **Step 5: Implement the minimal launchers and runtime contract**

The canonical launchers forward to existing tested tools. Add one pure validation
function in `sglang_hidden_runner.py` and call it before `load_model`. Preserve the
existing `model.norm` forward hook; do not introduce a vLLM hidden provider.

- [ ] **Step 6: Run focused and existing Stage A/B tests**

```bash
$PY -m unittest \
  tests.test_ascend_launchers \
  tests.test_sglang_stage_a \
  tests.test_sglang_hidden_runner \
  tests.test_hidden_extraction \
  tests.test_hidden_capture -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/generate_trajectories.sh scripts/extract_hidden_sglang.sh \
  scripts/train_drafter.sh src/glm_dflash2/sglang_hidden_runner.py \
  tools/extract_hidden_sglang.py tests/test_ascend_launchers.py \
  tests/test_sglang_hidden_runner.py tests/test_sglang_stage_a.py
git commit -m "refactor: separate GLM trajectory and hidden stages"
```

### Task 2: Build a common candidate-export core

**Files:**
- Create: `src/glm_dflash2/vllm_ascend/__init__.py`
- Create: `src/glm_dflash2/vllm_ascend/export_common.py`
- Create: `tests/test_vllm_export_common.py`

- [ ] **Step 1: Write the candidate-export tests**

Define a wished-for API with:

```python
candidate = write_candidate_export(
    output_dir,
    method="dflash",
    config=config,
    weights=weights,
    target_io=target_io,
    method_metadata=metadata,
)
```

Assert `config.json`, `model.safetensors`, optional remote-code files, and
`export_manifest.json` are atomically committed. The manifest must be schema v2,
status `candidate-not-deployable`, and bind config/weights/target-I/O/checkpoint
hashes, target/tokenizer identities, ordered layer IDs, block size, proposal count,
anchor policy, and method parameters. A deploy attestation must not exist.

- [ ] **Step 2: Verify RED**

```bash
$PY -m unittest tests.test_vllm_export_common -v
```

Expected: import failure because the common exporter does not exist.

- [ ] **Step 3: Implement shared validation and atomic candidate writing**

Move only `_sha256`, transformer-config serialization, target-I/O checks, common
frozen embedding/head insertion, and strict candidate loading out of the legacy
module. Reject duplicate keys and incompatible vocab/hidden sizes.

- [ ] **Step 4: Test corruption and partial-write recovery**

Add tests that mutate one byte of config or weights, remove a required identity,
or leave a temporary partial directory. Loading must fail closed and never mistake
the partial directory for a candidate.

- [ ] **Step 5: Verify GREEN**

```bash
$PY -m unittest tests.test_vllm_export_common tests.test_target_io -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/glm_dflash2/vllm_ascend tests/test_vllm_export_common.py
git commit -m "feat: add immutable vLLM candidate export core"
```

### Task 3: Split the DFlash exporter

**Files:**
- Create: `src/glm_dflash2/vllm_ascend/export_dflash.py`
- Create: `tests/test_vllm_export_dflash.py`

- [ ] **Step 1: Write failing B8/B16 config and key tests**

Assert architecture/config class, ordered auxiliary layer IDs, full-attention
five-layer Qwen3-shaped backbone, `sample_from_anchor=false`, B8→7 proposals,
B16→15 proposals, frozen target embedding/head, and an exact state-key set. Reject
other block sizes and any unexpected DFlash2/DSpark tensors.

- [ ] **Step 2: Verify RED**

```bash
$PY -m unittest tests.test_vllm_export_dflash -v
```

- [ ] **Step 3: Implement DFlash-only config and mapping**

Use the pinned `DFlashSpeculatorConfig` ABI. Keep runtime compatibility as a
candidate claim; do not mark GLM-5.2 target hidden extraction supported merely
because config generation succeeds.

- [ ] **Step 4: Verify export/load round-trip and negative keys**

```bash
$PY -m unittest tests.test_vllm_export_dflash -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/glm_dflash2/vllm_ascend/export_dflash.py \
  tests/test_vllm_export_dflash.py
git commit -m "feat: add method-specific DFlash export"
```

### Task 4: Split and align the DSpark exporter

**Files:**
- Create: `src/glm_dflash2/vllm_ascend/export_dspark.py`
- Create: `tests/fixtures/vllm_speculators_dspark_abi.json`
- Create: `tests/test_vllm_export_dspark.py`

- [ ] **Step 1: Freeze the upstream ABI fixture**

Record the tested `vllm-project/speculators` and vLLM commits in test metadata.
The fixture must encode `DSparkSpeculatorConfig`, five-layer DFlash backbone,
`markov_rank=256`, vanilla Markov head, confidence head, B8, seven proposals, and
`sample_from_anchor=false`. The fixture describes file/config/state keys, not the
public checkpoint's target layer IDs.

- [ ] **Step 2: Write failing DSpark config/key tests**

Assert ordered `[1,20,38,56,75]`, exact Markov embedding/projection keys, confidence
head keys, BF16 tensor shapes, and absence of DFlash2 selector tensors. Assert B16
is rejected for the project's DSpark setting.

- [ ] **Step 3: Verify RED**

```bash
$PY -m unittest tests.test_vllm_export_dspark -v
```

- [ ] **Step 4: Implement the minimal DSpark mapping**

Map training keys to the pinned vLLM/speculators ABI without changing training
FQNs. Include `num_speculative_tokens=7` explicitly in config and candidate
manifest.

- [ ] **Step 5: Add deterministic proposal-parity witnesses**

With fixed hidden states, anchor IDs, and teacher-forced predecessors, compare
offline Markov bias, confidence, and all seven proposal positions after
export/load. Store no reusable “runtime passed” boolean.

- [ ] **Step 6: Verify GREEN**

```bash
$PY -m unittest tests.test_vllm_export_dspark tests.test_dspark_model -v
```

- [ ] **Step 7: Commit**

```bash
git add src/glm_dflash2/vllm_ascend/export_dspark.py \
  tests/fixtures/vllm_speculators_dspark_abi.json \
  tests/test_vllm_export_dspark.py
git commit -m "feat: align DSpark export with vLLM ABI"
```

### Task 5: Isolate DFlash2 export and runtime adapter

**Files:**
- Create: `src/glm_dflash2/vllm_ascend/export_dflash2.py`
- Create: `integrations/vllm_ascend/dflash2_proposer.py`
- Create: `integrations/vllm_ascend/dflash2_model_loader.py`
- Create: `integrations/vllm_ascend/apply_patch.sh`
- Create: `integrations/vllm_ascend/VERSION`
- Create: `tests/test_vllm_export_dflash2.py`
- Create: `tests/test_dflash2_vllm_adapter.py`

- [ ] **Step 1: Write failing DFlash2 export tests**

Assert a distinct architecture/model type; B8/B16 proposal counts; dynamic
convolution metadata; selector rank 256 and top-k; exact selector/codebook/kernel
keys; and strict absence of DSpark heads. The candidate may never advertise
`method=dflash`.

- [ ] **Step 2: Verify RED**

```bash
$PY -m unittest tests.test_vllm_export_dflash2 -v
```

- [ ] **Step 3: Implement the DFlash2 exporter**

Use the common writer and preserve all runtime-required selector tensors. Mark the
candidate as requiring the version-pinned custom proposer.

- [ ] **Step 4: Write failing adapter contract tests with fake vLLM interfaces**

Test strict candidate loading, target auxiliary-hidden input shape, base top-k
creation, selector reranking, proposal count, and returned token IDs. Test version
mismatch, missing selector weights, wrong block size, and empty proposals.

- [ ] **Step 5: Implement the smallest adapter**

The adapter may import vLLM only inside runtime entry points so CPU tests can load
the repository. Training/cache modules must never import `integrations`.
`apply_patch.sh` checks exact source fingerprints or commits, supports dry-run, is
idempotent, and refuses an unknown vLLM-Ascend tree.

- [ ] **Step 6: Verify GREEN**

```bash
$PY -m unittest \
  tests.test_vllm_export_dflash2 \
  tests.test_dflash2_vllm_adapter \
  tests.test_dflash2_model -v
```

- [ ] **Step 7: Commit**

```bash
git add src/glm_dflash2/vllm_ascend/export_dflash2.py \
  integrations/vllm_ascend tests/test_vllm_export_dflash2.py \
  tests/test_dflash2_vllm_adapter.py
git commit -m "feat: isolate DFlash2 vLLM-Ascend adapter"
```

### Task 6: Bind capability and parity to a concrete candidate

**Files:**
- Create: `src/glm_dflash2/vllm_ascend/capability.py`
- Create: `src/glm_dflash2/vllm_ascend/parity.py`
- Create: `tools/attest_vllm_ascend_export.py`
- Create: `tests/test_vllm_capability.py`
- Create: `tests/test_vllm_deploy_attestation.py`

- [ ] **Step 1: Write failing runtime-identity tests**

Define a canonical identity containing vLLM, vLLM-Ascend, speculators, adapter,
CANN, torch-npu, driver/firmware, device, attention backend, runner generation,
TP/EP/PP/DP/nodes, graph, chunked-prefill, and prefix-cache values. Unknown
production fields must fail; a smoke report may remain explicitly untrusted.

- [ ] **Step 2: Verify RED**

```bash
$PY -m unittest tests.test_vllm_capability -v
```

- [ ] **Step 3: Implement capability collection and strict comparison**

Keep collection dependency-light and allow deterministic injected fixtures for
CPU tests. Never interpret “method appears in support matrix” as a passed parity
gate.

- [ ] **Step 4: Write failing candidate-bound attestation tests**

The attestation must bind export config/weights, target I/O, checkpoint, target,
tokenizer, method settings, runtime identity, topology, fixture IDs, numerical
thresholds, and individual load/logit/proposal/token/rejection/counter results.
Mutating one byte or field must make validation fail.

- [ ] **Step 5: Implement candidate → test results → atomic attestation**

`attest_vllm_ascend_export.py` consumes machine-readable parity results from the
actual runtime; it cannot fabricate them. It writes `deploy_attestation.json` only
when every required gate passes, then updates candidate status atomically without
changing config or weights.

- [ ] **Step 6: Verify GREEN**

```bash
$PY -m unittest \
  tests.test_vllm_capability \
  tests.test_vllm_deploy_attestation -v
```

- [ ] **Step 7: Commit**

```bash
git add src/glm_dflash2/vllm_ascend/capability.py \
  src/glm_dflash2/vllm_ascend/parity.py \
  tools/attest_vllm_ascend_export.py \
  tests/test_vllm_capability.py tests/test_vllm_deploy_attestation.py
git commit -m "feat: bind deploy attestation to runtime and export"
```

### Task 7: Make vLLM-Ascend evaluation formally lossless

**Files:**
- Modify: `src/glm_dflash2/vllm_eval.py`
- Modify: `tools/benchmark_vllm_ascend.py`
- Modify: `scripts/eval_vllm_ascend.sh`
- Modify: `tests/test_vllm_ascend_eval.py`

- [ ] **Step 1: Write failing token/counter/rejection tests**

Require raw response token IDs for greedy parity; output-text equality alone is
insufficient. Keep Prometheus snapshots immediately before and after measured
requests and use counter deltas only. Require counters to exist and proposal/draft
counts to be positive. Reject missing token-ID capability, baseline/speculative
sample-set differences, or changed sampling settings.

- [ ] **Step 2: Verify RED**

```bash
$PY -m unittest tests.test_vllm_ascend_eval -v
```

- [ ] **Step 3: Implement strict benchmark records**

Record output token IDs, completion counts, per-request latency, request fixture
checksum, runtime identity, topology, and sampling/rejection mode. For sampling,
require a runtime capability proving standard rejection sampling; do not require
independent sampled outputs to match.

- [ ] **Step 4: Gate the launcher with deploy attestation**

Read `METHOD`, `TARGET_MODEL`, `DRAFTER_EXPORT`, and `PROMPTS`/legacy
`PROMPTS_JSONL`. Validate current runtime against the bound attestation before
allocating the model. Start baseline and speculative servers serially on the same
devices, run identical warm-ups and measured requests, and fully terminate the
first server before the second launch.

- [ ] **Step 5: Verify GREEN**

```bash
$PY -m unittest tests.test_vllm_ascend_eval -v
bash -n scripts/eval_vllm_ascend.sh
```

- [ ] **Step 6: Commit**

```bash
git add src/glm_dflash2/vllm_eval.py tools/benchmark_vllm_ascend.py \
  scripts/eval_vllm_ascend.sh tests/test_vllm_ascend_eval.py
git commit -m "feat: enforce lossless vLLM-Ascend evaluation"
```

### Task 8: Install the compatibility facade and migrate callers

**Files:**
- Modify: `src/glm_dflash2/speculator_export.py`
- Modify: `src/glm_dflash2/vllm_ascend/__init__.py`
- Modify: `tools/train_drafter_offline.py`
- Modify: `tests/test_speculator_export.py`
- Modify: `tests/test_train_cli.py`

- [ ] **Step 1: Write failing facade compatibility tests**

Existing callers of `export_speculator` and `load_exported_speculator` must still
work, but dispatch into the method-specific modules. Old schema-v1 exports remain
readable and permanently untrusted; new exports use candidate schema v2.

- [ ] **Step 2: Verify RED**

```bash
$PY -m unittest tests.test_speculator_export tests.test_train_cli -v
```

- [ ] **Step 3: Replace the monolith with a thin facade**

Keep only dispatch and legacy-read compatibility in `speculator_export.py`. No
method config or state-key logic may remain there. Preserve checkpoint/trainer
state and export timing.

- [ ] **Step 4: Verify GREEN**

```bash
$PY -m unittest tests.test_speculator_export tests.test_train_cli -v
```

- [ ] **Step 5: Commit**

```bash
git add src/glm_dflash2/speculator_export.py \
  src/glm_dflash2/vllm_ascend/__init__.py tools/train_drafter_offline.py \
  tests/test_speculator_export.py tests/test_train_cli.py
git commit -m "refactor: dispatch speculator export by method"
```

### Task 9: Documentation, full regression, and hardware handoff

**Files:**
- Modify: `README.md`
- Modify: `AI_HANDOFF.md`
- Modify: `docs/ASCEND_910B_RUNBOOK.md`
- Modify: `docs/implementation-plan.md`

- [ ] **Step 1: Update documentation without claiming unrun hardware gates**

Document the six explicit stages, candidate/attestation lifecycle, supported block
sizes, DSpark anchor ABI, DFlash2 patch version, canonical launchers, and exact
remaining 910B tests. Remove vLLM hidden extraction from the recommended path.

- [ ] **Step 2: Run formatting/static checks**

```bash
git diff --check
bash -n scripts/*.sh integrations/vllm_ascend/*.sh
$PY -m compileall -q src tools
```

Expected: no output except explicit success messages.

- [ ] **Step 3: Run the complete CPU suite**

```bash
$PY -m unittest discover -s tests -v
```

Expected: all tests pass, with real Ascend tests skipped or represented only by
fixture/capability tests—not falsely reported as hardware validation.

- [ ] **Step 4: Run no-model smoke commands**

```bash
bash scripts/smoke_no_model.sh
bash scripts/smoke_train_no_npu.sh
```

Expected: PASS without importing vLLM/torch-npu on the local CPU host.

- [ ] **Step 5: Record hardware acceptance commands**

Record, but do not mark passed locally:

1. SGLang Stage B varied-length and final-norm parity on the actual BF16 topology.
2. DFlash B8/B16 candidate load/logit/proposal/acceptance gates.
3. DSpark B8 fixed-anchor predecessor/confidence/proposal gates.
4. DFlash2 B8/B16 adapter load/selector/proposal gates.
5. Greedy token-ID equality and sampling rejection tests.
6. Serial baseline/speculative TPS and acceptance measurement.

- [ ] **Step 6: Commit**

```bash
git add README.md AI_HANDOFF.md docs/ASCEND_910B_RUNBOOK.md \
  docs/implementation-plan.md
git commit -m "docs: hand off GLM vLLM-Ascend runtime gates"
```

## Local completion boundary

Local completion means all CPU tests, export round trips, deterministic adapter
witnesses, shell checks, and no-model smokes pass. It does **not** mean any method
is deployable. A deployable result exists only after the target 910B server runs
the pinned runtime parity suite and creates an attestation bound to that exact
candidate and environment.
