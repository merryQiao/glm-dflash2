# GLM-5.3-Flash Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, tested GLM-5.3-Flash-BF16 Stage A trajectory generator without changing or importing the parent GLM-5.2 implementation.

**Architecture:** Copy only the transitive Stage A modules into a new `glm53_stage_a` package, then adapt model identity, nested configuration, parser options, and runtime capability gates. Preserve the existing trajectory wire/schema behavior while enforcing that unverified endpoint smoke artifacts can never become production-frozen data.

**Tech Stack:** Python 3.11+, SGLang OpenAI-compatible API, Transformers tokenizer, Requests, PyArrow, pytest/unittest, Bash.

---

## File map

- Create `src/glm53_stage_a/`: standalone Stage A runtime package.
- Create `tools/generate_trajectories.py`: GLM-5.3 trajectory CLI.
- Create `tools/prepare_open_swe_trajectories.py`: SQLite preparation utility needed by trajectory-prefix rows.
- Create `scripts/run_stage_a_trajectories.sh`: Ascend/local or external-endpoint launcher.
- Create `tests/`: copied regression tests plus GLM-5.3-specific tests.
- Create `README.md`, `requirements.txt`: standalone usage and dependencies.
- Do not modify any path outside `glm-5.3-flash/`.

### Task 1: Establish the standalone boundary

**Files:**
- Create: `tests/test_glm53_identity.py`
- Create: `src/glm53_stage_a/__init__.py`
- Create: copied Stage A modules listed in the design
- Create: copied CLI tools and regression tests

- [ ] **Step 1: Write a failing import-isolation test**

The test imports every `glm53_stage_a` module and both tools, enumerates loaded
package modules, and asserts every module path is below `glm-5.3-flash/src`; it
also rejects loaded or source-level `glm_dflash2` references. Before copying,
record `git status --short` and SHA-256 digests for the parent Stage A source,
tools, launcher, and tests that serve as references.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src python -m pytest tests/test_glm53_identity.py -q`

Expected: FAIL because `glm53_stage_a` does not exist.

- [ ] **Step 3: Copy the minimal dependency closure and rename imports**

Copy only `agent_trajectory`, `jsonl`, `open_swe_trajectories`, `provenance`,
`sglang_stage_a`, `trajectory_tokens`, `vibe_coding`, `web_tools`, and
`workspaces`; move `model_revision` and tokenizer fingerprinting into
`provenance.py`. Copy the two Stage A tools. Copy and adapt the behavior covered
by `test_agent_trajectory.py`, `test_jsonl.py`, `test_provenance.py`,
`test_sglang_stage_a.py`, `test_stage_a_concurrency.py`,
`test_trajectory_tokens.py`, `test_vibe_coding.py`, and `test_workspaces.py`;
exclude the parent GLM-5.2 v0.5.16 patch-installer tests.

- [ ] **Step 4: Verify GREEN and regression behavior**

Run the isolation test and copied Stage A regression tests.

Expected: all pass without adding the parent repository to `PYTHONPATH`.

### Task 2: Add the GLM-5.3 server profile

**Files:**
- Modify: `src/glm53_stage_a/sglang_stage_a.py`
- Modify: `tools/generate_trajectories.py`
- Test: `tests/test_sglang_stage_a.py`

- [ ] **Step 1: Write failing tests**

Assert the defaults are `GLM-5.3-Flash-BF16`, `glm45`, and `glm47`; assert parser
settings are configurable and appear in the server command. Assert the run
contract records reasoning/tool parsers, device, attention backend, dtype,
quantization, MoE A2A backend, and DeepEP mode, and that resume rejects a change
to any of these fields.

- [ ] **Step 2: Verify RED**

Expected: FAIL because copied defaults still name GLM-5.2 and parser CLI options are absent.

- [ ] **Step 3: Implement the minimal profile changes**

Expose `--reasoning-parser` and `--tool-call-parser`, pass them through
`SGLangServerConfig`, replace GLM-5.2 defaults/docstrings/temp names, and persist
all rollout-affecting server fields in the service section of the run contract.

- [ ] **Step 4: Verify GREEN**

Run `tests/test_sglang_stage_a.py`.

### Task 3: Support multimodal nested model metadata and immutable templates

**Files:**
- Modify: `src/glm53_stage_a/provenance.py`
- Modify: `tools/generate_trajectories.py`
- Test: `tests/test_glm53_identity.py`
- Test: `tests/test_provenance.py`

- [ ] **Step 1: Write failing tests**

Test `text_config.vocab_size`, top-level legacy fallback, missing-vocab failure, and fingerprint changes caused by `chat_template.jinja` and processor metadata.

- [ ] **Step 2: Verify RED**

Expected: FAIL on nested vocab and template fingerprint assertions.

- [ ] **Step 3: Implement metadata helpers**

Add a strict `target_vocab_size` helper and extend model/tokenizer artifact patterns to include chat-template and processor metadata. Keep text-only request construction.

- [ ] **Step 4: Verify GREEN**

Run identity and provenance tests.

### Task 4: Make endpoint smoke artifacts non-production

**Files:**
- Modify: `tools/generate_trajectories.py`
- Test: `tests/test_glm53_smoke_contract.py`

- [ ] **Step 1: Write failing tests**

Test that `--allow-unverified-endpoint` requires `1 <= max_samples <= 50`, writes `production_eligible=false`, and completes with `status=smoke_unverified`, never `frozen`.

- [ ] **Step 2: Verify RED**

Expected: FAIL because the copied implementation permits an unbounded smoke and freezes its manifest.

- [ ] **Step 3: Implement the strict contract**

Validate arguments before loading the model/dataset, add manifest eligibility, and select the terminal status based on verified endpoint identity.

- [ ] **Step 4: Verify GREEN**

Run the smoke-contract and concurrency tests.

### Task 5: Remove version-specific patch coupling

**Files:**
- Modify: `src/glm53_stage_a/agent_trajectory.py`
- Test: `tests/test_sglang_stage_a.py`

- [ ] **Step 1: Write a failing test**

Assert token-ID capability errors describe the missing endpoint capability and do not direct users to the GLM-5.2 SGLang 0.5.16 patch.

- [ ] **Step 2: Verify RED**

Expected: FAIL because copied messages mention the v0.5.16 patch.

- [ ] **Step 3: Replace only the error text**

Preserve the capability probe behavior; make its error actionable and runtime-neutral.

- [ ] **Step 4: Verify GREEN**

Run the Stage A client tests.

### Task 6: Add launcher, documentation, and final verification

**Files:**
- Create: `scripts/run_stage_a_trajectories.sh`
- Create: `README.md`
- Create: `requirements.txt`
- Test: `tests/test_launchers.py`

- [ ] **Step 1: Write launcher tests and verify RED**

Check `bash -n`, GLM-5.3 defaults, parser pass-through, explicit model path, and no parent-directory imports.

- [ ] **Step 2: Implement launcher and concise README**

Document local SGLang and external endpoint modes, BF16 identity, the 50-sample smoke boundary, exact token-ID requirement, and the unverified Ascend runtime gate.
Declare the complete standalone dependencies, including `datasets`, `pyarrow`,
`requests`, and the compatible Transformers range.

- [ ] **Step 3: Run focused tests**

Run all tests under `glm-5.3-flash/tests` with only the standalone `src` on `PYTHONPATH`.

- [ ] **Step 4: Run static smoke checks**

Run `bash -n scripts/run_stage_a_trajectories.sh`, compile all Python files, run
`python tools/generate_trajectories.py --help`, and run
`python tools/prepare_open_swe_trajectories.py --help`.

- [ ] **Step 5: Verify parent immutability**

Compare parent repository status/digests captured in Task 1 and confirm no parent
GLM-5.2 source file changed.

- [ ] **Step 6: Record the unresolved hardware gate**

Report that real `glm5_next` serving and exact token-ID return still require a 10–50 sample smoke on the target Ascend SGLang image; do not claim hardware success locally.
