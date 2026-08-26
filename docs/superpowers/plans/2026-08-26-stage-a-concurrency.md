# Stage A Bounded Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded, resumable Stage A trajectory concurrency without changing the default serial behavior or the frozen-token data contract.

**Architecture:** Client workers run independent trajectory episodes with thread-local model clients. The main thread remains the sole owner of JSONL commits, error-ledger updates, and progress reporting. SGLang separately limits active model requests and the aggregate token pool.

**Tech Stack:** Python 3.11+, `concurrent.futures`, SGLang OpenAI endpoint, `unittest`, Bash.

---

### Task 1: Server token-pool limit

**Files:**
- Modify: `src/glm_dflash2/sglang_stage_a.py`
- Test: `tests/test_sglang_stage_a.py`

- [ ] Add a failing test that expects `--max-total-tokens` in the local server command.
- [ ] Add the optional server configuration field and CLI routing.
- [ ] Run the focused test.

### Task 2: Bounded trajectory executor

**Files:**
- Modify: `tools/generate_trajectories.py`
- Test: `tests/test_stage_a_concurrency.py`

- [ ] Add failing tests for bounded pending work, out-of-order completion, errors, and serial compatibility.
- [ ] Add thread-local clients and a small reusable bounded executor.
- [ ] Keep all JSONL and error-ledger writes on the main thread.
- [ ] Run focused tests.

### Task 3: Launcher and documentation

**Files:**
- Modify: `scripts/run_stage_a_trajectories.sh`
- Modify: `README.md`
- Test: `tests/test_ascend_launchers.py`

- [ ] Add failing launcher assertions for `WORKERS`, `MAX_RUNNING_REQUESTS`, and `MAX_TOTAL_TOKENS`.
- [ ] Route the variables with safe defaults and document the recommended 8/2 profile.
- [ ] Run launcher tests.

### Task 4: Verification

- [ ] Run all GLM DFlash2 unit tests.
- [ ] Run a mock/no-model Stage A smoke with multiple workers.
- [ ] Confirm defaults remain `workers=1` and `max-running-requests=1`.
