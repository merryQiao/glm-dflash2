# Omni Ascend Inference Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-aligned `inference_qwen3-omni.py` profiler for the vLLM-Ascend Qwen3-Omni Thinker path.

**Architecture:** Put input normalization and metric calculation in a small importable module. Keep the top-level CLI responsible for validated configuration, lazy engine allocation, batching, atomic output, and human-readable reporting. Reuse the existing production provider rather than duplicating model or sampling settings.

**Tech Stack:** Python 3.12, vLLM/vLLM-Ascend, Transformers processor, PyArrow, PyYAML, pytest.

---

### Task 1: Performance contracts

**Files:**
- Create: `omni-sd-ascend/tests/test_inference_profile.py`
- Create: `omni-sd-ascend/src/omni_sd/inference_profile.py`

- [ ] Write tests for record normalization, media path resolution, measured-only aggregation, percentiles, and empty-result rejection.
- [ ] Run the new test and verify it fails because the module does not exist.
- [ ] Implement only the pure data and metric helpers.
- [ ] Run the new test and all existing tests.

### Task 2: Production-aligned CLI

**Files:**
- Create: `omni-sd-ascend/inference_qwen3-omni.py`
- Modify: `omni-sd-ascend/tests/test_inference_profile.py`

- [ ] Add failing tests proving `--dry-run` is dependency-light and the CLI delegates engine/request/sampling construction to the production provider.
- [ ] Run the focused tests and verify the expected failure.
- [ ] Implement text, JSONL, and Parquet sources; warmup; measured batching; atomic JSONL/profile output; and concise console reporting.
- [ ] Run focused and complete test suites.

### Task 3: Ascend launcher and documentation

**Files:**
- Create: `omni-sd-ascend/scripts/profile_thinker_ascend.sh`
- Modify: `omni-sd-ascend/README.md`
- Modify: `omni-sd-ascend/tests/test_ascend_launchers.py`

- [ ] Add a failing launcher test for the shared Ascend environment and absence of torchrun/CUDA.
- [ ] Implement the launcher and document dry-run plus hardware performance commands.
- [ ] Run shell syntax, Python compile, focused tests, and the complete suite.
- [ ] Run `--dry-run` locally and record the hardware limitation for real TPS.
