# Omni SD Ascend Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested A2/A3-oriented two-stage Qwen3-Omni Thinker trajectory and hidden-state pipeline on vLLM-Ascend.

**Architecture:** A backend-neutral orchestration layer owns schemas, sharding, exact-token validation, and manifests. A lazy-imported vLLM-Ascend provider owns TP/EP generation and native hidden extraction; a Transformers provider is retained only for parity. Hardware-dependent capabilities fail closed and are recorded by attestation.

**Tech Stack:** Python 3.11+, PyArrow, safetensors, PyYAML, vLLM, vLLM-Ascend, Qwen Omni utilities, pytest/unittest.

---

### Task 1: Configuration contract and runtime identity

**Files:**
- Modify: `omni-sd-ascend/configs/generate_thinker_data.yaml`
- Create: `omni-sd-ascend/src/omni_sd/config.py`
- Create: `omni-sd-ascend/src/omni_sd/ascend_runtime.py`
- Test: `omni-sd-ascend/tests/test_config.py`

- [ ] Write failing tests for complete config validation, BF16/quantization rules,
  layer IDs, A2/A3 device identity, and deterministic runtime fingerprints.
- [ ] Run the tests and verify expected missing-API failures.
- [ ] Implement typed validation and lazy environment/version collection.
- [ ] Run the focused tests and verify pass.

### Task 2: Stage A request and exact-token generation provider

**Files:**
- Create: `omni-sd-ascend/src/omni_sd/vllm_ascend_generation.py`
- Modify: `omni-sd-ascend/scripts/data/generate_thinker_data_vllm.py`
- Modify: `omni-sd-ascend/src/omni_sd/thinker_generation.py`
- Test: `omni-sd-ascend/tests/test_vllm_ascend_generation.py`
- Test: `omni-sd-ascend/tests/test_generation_contract.py`

- [ ] Write failing tests for TP/EP engine kwargs, absence of `device_ids`,
  per-condition seeds, multimodal request preservation, explicit EOS policy,
  and exact engine token recording.
- [ ] Run the tests and verify expected failures.
- [ ] Implement the lazy vLLM-Ascend engine factory and provider protocol.
- [ ] Refactor the Stage A script to use worker IDs rather than torchrun ranks
  inside a TP engine and add recoverable error-ledger output.
- [ ] Run focused tests and verify pass.

### Task 3: Stage B native hidden provider

**Files:**
- Create: `omni-sd-ascend/src/omni_sd/vllm_ascend_hidden.py`
- Modify: `omni-sd-ascend/scripts/data/generate_thinker_hidden.py`
- Test: `omni-sd-ascend/tests/test_vllm_ascend_hidden.py`
- Test: `omni-sd-ascend/tests/test_hidden_contract.py`

- [ ] Write failing tests for native extractor configuration, exact token
  equality, `[T,L,H]` validation, final-normalized hidden capability gating,
  finite checks, and response loss masks.
- [ ] Run the tests and verify expected failures.
- [ ] Implement the lazy native extractor and connector result reader.
- [ ] Refactor Stage B orchestration while preserving atomic safetensors/index
  output.
- [ ] Run focused tests and verify pass.

### Task 4: Provenance, integrity, and resumability

**Files:**
- Create: `omni-sd-ascend/src/omni_sd/provenance.py`
- Modify: `omni-sd-ascend/src/omni_sd/thinker_generation.py`
- Modify: `omni-sd-ascend/scripts/data/generate_thinker_hidden.py`
- Test: `omni-sd-ascend/tests/test_provenance.py`

- [ ] Write failing tests for model/processor/runtime/media fingerprints,
  safetensors/index checksums, duplicate condition IDs, and mixed-run rejection.
- [ ] Run the tests and verify expected failures.
- [ ] Implement provenance records and checksum validation.
- [ ] Run focused tests and verify pass.

### Task 5: A2/A3 launchers and hardware smoke gate

**Files:**
- Create: `omni-sd-ascend/scripts/generate_thinker_trajectories_ascend.sh`
- Create: `omni-sd-ascend/scripts/extract_thinker_hidden_ascend.sh`
- Create: `omni-sd-ascend/scripts/smoke_ascend.sh`
- Create: `omni-sd-ascend/src/omni_sd/parity.py`
- Test: `omni-sd-ascend/tests/test_ascend_launchers.py`
- Test: `omni-sd-ascend/tests/test_parity.py`

- [ ] Write failing tests for visible-device handling, A3 `AIV`, A2 behavior,
  TP/EP flags, BF16 launch rules, and parity report requirements.
- [ ] Run the tests and verify expected failures.
- [ ] Implement launchers and machine-readable hardware attestation.
- [ ] Run shell syntax and focused tests.

### Task 6: Documentation and full verification

**Files:**
- Create: `omni-sd-ascend/README.md`
- Create: `omni-sd-ascend/requirements-data.txt`
- Remove: `omni-sd-ascend/inference_qwen3-omni.py`
- Remove: `omni-sd-ascend/scripts/data/generate_thinker_data.py`

- [ ] Document the official vLLM-Ascend image, A2/A3 device semantics, Stage A
  and Stage B commands, smoke gates, resume behavior, and known runtime gate for
  final normalized hidden.
- [ ] Remove the unrelated incomplete CUDA profiler and duplicate legacy
  Transformers generator.
- [ ] Run `python -m py_compile` over all Python files.
- [ ] Run the complete CPU test suite.
- [ ] Run `bash -n` over every launcher.
- [ ] Confirm the repository status contains only intended files.
