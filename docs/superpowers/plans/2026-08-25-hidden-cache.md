# SGLang Two-Pass Trajectory and Hidden Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce validated GLM-5.2 coding-agent trajectories and selected-layer hidden caches for all 630K source rows with two sequential SGLang passes.

**Architecture:** Vendor the proven SpecForge trajectory components into this self-contained project and adapt the runner to GLM-5.2/SGLang. Freeze committed trajectory JSONL before a standalone SGLang internal model-runner pass captures five target layers and writes an append-only mmap cache.

**Tech Stack:** Python 3.12, SGLang, PyTorch/torch-npu, Transformers tokenizer, PyArrow, requests, NumPy mmap, unittest.

---

### Task 1: Vendor and preserve the trajectory reference

**Files:**
- Create: `src/glm_dflash2/agent_trajectory.py`
- Create: `src/glm_dflash2/vibe_coding.py`
- Create: `src/glm_dflash2/web_tools.py`
- Create: `tools/generate_trajectories.py`
- Test: `tests/test_agent_trajectory.py`
- Test: `tests/test_vibe_coding.py`

- [ ] Copy the three focused SpecForge data modules and the reference runner into `glm-dflash2` without modifying the source checkout.
- [ ] Port the relevant reference tests and confirm they pass against the copied modules.
- [ ] Adapt imports, names, GLM-5.2 defaults, Ascend/SGLang launch flags, source sharding, locking, manifest, error ledger, and resume behavior. JSONL line fsync is the commit point; manifest progress may lag and is rebuilt from committed unique IDs.
- [ ] Add failing tests for the adapted CLI/server command and make them pass.

### Task 2: Freeze exact trajectory tokens and DFlash masks

**Files:**
- Create: `src/glm_dflash2/trajectory_tokens.py`
- Modify: `tools/generate_trajectories.py`
- Test: `tests/test_trajectory_tokens.py`

- [ ] Write failing tests for multiple assistant/tool turns, source assistant turns before `generation_start_message_index`, prefix-instability rejection, empty supervised spans, final-token zeroing, and hard length bounds. Add GLM tokenizer golden fixtures for structured tool calls, empty content, reasoning content, Unicode and historical assistants.
- [ ] Implement final chat-template rendering with frozen ordered tool schemas and identical chat-template kwargs, target-token-position `loss_mask` semantics, strict integer token validation, and per-round server token-ID checks when present.
- [ ] Store `input_ids`, `loss_mask`, tokenizer/chat-template fingerprint, complete replay inputs and token counts in each committed trajectory. Freeze only a complete unique-ID set with no unresolved errors.
- [ ] Run focused and vendored trajectory tests.

### Task 3: Implement the packed hidden cache

**Files:**
- Create: `src/glm_dflash2/hidden_cache.py`
- Test: `tests/test_hidden_cache.py`

- [ ] Write failing tests for variable-length append/read, data-shard ownership, segment rollover at every commit boundary, duplicate IDs, locking, stream tails, partial index lines, index-ahead-of-manifest recovery, committed truncation, checksum corruption, finalize interruption, and manifest mismatch.
- [ ] Implement little-endian raw int64 IDs, uint8 masks, BF16-word hidden streams, JSONL index, atomic/rebuildable manifest, bounded recovery, checksums, and `PackedHiddenDataset`.
- [ ] Add a DFlash adapter/collator returning right-padded `[B,T,30720]`, token IDs and unshifted masks; test the real SpecForge projector/strategy boundary when that checkout is available.
- [ ] Enforce the stream-fsync/index-fsync/manifest-replace commit order and refuse incomplete caches in the reader.
- [ ] Run focused tests.

### Task 4: Implement backend-neutral hidden extraction

**Files:**
- Create: `src/glm_dflash2/hidden.py`
- Create: `src/glm_dflash2/hidden_backends.py`
- Test: `tests/test_hidden.py`
- Test: `tests/test_hidden_backends.py`

- [ ] Write failing tests for deterministic mock extraction, batching, exact token/shape/dtype/finite validation, resume, owned-shard filtering, recoverable sample errors, and fatal backend errors.
- [ ] Implement a backend protocol, mock backend, extraction orchestration, error ledger, and strict completion accounting.
- [ ] Validate `[T, L * H]` and `[T, L, H]` backend outputs, preserving archival `[T,L,H]` and exposing a separate flattened DFlash training view.
- [ ] Run focused tests.

### Task 5: Add the SGLang selected-layer runner

**Files:**
- Create: `src/glm_dflash2/sglang_hidden_runner.py`
- Create: `tools/extract_hidden_states.py`
- Test: `tests/test_sglang_hidden_runner.py`
- Test: `tests/test_hidden_cli.py`

- [ ] Write contract tests with fake SGLang modules for lazy imports, explicit layer capture, `CaptureHiddenMode.FULL`, teacher-forced token IDs, TP rank behavior, cache clearing, and output splitting.
- [ ] Build `ServerArgs` and standalone runner initialization from SGLang's device-aware `benchmark.one_batch` path; expose global/local rank, TP/EP/node settings, reject PP/DP, disable implicit chunking and graphs for the correctness gate, and allow only global rank 0 to write.
- [ ] Prefer `set_dflash_layers_to_capture`; fail clearly if the loaded GLM-5.2 implementation lacks selected-layer capture. Record logical-to-physical mapping/capture convention and add a real hardware comparison against reference per-layer states.
- [ ] On rank 0, reshape and CPU-copy validated aux hidden; all ranks execute identical requests and synchronize before the next batch.
- [ ] Record exact SGLang version/commit and topology in the cache manifest.

### Task 6: Add one-command orchestration and documentation

**Files:**
- Create: `scripts/generate_glm52_trajectories_910b.sh`
- Create: `scripts/extract_glm52_hidden_910b.sh`
- Create: `scripts/prepare_glm52_dflash2_data.sh`
- Create: `tools/validate_hidden_cache.py`
- Modify: `scripts/smoke_no_model.sh`
- Modify: `README.md`
- Modify: `docs/design.md`
- Modify: `docs/implementation-plan.md`
- Modify: `requirements-data.txt`

- [ ] Add failing CLI/config tests for model placeholders, local/offline resolution, layer parsing, storage preflight, hard sequence bounds, topology/data-shard independence, and incomplete outputs.
- [ ] Implement two sequential phases; the orchestrator stops the rollout server before starting extraction and never co-resides two target replicas.
- [ ] Extend the no-model smoke through mock trajectory tokenization, mock hidden extraction, cache validation, resume, and mmap readback.
- [ ] Document full-scale commands, storage costs, workspace-map requirements, SGLang version sensitivity, and the mandatory real Ascend gate.

### Task 7: Verify

- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `bash scripts/smoke_no_model.sh`.
- [ ] Run `python -m compileall -q src tools`.
- [ ] Inspect mock stream sizes, offsets, checksums, manifests, and dataset tensors.
- [ ] Confirm the reference checkout, real model paths, source Parquet, and existing generated outputs were not modified.
- [ ] On Ascend, run 1--2 samples with the production multi-node topology and require exact IDs, logical-layer slices matching reference forward/hook values, finite BF16 `[T,5,6144]`, successful HCCL execution, commit/readback equality, and clean exit before scaling to 630K.
