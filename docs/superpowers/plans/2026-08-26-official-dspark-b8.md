# Official DSpark B8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the GLM-5.2 DSpark consumer with the official DeepSpec Markov/confidence objective and expose the requested DFlash/DFlash2 B8+B16 and DSpark-only B8 recipes.

**Architecture:** Preserve the common five-layer GLM backbone and schema-v2 cache. Replace only the DSpark method heads with the official vanilla predecessor-token Markov bias and Markov-aware confidence predictor; keep the exact full-vocabulary L1/acceptance target already implemented. Make physical block length a run-level setting while validating the method matrix explicitly.

**Tech Stack:** Python, PyTorch/torch-npu, Transformers, unittest, Bash.

---

### Task 1: Lock official DSpark head behavior with tests

**Files:**
- Modify: `tests/test_dspark_model.py`
- Modify: `tests/test_dspark_objective.py`
- Modify: `tests/test_dspark_trainer.py`

- [ ] Add a failing test showing Markov logits depend only on predecessor token, not hidden state.
- [ ] Add a failing test showing confidence receives `concat(hidden, markov_embedding)`.
- [ ] Add/retain dense-reference checks for `0.1 CE + 0.9 L1 + 1.0 BCE`, detached `1 - 0.5 L1` target, and shared depth weights.
- [ ] Run the focused tests and confirm the new head tests fail for the current hidden-gated implementation.

### Task 2: Implement official DSpark heads

**Files:**
- Modify: `src/glm_dflash2/dspark_model.py`
- Modify: `src/glm_dflash2/offline_trainer.py`

- [ ] Replace the hidden-gated head with `Embedding(V, r) -> Linear(r, V)` vanilla Markov bias.
- [ ] Build confidence logits from draft hidden concatenated with the same predecessor embedding.
- [ ] Thread teacher-forced predecessor IDs into both Markov and confidence paths.
- [ ] Run focused DSpark model/objective/trainer tests.

### Task 3: Enforce the experiment matrix and GLM DSpark recipe

**Files:**
- Modify: `tools/train_drafter_offline.py`
- Modify: `scripts/train_glm52_drafter_910b.sh`
- Modify: `tests/test_unified_train_cli.py`
- Modify: `tests/test_training_launchers.py`

- [ ] Add failing parser/validation tests: DFlash and DFlash2 accept physical B8/B16; DSpark accepts only physical B8.
- [ ] Make `build_glm52_dflash2_config` consume the selected physical block length.
- [ ] Set DSpark launcher defaults to one epoch, `lr=3e-4`, gamma 4; retain existing DFlash/DFlash2 defaults unless explicitly overridden.
- [ ] Record physical block length and seven DSpark proposal positions in semantic run metadata.
- [ ] Run CLI and launcher tests.

### Task 4: Document and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/ASCEND_910B_RUNBOOK.md`
- Modify: `docs/superpowers/specs/2026-08-26-glm52-unified-dflash-cache-design.md`

- [ ] Document `B8 = one anchor + seven predicted tokens` and the allowed method matrix.
- [ ] Document the official confidence soft target and method-specific defaults.
- [ ] Run all unit tests.
- [ ] Run the CPU three-method smoke and confirm finite loss/backward/optimizer updates.
