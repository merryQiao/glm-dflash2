# GLM-5.3 drafter audit fixes design

## Goal

Correct the confirmed GLM-5.3 production-contract and training-quality defects without changing the approved DSpark objective `0.1 CE + 0.9 TV + 1.0 confidence` or silently changing the Stage A data mixture.

## Design

1. **Official checkpoint contract.** Accept the official root `model_type=glm5_next`, nested `model_type=glm5_next_text`, and nested `dtype=bfloat16`. Tests use a fixture shaped like the published GLM-5.3 config.
2. **Mask identity.** Resolve `[MASK]` from the target tokenizer artifact, require ID `154821` for the production contract, and bind the literal token, ID, tokenizer fingerprint, and canonical target-I/O artifact identity into a versioned training contract. A caller-supplied mismatch and a legacy mask-less artifact fail before model construction.
3. **DFlash2 supervision.** Build a training candidate set that always contains the target: retain base top-k on hits, otherwise replace the weakest candidate. Compute unary top-k recall before injection, preserve the existing keep/depth weights and denominator, and fail if any positively weighted target remains absent. Keep inference top-k untouched.
4. **Initialization.** Apply one centralized Qwen/Transformers-style normal initialization (`std=initializer_range`, default `0.02`) to trainable linear/embedding weights and preserve modules that intentionally require identity/zero initialization. Initialize in FP32 before BF16/FSDP wrapping.
5. **Data transparency and coverage.** Preserve original-trajectory and online routes, persist one canonical route per committed row, and recompute route counts/fractions from committed JSONL so resume/partial failures cannot drift. Replace one-window-per-long-row training with deterministic cycling over eligible windows/anchor buckets, with resume-stable cycle position in the semantic config.
6. **Observability.** Log idempotent optimizer-step JSONL records containing LR, total and component losses, token denominators, DFlash2 unary recall, and DSpark confidence diagnostics. Add a deterministic, disjoint held-out split with fixed validation windows, route-aware counts, all-rank unbiased reduction, `no_grad`, and restored train/eval mode.
7. **Ascend boundary.** Keep the existing strict fail-closed SGLang hidden parity unchanged. Real 910B A2 compatibility remains a runtime smoke gate rather than an emulated success claim.

## Non-goals

- Do not change DSpark TV scaling.
- Do not remove original Open-SWE trajectories.
- Do not invent a GLM-5.3 speculative runtime or claim vLLM-Ascend deployment support.
- Do not relax hidden-state parity checks.

## Success criteria

- Published GLM-5.3 config validates; nearby invalid configs fail.
- Wrong/missing mask identity fails before training.
- Every supervised DFlash2 selector position has a target-bearing candidate set.
- Initialization statistics match the configured standard deviation.
- Long rows select different deterministic windows across epochs.
- Stage A and training emit auditable route/coverage/metric records.
- Full CPU test suite passes; no DSpark TV test changes.
- A characterization test locks `total = 0.1 CE + 0.9 TV + 1.0 confidence`, including denominator semantics.
