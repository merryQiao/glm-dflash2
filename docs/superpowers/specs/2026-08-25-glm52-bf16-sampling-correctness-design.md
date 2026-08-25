# GLM-5.2 BF16 Sampling DFlash2 Correctness Design

## Goal

Make the repository's production path internally consistent for a GLM-5.2 BF16
target whose online policy samples with `temperature=1.0` and `top_p=0.95`, while
preserving the route semantics of the reference `vibe_coding_qwen38.py` pipeline.

## Data and model contract

- Stage A, Stage B, and target I/O use the same immutable GLM-5.2 BF16 revision.
- Stage A defaults to the official GLM-5.2 sampling policy: temperature 1.0,
  top-p 0.95, and no explicit top-k truncation.
- Open-SWE prefix rows continue to restore their original complete trajectories;
  all other routes continue to use real workspace/tool rollouts.
- Stage B teacher-forces the immutable sampled token path and captures logical
  layers `1,20,38,56,75`.
- Training consumes only the frozen cache and frozen target embedding/LM head.

## Correctness changes

1. Reject duplicate Stage-A output IDs and make the output lock exception-safe.
2. Support the GLM SGLang capture hook already exposed as
   `set_eagle3_layers_to_capture`, while retaining the DFlash-specific alias.
3. Keep frozen hidden caches read-only during training and move checksum work out
   of the hot per-sample path.
4. Never sample an anchor whose predicted suffix contains zero supervised tokens.
5. Normalize base and selector losses over the global token mass across ranks.
6. Make checkpoint completion metadata durable before writing `COMPLETE`.
7. Export the official `DFlash2DraftModel` architecture identifier and keep
   serving compatibility behind an explicit parity gate.
8. Remove the historical vLLM response-only path and unused anchor helpers.

## Non-goals

- This change does not implement the Ascend SGLang DFlash2 serving kernels.
- It does not alter valid message/tool trajectories produced by the reference
  routing policy.
- It does not support quantized target I/O tensors.

## Verification

- Every behavior change receives a regression test before implementation.
- Run all repository unit tests under `transformers==4.57.3`.
- Run the no-NPU tiny training smoke.
- Real 910B validation remains the two-sample hidden-capture gate followed by the
  two-rank training/resume gate and runtime parity comparison.
