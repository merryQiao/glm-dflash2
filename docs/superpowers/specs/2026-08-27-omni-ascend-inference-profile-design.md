# Omni Ascend Inference Profile Design

## Goal

Restore a runnable `inference_qwen3-omni.py` entry point without reintroducing
the CUDA-only, incomplete legacy profiler. The new entry point must measure the
same Qwen3-Omni Thinker path used by Stage A on Ascend 910B A2/A3.

## Scope

- Thinker text generation only; Talker and Code2Wav are not loaded.
- Reuse the validated YAML, `load_engine`, `prepare_request`, and
  `sampling_kwargs` functions from the production trajectory generator.
- Accept a command-line text sample, a simple/native JSONL file, or accepted
  condition Parquet.
- Preserve exact prompt and response token IDs in JSONL output.
- Report model load time, warmup time, measured wall time, requests/s,
  completion tokens/s, total tokens/s, and latency percentiles.
- Support a dependency-light `--dry-run` that validates input/config and emits
  the exact engine/sampling plan without importing vLLM.

## Data flow

Input records are normalized into the same condition-row contract consumed by
Stage A. Request construction and sampling are delegated to production code.
Warmup requests run before measurement and are excluded from all aggregate
metrics. Measured requests are split into configured batches; every request
must return one non-empty completion with exact engine token IDs.

## Failure policy

The script fails before allocation on invalid configuration, missing media, an
empty input set, or unsafe overwrite. During inference it fails on missing or
empty vLLM outputs. It does not silently switch backend, dtype, checkpoint,
sampling policy, or target component.

## Verification

CPU contract tests cover input normalization, production-route reuse, dry-run,
metric arithmetic, warmup exclusion, and lazy vLLM imports. Actual throughput
claims require running the script in the pinned vLLM-Ascend image on the target
A2/A3 host.
