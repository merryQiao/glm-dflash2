# Qwen3-Omni Thinker data pipeline for Ascend 910B A2/A3

For server-side takeover, read [`AI_HANDOFF.md`](AI_HANDOFF.md) first. It
separates locally verified contracts from the A2/A3 gates that still require
the destination vLLM-Ascend image.

This directory contains the production-oriented, two-stage data path for a
Qwen3-Omni speculative drafter. It handles the **Thinker text stream only**;
Talker and Code2Wav are deliberately not loaded.

1. Stage A uses a vLLM-Ascend TP/EP engine to sample responses and stores the
   engine's exact prompt/response token IDs.
2. Stage B replays `prompt_token_ids + response_token_ids` as one teacher-forced
   prompt in vLLM's native `extract_hidden_states` mode. It never samples a new
   training target.
3. Every output shard is atomic and content-addressed. Resume rejects mixed
   model revisions, runtime settings, sampling policies, and corrupt files.

### Fidelity to the source Omni pipeline

The Ascend path retains the source pipeline's accepted-condition schema, Qwen chat
template and multimodal processor, sampling profile (`0.7/0.8/top-20`), EOS,
exact engine token IDs, and two-pass response-then-hidden workflow. The target
layers remain `[1, 12, 24, 36, 47]`. Three changes are intentional:

- CUDA/Transformers replicas are replaced by one vLLM-Ascend TP/EP engine;
- sampling seed is condition-local rather than batch-local, so changing batch
  composition cannot change the saved supervision target;
- the raw final decoder state is explicitly RMS-normalized with the immutable
  checkpoint weight before being labeled as lm-head input.

Contract tests pin these settings and fail if they drift.

## Runtime baseline

Use an official Qwen3-Omni-capable vLLM-Ascend image, currently:

```text
quay.io/ascend/vllm-ascend:v0.23.0
```

Do not reinstall `torch`, `torch-npu`, `vllm`, or `vllm-ascend` with pip inside
that image. Install only the auxiliary packages:

```bash
python -m pip install -r requirements-data.txt
```

The model is BF16. `quantization: ascend` is rejected by configuration
validation rather than silently changing the experiment.

## Required configuration edits

Edit `configs/generate_thinker_data.yaml` before any model is allocated:

- set immutable `model.revision` and `model.processor_revision` commits;
- set the local model and accepted-condition paths;
- set `runtime.hardware` to `a2` or `a3`;
- make `runtime.tensor_parallel_size` equal the number of chips exposed to the
  process;
- choose output and connector scratch paths on reliable storage.

The supplied hidden layer IDs are `[1, 12, 24, 36, 47]` for the 48-layer
Thinker. The extractor additionally requests synthetic layer ID `48`, which
vLLM defines as the raw post-decoder/pre-final-norm state. The pipeline reads
only `thinker.model.norm.weight` from the same immutable checkpoint and applies
Qwen3-Omni's official RMSNorm order offline. The cache therefore stores:

```text
target_hidden_states       [tokens, 5, 2048]
target_last_hidden_states  [tokens, 2048]
```

The second tensor has the explicit semantics
`post_final_norm_lm_head_input`; an unnormalised decoder output is never used as
a substitute.

## A2/A3 smoke gate

First prepare a small accepted-condition file containing at least one text,
image, audio, and video condition. Point a copy of the YAML at this file and
set its exact `expected_conditions` count. Then run on the destination host.

A2 example (four chips):

```bash
cd /path/to/omni-sd-ascend
ASCEND_HARDWARE=a2 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
CONFIG=$PWD/configs/smoke_a2.yaml \
bash scripts/smoke_ascend.sh
```

A3 example (two chips):

```bash
ASCEND_HARDWARE=a3 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 \
CONFIG=$PWD/configs/smoke_a3.yaml \
bash scripts/smoke_ascend.sh
```

`HCCL_OP_EXPANSION_MODE=AIV` is enabled only for A3. The smoke run must create
`ASCEND_SMOKE_ATTESTATION.json`; missing modalities, token mismatches, NaN/Inf,
or missing final-normalized semantics abort the run.

## Full production run

One Python process owns the full TP/EP engine. **Do not use torchrun.**

```bash
ASCEND_HARDWARE=a2 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
CONFIG=$PWD/configs/generate_thinker_data.yaml \
bash scripts/generate_thinker_trajectories_ascend.sh

ASCEND_HARDWARE=a2 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
CONFIG=$PWD/configs/generate_thinker_data.yaml \
bash scripts/extract_thinker_hidden_ascend.sh
```

For multiple independent engines, expose disjoint chip sets and assign shards
with `WORKER_ID=0..N-1` and `NUM_WORKERS=N`. Each process still owns its own
complete TP group.

## Thinker inference performance

`inference_qwen3-omni.py` profiles the same Thinker-only path used by Stage A.
It calls the same `load_engine`, `prepare_request`, and `sampling_kwargs`
provider, so it does not silently switch to greedy decoding or to a separate
Transformers implementation. Talker and Code2Wav are intentionally excluded.

Validate the exact engine and sampling plan without importing vLLM or
allocating a model:

```bash
python inference_qwen3-omni.py \
  --config configs/generate_thinker_data.yaml \
  --text "Describe speculative decoding." \
  --dry-run \
  --profile-json outputs/thinker_dry_run.json
```

Run a real A2 profile. Each warmup round uses the first real batch shape of
every actual modality present, is excluded from every throughput/latency
aggregate, and is followed by public `LLM.reset_mm_cache()` so measured media
cannot hit vLLM's warmup MM cache. HBM telemetry is required by default and is
collected from every vLLM TP worker through `LLM.collective_rpc` and the
rank-local `torch_npu` allocator:

```bash
ASCEND_HARDWARE=a2 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
CONFIG=$PWD/configs/generate_thinker_data.yaml \
bash scripts/profile_thinker_ascend.sh \
  --conditions-parquet /path/to/accepted_conditions.parquet \
  --limit 128 \
  --warmup 1 \
  --output-jsonl outputs/thinker_profile.jsonl \
  --profile-json outputs/thinker_profile.json
```

For a short functional run, use `--text`; for representative throughput, use
accepted-condition Parquet and retain the configured modality-specific batch
sizes. `--batch-size N` intentionally overrides every modality batch size and
is recorded only by the performance invocation, not the training manifest.
The profile separates host preprocessing, the enclosing `LLM.generate` call,
and outer end-to-end batch time. It reports engine-only and end-to-end
requests/s, completion-token TPS, and total-token TPS; per-request
preprocessing latency; vLLM request latency only when vLLM exposes valid
arrival/finish timestamps; and batch engine/end-to-end distributions. The
same report is repeated under `performance.by_modality` using modality-local
time denominators. Exact prompt and response token IDs are written for every
measured request.

Every successful real run publishes three files: the JSONL, the profile JSON,
and `<profile>.SUCCESS.json`. The success marker is written last and binds the
other two files by SHA-256. All three final paths are locked for the entire
preflight/inference/publish window so overlapping invocations cannot mix
artifacts. Treat files without this marker as incomplete.

The `memory` section contains per-rank current and peak allocated/reserved HBM,
plus precisely named TP reductions. These are `torch_npu_allocator` values,
not device-wide `npu-smi` usage. If the pinned vLLM build cannot execute worker
RPC, the profiler fails. `--allow-missing-hbm` is an explicit diagnostic-only
opt-out and makes the entire memory section unavailable; partial-rank numbers
are never reported.

Benchmark scoring is opt-in for JSONL input. Add one frozen scorer contract per
row:

```json
{
  "id": "example",
  "text": "Answer with one letter.",
  "evaluation": {
    "metric": "multiple_choice_accuracy",
    "reference": "B"
  }
}
```

Supported `omni_eval_v1` metrics are `exact_match`,
`normalized_exact_match`, and `multiple_choice_accuracy`. A run cannot mix
metric names. The report gives evaluated/skipped counts and accuracy overall
and by modality; rows without references are skipped rather than scored as
wrong.

`components` distinguishes `loaded`, `executed`, and `timing_available`.
Audio/Vision Encoder and Thinker internal device-event times are not exposed by
the current vLLM-Ascend route. Talker, MTP/code predictor, and Code2Wav are not
loaded. Their timings are explicitly unavailable, never approximated or
reported as zero. The profile remains the same Thinker-only vLLM path that a
future drafter adapter must accelerate; no native/Transformers fallback is
used.

## Hard runtime gate

CPU contract tests pass without Ascend hardware, but that does **not** prove
the destination image supports the full extractor path. Before a production
run, the A2/A3 smoke must establish that its pinned vLLM-Ascend build supports:

- `Qwen3OmniMoeThinkerForConditionalGeneration` in core vLLM;
- `method=extract_hidden_states` for this multimodal target;
- `ExampleHiddenStatesConnector` on NPU (including its async host-copy path);
- custom save paths and `[tokens, layers, hidden]` connector layout.

Upstream vLLM documents that chunked prefill is incompatible with hidden-state
extraction, so both engines force it off. If the NPU connector fails, stop at
the smoke gate and patch that version's connector; do not fall back silently to
retokenized responses or mislabeled final hidden states.

## Local verification

```bash
python -m pytest -q
python -m py_compile inference_qwen3-omni.py $(find src scripts/data -name '*.py' -type f)
for script in scripts/*.sh; do bash -n "$script"; done
```
