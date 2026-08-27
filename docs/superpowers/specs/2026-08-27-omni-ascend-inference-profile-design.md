# Omni vLLM-Ascend Inference Profile Design

## Goal

Extend `omni-sd-ascend/inference_qwen3-omni.py` with every metric that can be
measured faithfully on the same vLLM-Ascend Thinker path that will later host
the speculative drafter. The profiler must never switch to a native
Transformers model or report unavailable full-Omni stages as zero.

## Scope and non-goals

The measured system is the vLLM-Ascend Qwen3-Omni Thinker. Baseline and future
speculative runs therefore share the model, engine, processor, hardware,
sampling policy, batches, and timing boundaries.

This change adds:

- preprocessing and engine-generation wall time, separately and end to end;
- per-request and per-batch latency distributions;
- request, completion-token, and total-token throughput;
- per-modality copies of the same counts, throughput, and latency metrics;
- per-rank NPU allocator current/peak allocated and reserved HBM, collected
  through vLLM worker RPC rather than from the profiler process;
- opt-in benchmark evaluation from references carried in JSONL input;
- an explicit component-availability section.

The current vLLM-Ascend engine does not load Talker, MTP, or Code2Wav and does
not expose stable request-scoped Audio Encoder, Vision Encoder, or internal
Thinker NPU events. Those fields are emitted as `unavailable` with a reason.
They are not omitted, approximated with wall time, or reported as zero.

## Measurement boundaries

For each measured batch:

1. Start the end-to-end wall clock.
2. Time every `prepare_request` call and retain one preprocessing latency per
   request.
3. Start the engine wall clock immediately before `LLM.generate` and stop it
   after all outputs return.
4. Stop the end-to-end wall clock.
5. Read exact prompt/response token IDs from vLLM outputs.
6. Read worker allocator statistics through `LLM.collective_rpc`.

Warmup performs the same work but is excluded from every measured aggregate.
The report keeps these clocks distinct:

- `preprocess_seconds`: host-side chat rendering and media decoding;
- `engine_seconds`: the enclosing vLLM generation calls;
- `end_to_end_seconds`: preprocessing plus engine execution;
- vLLM request latency: engine-provided arrival-to-finish time when available;
- request end-to-end latency: the enclosing batch end-to-end time, because all
  requests in an offline batch complete at the batch return boundary.

The existing completion TPS remains engine-only for backward compatibility.
New names explicitly distinguish engine-only and end-to-end TPS.

## Modality aggregation

Batches are already homogeneous by actual media payload. Each batch is tagged
with its computed kind (`text`, `image`, `multi_image`, `audio`, `video`, or
`other`). The profiler aggregates each kind independently using only that
kind's batch wall times. It never divides per-modality tokens by the global
wall time or sums concurrent request latency as elapsed time.

## HBM collection

The parent profiler process does not own vLLM model memory. HBM statistics are
therefore queried on all vLLM workers via `LLM.collective_rpc`:

- reset each worker's NPU allocator peak immediately after warmup;
- read current and peak allocated/reserved bytes after each measured batch;
- retain per-rank maxima;
- report both the maximum rank and sum across ranks.

The report labels these values `torch_npu_allocator`; they are not device-wide
`npu-smi` usage and do not include unrelated processes. If the pinned vLLM
version lacks worker RPC or torch-npu memory APIs, profiling fails explicitly
when HBM is required. `--allow-missing-hbm` is the only opt-out and records the
reason in the report.

## Benchmark evaluation

Benchmark evaluation is opt-in and framework-independent. A JSONL row may
carry:

```json
{
  "evaluation": {
    "metric": "normalized_exact_match",
    "reference": "expected answer"
  }
}
```

Supported initial metrics are `exact_match`, `normalized_exact_match`, and
`multiple_choice_accuracy`. Evaluation metadata survives normalization but is
never passed into the model request. The report contains overall and
per-modality accuracy plus evaluated/skipped counts. Mixing different metric
names in one run is rejected. Inputs without evaluation metadata remain valid
and produce an explicit unavailable evaluation section.

This does not claim to recreate the deleted legacy `benchmark_helper`, which
was absent from the supplied archive. Dataset-specific answer extraction must
be expressed when building the evaluation JSONL rather than hidden inside the
performance profiler.

## Output contract

The top-level profile contains:

- runtime/model/engine/sampling identity;
- `performance.overall` and `performance.by_modality`;
- `memory` with availability, source, per-rank peaks, max-rank and TP-sum;
- `evaluation` with availability, metric, overall and per-modality values;
- `components` with availability and reasons for Audio Encoder, Vision
  Encoder, Thinker, Talker, MTP, and Code2Wav.

Every per-request JSONL record includes modality, exact token IDs,
preprocessing, engine, and end-to-end latency plus evaluation metadata/result
when requested.

## Failure policy

The profiler fails before allocation on invalid input/configuration and fails
during inference on missing outputs, empty completions, mismatched counts,
mixed evaluation metrics, malformed references, or unavailable required HBM
telemetry. There is no backend fallback, native-model path, retokenization, or
silent metric omission.

## Verification

CPU tests use a fake vLLM engine and fake worker RPC to prove:

- preprocessing and engine clocks are not conflated;
- warmup is excluded;
- overall and per-modality arithmetic is correct;
- exact IDs are preserved;
- HBM peaks are reduced correctly across TP ranks;
- evaluation parsing and scoring are deterministic;
- unavailable components are explicit;
- production request construction and sampling providers remain unchanged.

Actual performance claims still require the pinned A2/A3 vLLM-Ascend image.
The destination smoke must verify worker RPC, torch-npu allocator telemetry,
and every requested modality before reporting numbers.
