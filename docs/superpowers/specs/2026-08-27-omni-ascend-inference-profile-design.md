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

The current vLLM-Ascend engine does not load Talker, MTP, or Code2Wav. Audio
Encoder, Vision Encoder, and Thinker may be loaded and executed but do not
expose stable request-scoped internal NPU events. The component report records
`loaded`, `executed`, `timing_available`, and `reason` separately. Unobservable
timings are not omitted, approximated with wall time, or reported as zero.

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
- batch end-to-end latency: first preprocessing start to batch return.

The profiler does **not** copy batch latency into every request. Each request
has its own preprocessing latency. Request engine latency is included only when
vLLM exposes valid arrival/finish timestamps; otherwise its availability is
false. Offline generation provides only a common batch return boundary, so no
request end-to-end latency distribution is invented.

The existing completion TPS remains engine-only for backward compatibility.
New names explicitly distinguish engine-only and end-to-end TPS. For either
the full run or one modality, throughput is always computed from sums, never
as the mean of batch TPS values:

```text
request_tps    = sum(request_count) / sum(relevant_batch_seconds)
completion_tps = sum(response_token_count) / sum(relevant_batch_seconds)
total_tps      = sum(prompt_token_count + response_token_count)
                 / sum(relevant_batch_seconds)
```

`engine_*` uses `sum(batch.engine_seconds)`. `end_to_end_*` always uses the
separately measured outer `sum(batch.end_to_end_seconds)`; the sum of the
preprocessing and engine sub-timers is diagnostic only and never serves as a
denominator because it excludes call-boundary and batch-assembly overhead.
Prompt and response counts come only from exact vLLM token IDs.

## Modality aggregation

Batches are already homogeneous by actual media payload. Each batch is tagged
with its computed kind (`text`, `image`, `multi_image`, `audio`, `video`, or
`other`). The profiler aggregates each kind independently using only that
kind's batch wall times. It never divides per-modality tokens by the global
wall time or sums concurrent request latency as elapsed time.

## HBM collection

The parent profiler process does not own vLLM model memory. HBM statistics are
therefore queried on all vLLM workers via `LLM.collective_rpc`:

- verify RPC returns exactly `tensor_parallel_size` ranks and that every rank
  identifies one unique physical NPU;
- synchronize every rank and reset allocator peaks immediately before each
  measured batch;
- synchronize and read current/peak allocated and reserved bytes immediately
  after that batch;
- retain each batch/rank post-batch current value, each rank's largest batch
  peak, `final_current` from the final measured batch,
  `max_post_batch_current`, the largest rank peak, and
  `max_batch_sum_of_rank_peaks`.

The report labels these values `torch_npu_allocator`; they are not device-wide
`npu-smi` usage and do not include unrelated processes. A sum of independently
observed run peaks is never called a simultaneous TP peak. If the pinned vLLM
version lacks worker RPC or torch-npu memory APIs, profiling fails explicitly
when HBM is required. `--allow-missing-hbm` is the only opt-out: any missing or
duplicate rank makes the entire memory section unavailable rather than
aggregating partial ranks.

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

The frozen scorer version is `omni_eval_v1`:

- `exact_match`: Python Unicode strings must be byte-for-byte equal after JSON
  decoding; no whitespace, case, or Unicode normalization is applied.
- `normalized_exact_match`: apply Unicode NFKC, `casefold()`, replace every
  Unicode punctuation category (`P*`) with one space, collapse all Unicode
  whitespace runs to one ASCII space, and strip leading/trailing whitespace.
- `multiple_choice_accuracy`: the reference is exactly one ASCII letter A-Z.
  Apply NFKC, uppercase, and outer-whitespace stripping to the prediction. It
  must fully match `^(?:ANSWER\s*[:=]\s*|OPTION\s+)?([A-Z])(?:[.)])?$`;
  otherwise it is incorrect. Natural-language letter search is forbidden.

Evaluation metadata survives normalization but is never passed into the model
request. The report contains overall and per-modality accuracy plus evaluated
and skipped counts. Mixing metric names in one run is rejected. A run with
some references is available and reports skipped rows; a modality with zero
evaluated rows is unavailable, not zero. Only a run with no references has an
entirely unavailable evaluation section.

This does not claim to recreate the deleted legacy `benchmark_helper`, which
was absent from the supplied archive. Dataset-specific answer extraction must
be expressed when building the evaluation JSONL rather than hidden inside the
performance profiler.

## Output contract

The top-level profile contains:

- runtime/model/engine/sampling identity;
- `performance.overall` and `performance.by_modality`;
- `memory` with availability, source, per-rank current/peak values and the
  precisely named `max_batch_sum_of_rank_peaks`; this is a sum of per-rank
  allocator peaks within one batch, not a simultaneously sampled TP HBM peak;
- `evaluation` with availability, metric, overall and per-modality values;
- `components` with availability and reasons for Audio Encoder, Vision
  Encoder, Thinker, Talker, MTP, and Code2Wav.

Every per-request JSONL record includes modality, exact token IDs, its own
preprocessing latency, optional vLLM engine request latency, enclosing batch
engine/end-to-end latency, and evaluation metadata/result when requested.

## Comparison identity

Each profile records:

- `comparison_identity.fingerprint`: declared target and processor revisions,
  vLLM/vLLM-Ascend/torch/torch-npu package identities, dtype and
  quantization, hardware and TP/EP topology, every non-speculative engine
  option (including model length, sequence and batched-token limits,
  scheduler, prefix cache, chunked prefill, and graph/eager mode), normalized
  request content and order, computed batch boundaries, sampling parameters,
  per-request seeds, warmup, and measurement rounds. Local media are bound by
  content digest and size, not path alone. Strict pairing rejects remote media
  unless it is first materialized and the actual fetched bytes are frozen and
  hashed. When evaluation is enabled, the invariant also binds scorer version,
  metric, and reference digest.
- `comparison_identity.strict_artifact_manifest_available=false`: this version
  does not hash the full local model/processor tree and therefore does not
  overclaim immutable artifact-level pairing.
- `variant_identity={"kind":"target_only"}`: this entry point does not load a
  drafter. A future speculative runtime must populate its own identity from the
  adapter and actual drafter artifact; a CLI label alone is forbidden.

This profiler produces one self-contained variant profile and does not compute
a paired speedup. A downstream baseline/speculative comparator may pair two
profiles only when their fingerprints match, and may claim strict artifact
pairing only after a future file-level artifact manifest is added. Strict
paired latency additionally requires exact prompt and response token IDs for
every request in every decoding mode; otherwise only independent throughput
distributions may be reported. Multi-round orchestration and speedup
statistics belong to that downstream comparison tool and are outside this
metric-completion change.

## Failure policy

The profiler fails before allocation on invalid input/configuration and fails
during inference on missing outputs, empty completions, mismatched counts,
mixed evaluation metrics, malformed references, or unavailable required HBM
telemetry. JSONL and profile data are written to temporary files and atomically
renamed only after every batch, worker RPC, and scorer succeeds; a success
marker binds their checksums. All three final paths are locked from preflight
through publication. Failed runs cannot leave outputs that look complete.
There is no backend fallback, native-model path, retokenization, or silent
metric omission.

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
