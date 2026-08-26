# GLM-5.2 DFlash2 Two-Pass SGLang Data Design

## Goal

Build a GitHub-ready, resumable data pipeline for all 630,000 rows of
`cbyzju/vibe_coding_630k`. The target is GLM-5.2 on Ascend 910B. One command
runs two sequential SGLang phases: real coding-agent trajectory rollout, then
teacher-forced selected-layer hidden extraction. Model weights and generated
multi-terabyte artifacts remain external to Git.

## Phase A: agentic trajectory rollout

The implementation is copied from and kept behaviorally aligned with
`SpecForge/scripts/vibe_coding_qwen38.py`, including its heterogeneous dataset
routing, workspace materialization, bounded repository tools, optional web
tools, multi-round tool execution, validation, and ID-based resume.

The GLM version launches an OpenAI-compatible SGLang server or reuses an
existing endpoint. It stores the complete canonical message trajectory,
ordered tool schemas, tool events, generation-start message index, exact
chat-template kwargs, structured reasoning/tool-call fields, per-round
prompt/output token IDs required from SGLang for every new rollout,
model/sampling metadata,
source metadata, and validation result. Output is
append-only JSONL with a single-writer lock and a manifest. A trajectory is
committed only after it ends with a resolved assistant response and passes
message/tool validation.

After rollout, the same target tokenizer renders the complete final message
list with the frozen ordered tools and identical chat-template kwargs. Prefix
stability is checked around every generated assistant turn, and server
prompt/response token IDs are mandatory per-round golden checks for new
rollouts. Missing or mismatched IDs reject the sample instead of falling back
to detokenize/re-tokenize. Restored original Open-SWE trajectories are recorded
as a separate teacher-forced source route because their historical server IDs
are unavailable.
The stored `input_ids` are the exact rendered final trajectory and
`loss_mask` is token-aligned with SpecForge DFlash: tokens belonging to
generated assistant turns are 1; system, source user, later user, and tool
observation tokens are 0; the final token is forced to 0. This is intentionally
not an AR predictor-position mask.

## Phase B: selected-layer hidden extraction

The SGLang rollout server is stopped before hidden extraction, so only one
GLM-5.2 replica is resident. The extractor reads committed trajectories and
submits their saved `input_ids` as teacher-forced prefills. It never decodes
response text or re-tokenizes it.

For the 78-layer GLM-5.2 model, the default target layers are
`[1, 20, 38, 56, 75]`. SGLang's internal model runner is used because the
public HTTP hidden-state API exposes the final layer but not an arbitrary set
of intermediate layers. GLM-5.2's SGLang implementation inherits the
DeepSeek-V2 auxiliary hidden capture route and supports explicit DFlash layer
capture. The runner requests `CaptureHiddenMode.FULL`, receives packed
`[T, 5 * 6144]` auxiliary states and validates their configured layer order.
The manifest records logical layer IDs, the SGLang model's physical capture
points (including its model-specific `+1` convention), and the exact SGLang
version/commit. The hardware gate compares each captured slice against a
reference forward/hook on a short input; shape and finite checks alone are not
accepted as proof of correct layer identity.

The backend is lazy-imported and isolated behind a small protocol. CPU tests
use a deterministic mock backend. The real backend follows SGLang's
`benchmark.one_batch` standalone initialization path, which is device-aware
and supports NPU/HCCL, rather than importing the CUDA-specific SpecForge
training wrapper. SGLang internal APIs are version-sensitive, so the deployed
SGLang commit/version is recorded in the manifest and a real 1--2 sample
Ascend acceptance gate is mandatory before a full run.

## Packed cache

Each data shard contains one or more physical append-only segments:

- `input_ids.bin`: little-endian signed 64-bit token IDs;
- `loss_mask.bin`: one byte per token;
- `hidden_states.bin`: raw BF16 words;
- `index.jsonl`: sample ID, source index, offsets, shapes, checksums;
- `manifest.json`: schema, fingerprints, layer order, topology, segment and
  completion state.

A sample commit is ordered as follows: append all three streams; flush and
`fsync` them; append one complete index line; `fsync` the index; atomically
replace the manifest. The index line is the commit point. Resume removes only
unindexed stream tails and a partial final index line. If the index is ahead
of the manifest, resume rebuilds the manifest from committed index entries and
does not repeat samples. Segment rollover uses the same rule: only an fsynced
index entry commits bytes, while segment-list publication may lag and is
reconstructed. A committed index that
extends past a stream or fails checksum validation is a hard error.

Phase A follows the same durability model: write one complete JSONL line,
flush/fsync it, then publish manifest progress. Resume repairs only a partial
last line and rebuilds manifest counts from committed unique IDs. A frozen
trajectory manifest includes JSONL size/digest, the unique expected ID set
digest and zero unresolved errors; Phase B refuses any non-frozen input.

The manifest binds the source dataset revision/content, committed trajectory
manifest/content, target model and tokenizer fingerprints, SGLang version,
layer IDs/order, hidden size, dtype, data-shard ownership, and expected,
completed, skipped, and error counts. A cache is complete only when every
expected owned trajectory is present and no unresolved error remains.

## Training reader

`PackedHiddenDataset` memory-maps only requested sample slices. Its archival
view returns:

- `input_ids`: `torch.int64[T]`;
- `loss_mask`: `torch.bool[T]` with DFlash token-position semantics;
- `hidden_states`: `torch.bfloat16[T, 5, 6144]`.

Its DFlash training adapter flattens only the final two dimensions and returns
`hidden_states: torch.bfloat16[T, 30720]`, matching the real target projector.
The collator right-pads variable lengths, pads `loss_mask` with false, and
tests the real SpecForge strategy/projector boundary without shifting masks.

It refuses incomplete caches and incompatible fingerprints. All streams are
little-endian. BF16 is stored as raw 16-bit words to avoid dependence on NumPy
BF16 support.

## Scale and deployment gates

Five BF16 layers at hidden size 6144 cost 61,440 bytes per token. At an average
of 1,000 tokens, 630K samples require about 38.7 TB before filesystem overhead;
2,000 tokens require about 77.4 TB. Capacity must be checked before launch.

The first implementation disables implicit SGLang chunked prefill for hidden
capture. A configured `max_sequence_tokens` is a hard, recorded eligibility
bound rather than silent truncation. Such a failure keeps Phase A incomplete;
the full run must choose a bound/topology large enough for every trajectory or
add a separately validated explicit KV-preserving extend implementation.

Only TP plus optional expert parallelism is supported by the first standalone
extractor; PP and DP are rejected. Global/local ranks, node count/rank and the
single global-rank-0 writer are explicit. Model ranks are independent of data
shards, which require separate full model replicas.

Local tests cover parsing, trajectory masking, locking, sharding, interrupted
writes, resume, mock hidden extraction, mmap reads, and CLI validation. The
actual Ascend gate must verify SGLang/torch-npu/CANN compatibility, GLM-5.2
logical-to-physical layer identity, exact token equality, finite BF16
`[T, 5, 6144]` output, packed
commit/readback, and multi-rank HCCL execution. The current host cannot claim
that hardware gate.
