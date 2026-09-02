# GLM-5.3 W8A8: Stage B / Stage C

This is an isolated pipeline for the formal GLM-5.3 ModelSlim **W8A8** target
on Ascend 910B A2. It does not modify or import the adjacent GLM-5.2,
GLM-5.3-Flash, Omni, or legacy DFlash code.

```text
Stage A (external rollout producer)
  -> exact token-ID trajectory JSONL + manifest
Stage B (vLLM-Ascend)
  -> five intermediate hidden streams + normalized final hidden stream
Stage C (torch-npu / BF16)
  -> DSpark B8 or DFlash2 B8/B16 drafter checkpoints
```

The implementation is deliberately split at the Stage-A/Stage-B boundary. The
target is loaded only for Stage B and for the one-time embedding/LM-head export;
Stage C never loads the 78-layer GLM transformer.

All persisted artifacts use versioned v2 schemas (`target-io-v2`,
`hidden-cache-v2`, and `stage-c-checkpoint-v2`).  The loaders reject v1 artifacts
because they do not carry the provenance and alignment fields needed to make a
W8A8 run reproducible.

## Fixed contract

Target-side validation is fail-closed:

| Field | Value |
|---|---|
| architecture | `GlmMoeDsaForCausalLM` |
| model type | `glm_moe_dsa` |
| base decoder depth | 78 |
| hidden / intermediate | 6144 / 12288 |
| target Q/KV heads | 64 / 64 |
| target head dimension | 192 |
| vocabulary | 154880 |
| embedding/head | untied, dense BF16, no head bias |
| quantization marker | ModelSlim W8A8 (`--quantization ascend`) |

For a real ModelSlim export, `quant_model_description.json` is the authoritative
quantization ABI and must be present beside `config.json`; the extractor scans
its nested per-tensor entries for W8A8/W8A8_DYNAMIC metadata. A compact
`config.json` marker is accepted only by the tiny local smoke fixtures.

The five logical target taps are `[1, 20, 38, 56, 75]`. vLLM's extraction
configuration additionally requests layer `78`; vLLM documents this as the raw
output of the last decoder block. That raw stream is normalized with the real
checkpoint final RMSNorm before it is used as the DSpark teacher hidden.

The drafter is a five-layer dense BF16 network:

```text
hidden=6144, intermediate=12288, Q/KV heads=64/64, head_dim=64
```

Both methods use a **physical 2048-token causal sliding window** in every
draft layer. For an anchor at absolute position `a`, the context is exactly
`[max(0, a - 2048), a)`. It never includes the anchor or a future trajectory
token. The local block contains one real anchor followed by mask tokens, and
all positions in that local block are mutually visible. Absolute position IDs
are retained when a long trajectory is cropped.

## Stage A input contract

Stage A is intentionally not reimplemented here: it may be the existing
SGLang rollout driver or another trusted GLM-5.3 W8A8 rollout producer. It must
write one JSON object per line and a sidecar named
`<trajectory>.manifest.json`.

Each row must contain either:

```json
{"sample_id":"stable-id", "input_ids":[...], "loss_mask":[false,true,...]}
```

or exact server token streams:

```json
{"sample_id":"stable-id", "prompt_token_ids":[...], "response_token_ids":[...]}
```

The second form is flattened without retokenization. The sidecar must include:

```json
{
  "trajectory_sha256": "...",
  "source_model_fingerprint": "...",
  "tokenizer_fingerprint": "...",
  "target_quantization": "W8A8"
}
```

Stage B refuses a missing or mismatching checksum, tokenizer/model identity, or
non-W8A8 provenance. Sampling parameters belong to Stage A and are not silently
changed by Stage B; Stage B is a deterministic teacher-forced replay of the
stored token IDs.

## Stage B: vLLM-Ascend hidden extraction

Run this in the vendor vLLM-Ascend + ModelSlim W8A8 environment. The target
must be the exact same W8A8 checkpoint used for Stage A.

First export the frozen dense target input/output tensors:

```bash
cd /path/to/glm-dflash2/glm-5.3-w8

MODEL_PATH=/models/GLM-5.3-W8A8 \
OUTPUT_DIR=/data/glm53_target_io \
PY=/path/to/vendor/python \
bash scripts/extract_target_io.sh
```

Then replay trajectories through vLLM's native
`method=extract_hidden_states` path:

```bash
MODEL_PATH=/models/GLM-5.3-W8A8 \
TRAJECTORY_JSONL=/data/stage_a/trajectories.jsonl \
TARGET_IO_DIR=/data/glm53_target_io \
OUTPUT_DIR=/data/glm53_hidden_cache \
SCRATCH_ROOT=/dev/shm/glm53_hidden_connector \
TP_SIZE=8 \
BATCH_SIZE=1 \
MAX_MODEL_LEN=131072 \
PY=/path/to/vendor/python \
bash scripts/run_stage_b_hidden.sh
```

The extractor sets:

```text
quantization=ascend
speculative_config.method=extract_hidden_states
eagle_aux_hidden_state_layer_ids=[1,20,38,56,75,78]
enable_chunked_prefill=False
enable_prefix_caching=False
```

`ExampleHiddenStatesConnector` writes a safetensors file per replay request.
The code verifies connector token IDs byte-for-byte against Stage A, verifies
all six streams are finite BF16 with shape `[T, 5, 6144]` plus `[T, 6144]`,
normalizes raw layer 78, and removes the temporary connector file only after a
successful cache commit. A bounded `MAX_SAMPLES` run leaves a resumable
`status=building` cache; omit it for a production freeze.

The prefix-cache switch is intentional: vLLM has had versions where prefix
reuse made the hidden stream shorter than `token_ids`. The chunked-prefill
switch is also required by vLLM's hidden extraction contract. Confirm the
vendor vLLM-Ascend build supports these options before a long run.

## Stage C: offline BF16 drafter training

Stage C loads only the frozen target I/O artifact and the frozen hidden cache.
For production, use `torch-npu`, HCCL, and FSDP2; use one process per NPU.

### DSpark B8

```bash
METHOD=dspark \
BLOCK_SIZE=8 \
HIDDEN_CACHE=/data/glm53_hidden_cache \
TARGET_IO_DIR=/data/glm53_target_io \
OUTPUT_DIR=/data/checkpoints/glm53-dspark-b8 \
MASK_TOKEN_ID=<GLM-5.3-mask-token-id> \
NPROC_PER_NODE=8 \
PY=/path/to/torch-npu/python \
bash scripts/train_drafter.sh
```

The fixed DSpark recipe is three epochs, LR `6e-4`, gamma `4`, rank-256
Markov residual, and
`0.1 * CE + 0.9 * exact-TV + 1.0 * confidence-BCE`. The confidence target is
`clamp(1 - TV, 0, 1)`. Query zero is the clean anchor; the teacher-forced
predecessor for each later query is the preceding ground-truth token.  B8 means
one anchor plus seven successor positions (`num_speculative_tokens=7`), and
`sample_from_anchor=false` is recorded in the semantic configuration.

### DFlash2 B8 / B16

```bash
METHOD=dflash2 \
BLOCK_SIZE=8 \
HIDDEN_CACHE=/data/glm53_hidden_cache \
TARGET_IO_DIR=/data/glm53_target_io \
OUTPUT_DIR=/data/checkpoints/glm53-dflash2-b8 \
MASK_TOKEN_ID=<GLM-5.3-mask-token-id> \
NPROC_PER_NODE=8 \
PY=/path/to/torch-npu/python \
bash scripts/train_drafter.sh
```

Use `BLOCK_SIZE=16` and a different `OUTPUT_DIR` for the B16 run. DFlash2 uses
the two-tap grouped dynamic causal convolution around attention and MLP plus a
rank-256, top-16 candidate selector. Its objective is the official DFlash2
form: decay-weighted full-vocabulary base CE **plus independently normalized**
decay-weighted selector CE.  Selector CE is applied only when the ground-truth
successor is already in the base top-16 list; a miss has no selectable target
and is excluded from this term.  The hit ratio is reported as candidate recall,
and the selector always consumes the same candidate list used at runtime.
Gamma is `4` for B8 and `7` for B16; LR is `6e-4`, three epochs.

The launcher defaults are 512 deterministic anchors/sample, anchor chunk size
8, per-rank batch 1, gradient accumulation 8, 4096-token supervised cores,
1000-step warmup, cosine decay, and checkpoint every 1000 optimizer steps.
If HBM is tight, set `GRADIENT_CHECKPOINTING=1` in the launcher; this only
recomputes the five draft layers during backward and does not alter the loss
or token/window contract.
Override only with explicit experiment metadata. Checkpoints are written at
optimizer boundaries and include model, optimizer, scheduler, progress, and
semantic configuration.  FSDP2 accumulation uses public gradient-sync and
reshard-after-backward controls; the final partial accumulation window is
normalized by its real (non-padding) rank count.  Resume requires the same
world size and all semantic fields.

For a long trajectory, a 4096-token supervised core is read together with a
left 2048-token physical halo and up to `block_size - 1` right-tail tokens.
Only the core can supply anchors; this prevents silently training on a cropped
history while preserving successors for anchors near the core boundary.

## Storage and resource notes

The six BF16 hidden vectors cost
`(5 * 6144 + 6144) * 2 = 73,728` bytes/token, before IDs, masks, metadata and
safetensors overhead. A 128K-token trajectory therefore needs about 9.0 GiB
of hidden values. The writer shards by byte size and updates a crash-safe
manifest only after a shard checksum is durable. Storage can be HDD; the
connector scratch directory should be a local fast filesystem such as `/dev/shm`
when available.

Stage C's full-vocabulary projection is intentionally chunked (`8192` by
default) to avoid materializing `[anchors, block, vocab]` logits. It still
performs the mathematically exact vocabulary normalization. A production run
should choose the chunk size from NPU HBM measurements, not silently switch to
sampled or top-k-only CE.

## Verification

The local suite is CPU-only and validates contracts, alignment, finite losses,
physical windows, cache resumability, and launcher syntax. It does **not** prove
that a particular vendor vLLM-Ascend build supports GLM-5.3 W8A8; that remains a
real-hardware gate.

```bash
PY=/path/to/python bash scripts/smoke_no_npu.sh
```

Before a full extraction, run a small `MAX_SAMPLES=1` replay and inspect:

1. exact `token_ids` equality;
2. hidden shape/dtype/finite checks;
3. cache manifest provenance and checksums;
4. one DSpark and one DFlash2 CPU/NPU finite optimizer step;
5. a resume from a completed checkpoint.

This directory contains no vLLM proposer/evaluation adapter. Export and
runtime acceptance/TPS integration is a separate downstream gate after Stage C
produces a checkpoint.
