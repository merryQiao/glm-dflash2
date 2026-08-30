# GLM-5.3-Flash DFlash-family training

This directory is a standalone, text-only Ascend 910B A2 pipeline for
**GLM-5.3-Flash-BF16**. It does not import or modify the adjacent GLM-5.2
implementation. The implemented path is:

```text
sampled coding-agent rollout (Stage A)
  -> exact-token teacher replay and hidden capture (Stage B)
  -> frozen target embedding/lm_head extraction
  -> offline DFlash / DFlash2 / DSpark training
```

Stage A preserves assistant/tool messages, exact prompt and sampled response
token IDs, flattened `input_ids`, and the assistant `loss_mask`. Stage B
teacher-forces that exact path and stores five target hidden streams plus the
post-final-norm hidden state. It never regenerates the response.

The production path is NPU-only. CUDA, NCCL, FlashAttention2, and a generic
CPU runner are not accepted as production substitutes. CPU tests validate
contracts only; they cannot mint a production-frozen cache.

## Requirements

```bash
python -m pip install -r requirements.txt
```

The Stage A endpoint must expose an OpenAI-compatible chat API and return both
`prompt_token_ids` and `response_token_ids`. The generator probes this before
committing trajectories. Parser defaults are `glm45` for reasoning and `glm47`
for tool calls; override them only when the deployed runtime requires different
registered parser names.

`torch`, `torch_npu`, CANN, and the vendor SGLang build must come from one
compatible Ascend image. They are intentionally not installed by
`requirements.txt`, because installing generic PyPI PyTorch over that image can
break the NPU runtime.

## Prepare optional Open-SWE prefixes

Rows routed through existing Open-SWE trajectories require a local SQLite
store:

```bash
PYTHONPATH=src python tools/prepare_open_swe_trajectories.py --help
```

The generator fails before rollout if such rows are selected and the store is
missing.

## Run with a local SGLang server

```bash
MODEL_PATH=/models/GLM-5.3-Flash-BF16 \
DATASET=/data/vibe_coding_630k \
OUTPUT_JSONL=$PWD/outputs/glm53/trajectories.jsonl \
TP_SIZE=16 \
bash scripts/run_stage_a_trajectories.sh
```

The default contract is BF16, NPU, Ascend attention, temperature `1.0`, top-p
`0.95`, and top-k disabled. All rollout-affecting service, sampling, tokenizer,
template, dataset, workspace, and sharding identities are stored in the JSONL
manifest. Resume is rejected if any immutable field changes.

## Run against an existing endpoint

Production generation requires an endpoint attestation manifest:

```bash
MODEL_PATH=/models/GLM-5.3-Flash-BF16 \
DATASET=/data/vibe_coding_630k \
OUTPUT_JSONL=$PWD/outputs/glm53/trajectories.jsonl \
ENDPOINT=http://127.0.0.1:30000 \
ENDPOINT_MANIFEST=/path/to/endpoint-manifest.json \
bash scripts/run_stage_a_trajectories.sh
```

The attestation must identify the exact endpoint configuration rather than the
local launcher defaults. Its required shape is:

```json
{
  "schema": "glm-sglang-endpoint-v1",
  "served_model_name": "GLM-5.3-Flash-BF16",
  "model_fingerprint": "<sha256 from the local MODEL_PATH>",
  "tokenizer_fingerprint": "<sha256 from the local MODEL_PATH>",
  "dtype": "bfloat16",
  "runtime": {
    "sglang_version": "<version>",
    "cann_version": "<version>",
    "image_digest": "<immutable image digest>",
    "tp_size": 16,
    "device": "npu",
    "attention_backend": "ascend",
    "reasoning_parser": "glm45",
    "tool_call_parser": "glm47",
    "context_length": 131072,
    "max_total_tokens": 131072,
    "quantization": null,
    "moe_a2a_backend": null,
    "deepep_mode": null
  }
}
```

For an external endpoint, parser/backend/model settings in the run manifest are
taken from this attestation; local `REASONING_PARSER`, `DEVICE`, and related
launcher values are not misreported as properties of the remote server.

An endpoint without a manifest is allowed only for a bounded hardware smoke:

```bash
MODEL_PATH=/models/GLM-5.3-Flash-BF16 \
DATASET=/data/vibe_coding_630k \
OUTPUT_JSONL=$PWD/outputs/smoke.jsonl \
ENDPOINT=http://127.0.0.1:30000 \
ALLOW_UNVERIFIED_ENDPOINT=1 \
MAX_SAMPLES=10 \
bash scripts/run_stage_a_trajectories.sh
```

Unverified runs are hard-limited to 1–50 samples. Their manifest records
`production_eligible=false` and `status=smoke_unverified`; they can never become
a frozen production artifact.

## Stage B: exact hidden replay

Run this on the real Ascend 910B A2 deployment image:

```bash
MODEL_PATH=/models/GLM-5.3-Flash-BF16 \
TRAJECTORY_JSONL=$PWD/outputs/glm53/trajectories.jsonl \
OUTPUT_DIR=$PWD/outputs/glm53_hidden_cache \
TP_SIZE=16 \
bash scripts/run_stage_b_hidden.sh
```

The fixed logical target taps are `[1,11,22,32,42]`; their Transformers-style
hidden-state indices are `[2,12,23,33,43]`. Every cache row has:

```text
input_ids          [T]          int64
loss_mask          [T]          bool
aux_hidden_states  [T,5,4096]   bfloat16
target_final_hidden[T,4096]     bfloat16
```

Production freeze is fail-closed. For every row the runner independently hooks
all five decoder blocks, proves their order against the packed capture, hooks
the final norm, projects it through the real TP-aware SGLang logits path, and
compares with native logits. At freeze time it re-probes the live
`torch_npu` device, actual SGLang classes, installed versions, and CANN version.
An unverified endpoint, CPU fake, non-910B device, TP shard, missing hook, or
failed parity leaves the cache non-production.

For multi-node extraction, launch the same command on each node and set
`NNODES`, `NODE_RANK`, `DIST_INIT_ADDR`, and `DIST_PORT`. Only global rank zero
writes the packed cache.

## Frozen target token I/O

The target embedding and LM head are extracted once and bound to the immutable
cache identity:

```bash
MODEL_PATH=/models/GLM-5.3-Flash-BF16 \
CACHE_DIR=$PWD/outputs/glm53_hidden_cache \
TARGET_IO_DIR=$PWD/outputs/glm53_target_io \
bash scripts/extract_target_io.sh
```

The extractor and loader both require the exact unquantized GLM5Next contract:
45 text layers, hidden size 4096, vocabulary 154880, RMS epsilon `1e-5`, BF16,
untied embedding/head, no head bias, and no logit scaling or softcap.

## Offline drafter training

All methods reuse the same hidden cache and frozen target I/O. Production
launches use `torch_npu`, HCCL, BF16 and FSDP2:

```bash
MASK_TOKEN_ID=154821 \
NPROC_PER_NODE=8 \
bash scripts/train_drafter.sh \
  --method dflash \
  --block-size 16 \
  --hidden-cache "$PWD/outputs/glm53_hidden_cache" \
  --target-io "$PWD/outputs/glm53_target_io" \
  --output-dir "$PWD/outputs/dflash-b16"
```

Valid experiment combinations are:

| Method | External block | Internal queries | Proposed tokens |
|---|---:|---:|---:|
| DFlash | 8 / 16 | 8 / 16 | 7 / 15 |
| DFlash2 | 8 / 16 | 8 / 16 | 7 / 15 |
| DSpark | 8 | 7 | 7 |

DFlash/DFlash2 supervise positions after the clean anchor with exact
full-vocabulary CE and `exp(-depth/gamma)` (`gamma=4/7`). DFlash2 additionally
follows the NeMo AutoModel selector objective: it reranks the real base top-k
candidates and applies selector CE only where that top-k already contains the
target. A miss is excluded from both selector numerator and denominator; the
target is never injected, so training and inference use the same candidates.
`unary_recall` reports the base top-k hit rate over valid positions. This selector path
remains experimental until the Ascend runtime parity gate is implemented.
DSpark follows its
method-specific convention: query zero is the verified anchor and predicts the
first successor, so all seven internal outputs are supervised. Its objective
is `0.1 CE + 0.9 exact-TV + 1.0 confidence-BCE`, with a rank-256 token-ID
Markov residual head. All recipes use 64 deterministic anchors, three epochs,
LR `6e-4`, warmup 1000 optimizer steps, cosine decay, per-rank batch 1 and
gradient accumulation 8. Long trajectories use a deterministic cycle of
overlapping 4096-token windows while retaining an assistant-contained block and
absolute position IDs; anchor buckets cycle independently across epochs instead
of repeatedly selecting the same prefix.

Stage A persists `generation_route` on every committed row and recomputes route
counts from the durable JSONL at shutdown. The hidden-cache writer carries that
route forward. Training reserves a deterministic, disjoint one-percent
held-out set capped at 128 rows, records route counts and the split identity in
`semantic_config.json`, and evaluates fixed epoch-0 validation windows every
1000 optimizer steps and at the final step. `optimizer_metrics.jsonl` contains
idempotent train/validation records with LR and additive numerator,
denominator, and mean for every method-specific component. FSDP padding rows
run collectively but have zero metric/loss weight.

The frozen target-I/O contract resolves the exact special tokenizer token
`[MASK]` and requires production ID `154821`. Training rejects a caller ID,
tokenizer fingerprint, or target-I/O artifact identity mismatch before model
construction.

Checkpoints are written only at optimizer boundaries and include sharded model,
optimizer, scheduler, RNG, sampler cursor and semantic configuration. Resume
rejects any semantic mismatch.

## Contract smoke and Ascend acceptance gate

The CPU smoke checks all five method/block recipes without claiming NPU
support:

```bash
bash scripts/smoke_stage_b_training.sh
```

Unit tests validate command construction and data contracts, not vendor runtime
support. Before a full run, execute a 10–50 sample smoke on the actual Ascend
SGLang image and verify:

1. GLM-5.3-Flash loads with the configured parser names and attention backend.
2. Chat responses include exact prompt and sampled response token IDs.
3. Tool calls execute and replay to exactly the same frozen token sequence.
4. Every Stage B row passes tap-order and final-logit parity.
5. A real BF16/HCCL/FSDP2 update, save and resume is identical to an
   uninterrupted update.
6. Bool-mask SDPA works for the fixed 4096-token training window and leaves
   representative HBM headroom.

Until this gate passes on the target 910B environment, runtime compatibility is
an explicit unresolved hardware dependency rather than a claimed result.

## Deliberate evaluation boundary

The produced checkpoint is a training-complete candidate, not yet a deployable
GLM-5.3 speculative runtime artifact. GLM-5.3 has KDA/linear-attention recurrent
and short-convolution states. After partial block acceptance, all such states
must equal the state obtained by committing exactly that accepted prefix. A KV
crop alone is insufficient. Candidate capability metadata therefore remains
`runtime_attested=false`; acceptance-length/TPS evaluation must stay blocked
until a rollback strategy passes all-state parity on Ascend 910B A2.
