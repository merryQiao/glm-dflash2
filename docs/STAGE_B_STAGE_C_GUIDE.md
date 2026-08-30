# Qwen3-Omni Stage B and Stage C Guide

## 1. Scope and boundary

This package starts from **completed, immutable Stage A trajectory shards**.
Those shards must contain the exact engine-produced `prompt_token_ids` and
`response_token_ids`, the serialized multimodal conversation, checksums, the
model/processor revisions, and the Stage A generation fingerprint.

This package does not regenerate responses. Stage B never samples and never
turns saved response text back into token IDs. Its only job is to teacher-force
the exact saved token path through the same Qwen3-Omni Thinker and materialize
the tensors Stage C needs.

Only the Thinker text stream is trained. Talker, MTP/code predictor, and
Code2Wav are outside this package.

## 2. What Stage B does

Entry points:

- `scripts/extract_thinker_hidden_ascend.sh`
- `scripts/data/generate_thinker_hidden.py`

For each accepted Stage A condition, Stage B performs the following:

1. Reconstructs the original multimodal prompt with the pinned official
   processor.
2. Supplies `prompt_token_ids + response_token_ids` directly to vLLM-Ascend's
   `extract_hidden_states` path. Sampling is disabled; `max_tokens=1` exists
   only to drive the connector.
3. Requires connector token IDs to equal the saved trajectory IDs exactly.
4. Archives logical Thinker layers `[1,12,24,36,47]`, each with width 2048.
5. Obtains the raw layer-48 output and applies the exact checkpoint final
   RMSNorm offline, producing the real LM-head input.
6. Calls the official Qwen3-Omni `get_rope_index` path for the multimodal
   prompt and extends the response positions causally, producing the exact
   interleaved mRoPE coordinates.
7. Atomically writes safetensors shards, Parquet indices, checksums, runtime
   identity, model/processor revisions, and a final manifest.

Each sample is packed into these tensors:

| Tensor | Shape | Meaning |
|---|---:|---|
| `input_ids` | `[T]` | Exact Stage A prompt + response token IDs |
| `loss_mask` | `[T]` | True only on response tokens |
| `target_hidden_states` | `[T,5,2048]` | Selected Thinker layer states |
| `target_last_hidden_states` | `[T,2048]` | Post-final-RMSNorm LM-head input |
| `position_ids` | `[T,3]` | Temporal, height, width mRoPE coordinates |
| `offsets` | `[N+1]` | Packed sample boundaries within a shard |

### What changed in Stage B specifically for Stage C

The prior v2 cache already preserved exact token IDs, the response loss mask,
five selected hidden layers, and the final-normalized hidden state. Stage C
exposed one missing dependency: an Omni drafter cannot safely reconstruct
multimodal rotary positions from a flat token index.

The Stage C-compatible revision therefore makes these changes:

| Area | Prior Stage B | Stage C-compatible Stage B |
|---|---|---|
| Position information | Not archived | Exact official `position_ids [T,3]` archived |
| Cache schema | `omni-thinker-hidden-cache-v2` | `omni-thinker-hidden-cache-v3` |
| Position provenance | Implicit | `official_transformers_get_rope_index` recorded |
| Shard metadata | Hidden layout only | Adds position layout/axes and schema |
| Consumer behavior | v2 accepted | Stage C rejects v2 and missing positions |

The final-normalized hidden state and artifact identity checks were not
invented by Stage C; they were retained and made mandatory in the Stage C
loader. This prevents a raw pre-norm layer output or a cache from another model
revision from silently supervising the drafter.

Consequently, an old v2 cache must be regenerated. Adding synthetic 1-D
positions after the fact is intentionally unsupported.

## 3. What Stage C does

Entry points:

- `scripts/extract_thinker_target_io.sh`
- `scripts/train_thinker_drafter.sh`
- `tools/train_thinker_drafter.py`
- `src/omni_stage_c/`

Stage C first extracts exactly two BF16 tensors from the official target
checkpoint:

- `thinker.model.embed_tokens.weight`
- `thinker.lm_head.weight`

They are frozen and stored separately with checksums and the exact Stage B
cache fingerprint. The target model itself is not loaded during training.
The vocabulary is padded to 152064 and row 152063 is used as the draft mask;
extraction proves that this row is absent from all cached trajectories.

The drafter is a five-layer dense full-attention model with Qwen3-Omni Thinker
geometry: hidden size 2048, MLP size 6144, 32 query heads, 4 KV heads, head dim
128, Q/K norm, RMSNorm epsilon `1e-6`, RoPE theta `1e6`, and mRoPE section
`[24,20,20]`.

Stage C samples 512 response-contained anchors per sample with seed 42 and a
deterministic 4096-token window. On Ascend, context K/V is projected once per
layer and anchor blocks are evaluated in chunks of 16. This preserves the
intended block-local visibility without CUDA FlexAttention or cross-block
leakage.

### Supported training routes

| Method | Block | Objective |
|---|---:|---|
| DFlash | 8 | Exact full-vocab CE, depth gamma 4 |
| DFlash | 16 | Exact full-vocab CE, depth gamma 7 |
| DFlash2 | 8/16 | DFlash base CE + hit-only top-16 selector CE |
| DSpark | physical 8 / 7 successors | `0.1 CE + 0.9 full-vocab TV + 1.0 confidence BCE` |

DFlash2 uses the official-style two-tap group-16 dynamic convolution and a
rank-256 top-16 selector. The target token is never injected into the candidate
set. Base and selector components are normalized independently, including
cross-rank and gradient-accumulation normalization.

DSpark uses teacher-forced predecessor IDs. Its confidence target is detached
`1 - TV`, and its seven successor queries correspond to physical block size 8.

All routes use AdamW with lr `6e-4`, betas `(0.9,0.95)`, weight decay 0,
1000-step warmup, cosine decay to 0.1x, three epochs, gradient accumulation 8,
BF16, HCCL, and FSDP2.

## 4. End-to-end commands

### Stage B

Set at least these fields in `configs/generate_thinker_data.yaml`:

- `output.root`: completed Stage A trajectory root;
- `model.path`, `revision`, and `processor_revision`;
- `runtime.tensor_parallel_size` and Ascend runtime fields;
- `hidden_states.output_root` and `scratch_root`.

Then run:

```bash
cd /path/to/qwen3-omni-stageb-stagec-ascend

ASCEND_HARDWARE=a2 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
CONFIG=$PWD/configs/generate_thinker_data.yaml \
bash scripts/extract_thinker_hidden_ascend.sh
```

Stage B resumes at completed source shards and publishes `manifest.json` only
after every shard and checksum has passed verification.

### Frozen target I/O

```bash
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
HIDDEN_CACHE_DIR=/path/to/thinker_hidden \
TARGET_IO_DIR=/path/to/thinker_target_io \
bash scripts/extract_thinker_target_io.sh
```

### Stage C

```bash
ASCEND_HARDWARE=a2 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
bash scripts/train_thinker_drafter.sh \
  --method dflash2 \
  --block-size 16 \
  --hidden-cache-dir /path/to/thinker_hidden \
  --target-io-dir /path/to/thinker_target_io \
  --output-dir /path/to/outputs/dflash2-b16
```

Valid method/block pairs are:

```text
dflash:8  dflash:16  dflash2:8  dflash2:16  dspark:8
```

Resume requires a checkpoint directory containing `COMPLETE`; method, block,
cache identity, target-I/O checksums, semantic recipe, and world size must
match.

## 5. Outputs and observability

Stage B writes content-addressed safetensors and Parquet index shards plus a
verified manifest. Stage C writes atomic checkpoints and appends one JSON row
per optimizer step to `train_metrics.jsonl`.

Besides total loss and learning rate, the metrics include the components
relevant to the selected method: base loss, selector loss and candidate recall
for DFlash2; CE, TV, confidence, and target-top1 agreement for DSpark.

## 6. Tests and remaining gates

Local contract suite:

```bash
PYTHONPATH=$PWD/src:$PWD python -m pytest -q
python -m compileall -q src tools scripts/data tests
for script in scripts/*.sh; do bash -n "$script"; done
bash scripts/smoke_stage_c_training.sh
```

These checks validate schema, shapes, exact token equality, mRoPE construction,
objectives, model routes, checkpoint contracts, and CPU forward/backward.
They do not prove the destination image's NPU behavior.

Before a production run, the server operator must still verify:

1. The pinned vLLM-Ascend image exposes the native hidden connector and the
   requested logical layers with the assumed raw layer-48 semantics.
2. A representative multimodal sample reproduces exact token IDs and finite
   `[T,3]` positions.
3. All five Stage C routes complete an NPU optimizer step without CPU fallback.
4. HCCL/FSDP2 save and resume reproduce optimizer/scheduler state.
5. B16 with 512 anchors fits the target A2/A3 HBM budget.

Until these gates pass, this package is contract-complete but not claimed as
hardware-validated production code.
