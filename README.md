# GLM-5.2 unified speculative-drafter pipeline

This repository builds one sampled GLM-5.2 BF16 training corpus and one
schema-v2 hidden cache, then trains three aligned offline consumers on Ascend
910B: **DFlash**, **DFlash2**, and **DSpark**. The target backbone is used only
during trajectory generation and hidden extraction; offline training loads no
GLM-5.2 transformer weights.

The production contract is fixed:

- target depth 78; logical hidden taps `[1,20,38,56,75]`;
- auxiliary hidden `[T,5,6144]` plus post-final-norm hidden `[T,6144]`;
- five-layer draft, hidden/intermediate `6144/12288`;
- `64/64` query/KV heads, head dimension 64, full attention, no sliding window;
- RoPE theta `8e6`, RMS epsilon `1e-5`;
- DFlash and DFlash2 train physical block sizes 8 and 16; DSpark trains only
  physical block size 8. A physical block contains one clean anchor, so B8/B16
  predict 7/15 speculative tokens respectively.

See [AI_HANDOFF.md](AI_HANDOFF.md) for project context and
[docs/ASCEND_910B_RUNBOOK.md](docs/ASCEND_910B_RUNBOOK.md) for executable
commands and hardware gates.

## End-to-end flow

1. **Stage A — sampled coding-agent trajectories.** SGLang runs GLM-5.2 with
   the production sampling policy (default `temperature=1.0`, `top_p=0.95`,
   top-k disabled), executes bounded tools, and freezes the resulting
   `input_ids` and target-position `loss_mask`.
2. **Stage B — one teacher-forced target forward.** A fresh SGLang internal
   runner consumes the frozen token path and captures the five auxiliary
   streams and post-final-norm LM-head input together.
3. **Target token I/O extraction.** Dense BF16/FP16/FP32
   `embed_tokens.weight` and `lm_head.weight` are extracted once and frozen.
4. **Offline training.** All methods use the same cache rows, stable sample IDs,
   deterministic anchors, common backbone, optimizer/checkpoint framework and
   absolute-position contract.

Stage A and Stage B intentionally run separately so two 753B target replicas
are never resident simultaneously. The frozen sampled path is teacher-forced;
Stage B does not regenerate or greedy-decode the answer.

## Cache schema v2

```text
hidden-cache/
  manifest.json
  index.jsonl
  segment-00000/
    input_ids.bin               # int64 [T]
    loss_mask.bin               # uint8/bool [T]
    aux_hidden_states.bin       # BF16 [T,5,6144]
    target_final_hidden.bin     # BF16 [T,6144]
```

Every stream slice has a SHA-256 digest. Data streams are flushed before the
index line, which is the sample commit point. `PackedHiddenDataset` exposes
`layer_hidden_states [T,5,6144]`, a flattened compatibility view
`hidden_states [T,30720]`, and `target_final_hidden [T,6144]`.

Schema v1 remains readable only through the explicit legacy adapter. It lacks
the final hidden stream and is rejected by all aligned DFlash/DFlash2/DSpark
training entrypoints; it is never silently upgraded.

The six BF16 hidden vectors cost 73,728 bytes per token, excluding IDs, masks,
index and filesystem overhead. At 630K samples and 1,000 tokens/sample this is
about 46.4 TB, so storage planning is mandatory.

## What each method trains

- **DFlash:** the common five-layer backbone with exact full-vocabulary CE,
  weighted by `exp(-depth/7)`.
- **DFlash2:** the same backbone plus identity-initialized two-tap grouped
  dynamic convolution and a rank-256/top-16 candidate selector. It uses base CE
  plus selector CE when the target occurs in base top-16.
- **DSpark:** the plain common backbone plus the official rank-256 vanilla
  Markov head `Embedding(V,256) -> Linear(256,V)` and a Markov-aware confidence
  head. Target distributions are reconstructed from
  `target_final_hidden @ frozen_lm_head.T`; its exact chunked loss is
  `0.1*CE + 0.9*full_vocab_L1 + 1.0*BCE`.

For DSpark, target token position `p` uses final hidden position `p-1`. At
depth zero the predecessor is the clean anchor; later depths use the
teacher-forced previous target. The LM-head matmul stays BF16 and logits/loss
normalization use FP32.

The DSpark confidence soft target is `1 - 0.5 * full_vocab_L1`; its default
GLM-5.2 five-layer preview recipe is three epochs, learning rate `6e-4`, gamma 4, and loss
`0.1*CE + 0.9*L1 + 1.0*confidence_BCE`. DFlash/DFlash2 retain three epochs
and learning rate `6e-4`; gamma is 4 for B8 and 7 for B16.

## Quick local verification

```bash
PY=/path/to/python bash scripts/smoke_no_model.sh
```

This runs the unit suite, creates and validates a schema-v2 mock cache, and
performs one finite optimizer update for DFlash, DFlash2 and DSpark.

## Production commands

Generate trajectories:

```bash
MODEL_PATH=/shared/models/GLM-5.2-bf16 \
ENDPOINT=http://glm52-sglang-service:30000 \
SERVED_MODEL_NAME=GLM-5.2 \
OUTPUT_JSONL=/shared/out/trajectories.jsonl \
bash scripts/run_stage_a_trajectories.sh
```

Extract both hidden streams with one target forward (run on every node with a
different `NODE_RANK`):

```bash
MODEL_PATH=/shared/models/GLM-5.2-bf16 \
TRAJECTORY_JSONL=/shared/out/trajectories.jsonl \
OUTPUT_DIR=/shared/out/hidden-v2 \
TP_SIZE=32 EP_SIZE=32 NNODES=2 NODE_RANK=0 \
DIST_INIT_ADDR=<rank-0-host> NCCL_PORT=29500 \
bash scripts/run_stage_b_hidden.sh
```

Extract the frozen target token I/O:

```bash
MODEL_PATH=/shared/models/GLM-5.2-bf16 \
TARGET_IO_DIR=/shared/out/glm52-target-io \
bash scripts/extract_glm52_io.sh
```

Train any aligned method:

```bash
METHOD=dflash2 \
BLOCK_SIZE=16 \
CACHE_DIR=/shared/out/hidden-v2 \
TARGET_IO_DIR=/shared/out/glm52-target-io \
OUTPUT_DIR=/shared/out/glm52-dflash2 \
MASK_TOKEN_ID=<verified-mask-id> PAD_TOKEN_ID=<verified-pad-id> \
NUM_NPUS=8 \
bash scripts/train_glm52_drafter_910b.sh
```

Set `METHOD=dflash`, `dflash2`, or `dspark`. DFlash/DFlash2 accept
`BLOCK_SIZE=8` or `16`; DSpark requires `BLOCK_SIZE=8`. Multi-node training also accepts
`NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`. The deprecated
`train_glm52_dflash2_910b.sh` is only a compatibility wrapper.

## Numerical hidden-capture gate

Before producing the full cache, collect three independent direct-forward
captures of the same short token sequence plus deliberately shifted-layer and
pre-final-norm controls. Calibrate once:

```bash
python tools/calibrate_hidden_capture_gate.py \
  --direct-run direct-1.pt --direct-run direct-2.pt --direct-run direct-3.pt \
  --shifted-layer-control shifted.pt --pre-norm-control prenorm.pt \
  --target-fingerprint <sha> --model-revision <revision> \
  --tokenizer-fingerprint <sha> --cann-version <version> \
  --torch-npu-version <version> --sglang-version <version> \
  --output hidden-parity-gate.json
```

The stored bound for every metric is exactly
`max(explicit_floor, 2 * worst_direct_vs_direct_variation)` and must remain
strictly below both negative controls. Validation only loads this artifact; it
never recalibrates or widens thresholds silently:

```bash
python tools/validate_hidden_cache.py \
  --cache-dir /shared/gate/hidden-v2 \
  --reference-pt direct-1.pt \
  --parity-gate hidden-parity-gate.json \
  --runtime-identity-json runtime-identity.json
```

## Explicit limits

- The real 910B hidden parity, two-rank FSDP2 resume, and serving ABI gates must
  still be run on the actual CANN/torch-npu/SGLang deployment stack.
- The repository does not claim a final Ascend speculative-decoding runtime.
  Export parity must compare method-specific logits/heads in the serving fork.
- `MASK_TOKEN_ID` is mandatory and must come from the actual tokenizer/runtime;
  EOS or PAD must not be substituted.
- Quantized target token-I/O artifacts, mismatched model revisions, ambiguous
  hidden taps and pre-final-norm substitutes fail closed.
