# Qwen3-Omni Thinker Stage C

For the complete Stage B-to-Stage C artifact flow and the exact changes made
to Stage B, read [`STAGE_B_STAGE_C_GUIDE.md`](STAGE_B_STAGE_C_GUIDE.md).

Stage C trains an embedding/head-free speculative drafter from the immutable
Stage A trajectory and Stage B hidden cache. It never loads the 30B target
during training.

## Fixed contract

| Item | Value |
|---|---|
| Target stream | Qwen3-Omni Thinker text tokens only |
| Target layers | 48; cached logical layers `[1,12,24,36,47]` |
| Residual width | 2048 |
| Attention | 32 Q heads, 4 KV heads, head dim 128, Q/K norm |
| Positions | exact cached three-axis interleaved mRoPE `[T,3]` |
| Draft backbone | 5 dense full-attention layers, MLP width 6144 |
| Frozen I/O | `thinker.model.embed_tokens.weight`, `thinker.lm_head.weight` |
| Mask row | padded vocabulary row 152063; extraction proves it is absent from trajectories |
| Sampling | 512 valid response-contained anchors/sample, seed 42 |
| Window | deterministic 4096-token window |
| Optimizer | AdamW, lr `6e-4`, betas `(0.9,0.95)`, no weight decay |
| Schedule | 1000-step warmup then cosine to 0.1x |
| Training | 3 epochs, gradient accumulation 8, BF16 FSDP2/HCCL |

The Ascend implementation preserves the official 512-anchor semantics without
requiring CUDA FlexAttention: target-context K/V are projected once per layer,
anchors are evaluated in exact block-local chunks of 16, and five-layer
activation checkpointing is enabled. This removes mathematically irrelevant
cross-block attention while preserving the official visibility mask.

Supported routes are DFlash B8/B16, DFlash2 B8/B16, and DSpark B8.

- DFlash uses exact chunked full-vocabulary CE with depth decay gamma 4 (B8)
  or 7 (B16).
- DFlash2 adds official two-tap group-16 dynamic convolutions and a rank-256,
  top-16 candidate selector. The selector sees only the real base top-16;
  missed targets are never injected. Base CE and hit-only selector CE are
  normalized independently and summed.
- DSpark has seven successor queries for physical B8 and optimizes
  `0.1 CE + 0.9 full-vocabulary TV + 1.0 confidence BCE`, with gamma 4.
  Its predecessor tokens are teacher-forced and the confidence target is
  detached `1-TV`.

## 1. Extract frozen target I/O

This reads only two BF16 tensors from the official checkpoint and binds them
to the exact Stage B cache fingerprint. Quantized checkpoints, tied/biased/
scaled heads, a reused mask ID, corrupt cache files, or a model/cache identity
mismatch fail before training.

```bash
cd /path/to/omni-sd-ascend
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
HIDDEN_CACHE_DIR=/path/to/thinker_hidden \
TARGET_IO_DIR=/path/to/thinker_target_io \
bash scripts/extract_thinker_target_io.sh
```

## 2. CPU functional smoke

The smoke executes a real forward, backward, and AdamW step for all five
method/block routes. It validates code contracts, not Ascend kernels.

```bash
PYTHONPATH=$PWD/src:$PWD python - <<'PY'
from tools.train_thinker_drafter import run_tiny_smoke
for route in [('dflash',8),('dflash',16),('dflash2',8),('dflash2',16),('dspark',8)]:
    print(route, run_tiny_smoke(method=route[0], block_size=route[1]))
PY
```

## 3. Production A2/A3 training

One process is launched per visible NPU. Unlike Stage A/B, Stage C uses
`torchrun` because the 30B target is absent and the five-layer drafter is
sharded with FSDP2.

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

Resume only from a directory containing `COMPLETE`; semantic configuration,
cache identity, target-I/O checksum, and world size must match exactly:

```bash
... bash scripts/train_thinker_drafter.sh \
  --method dflash2 --block-size 16 \
  --hidden-cache-dir /path/to/thinker_hidden \
  --target-io-dir /path/to/thinker_target_io \
  --output-dir /path/to/outputs/dflash2-b16 \
  --resume /path/to/outputs/dflash2-b16/checkpoints/step-00001000
```

Rank 0 appends optimizer-step metrics to `train_metrics.jsonl`. Besides the
actual optimized `loss` and learning rate, it records the additive global
components available for the selected method: base/selector/candidate recall
for DFlash2, and CE/TV/confidence/target-top1 agreement for DSpark.

## Hardware gates still required

Local tests do not establish NPU throughput or numerics. On the destination
image, run all five routes for at least one optimizer step and one save/resume
cycle, then verify: HCCL completes, FSDP2 restores exactly, BF16 GQA SDPA is
finite, representative B16/512-anchor HBM fits, and no fallback to CPU occurs.
Do not claim production readiness until those artifacts are recorded.
