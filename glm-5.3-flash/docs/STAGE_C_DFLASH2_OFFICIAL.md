# GLM-5.3-Flash Stage C: DFlash2 offline training

This document is the standalone entry point for training the GLM-5.3-Flash
DFlash2 drafter from frozen Stage B artifacts on Ascend 910B A2.

## Scope

Stage C consumes, but does not regenerate:

- a production-frozen hidden cache containing exact target-trajectory token IDs,
  assistant loss masks, five auxiliary hidden streams, and post-final-norm hidden;
- a frozen target-I/O artifact containing the GLM-5.3 embedding and LM head;
- the exact `[MASK]` token ID and tokenizer fingerprint bound to those artifacts.

The implementation is GLM-specific in model shape and hidden-layer selection,
while the DFlash2 optimization semantics follow NVIDIA NeMo AutoModel's public
`train_dflash2.py`, `dflash2_core.py`, and `draft_qwen3_dflash2.py` behavior.

## Model and objective

The GLM-5.3 draft backbone uses five dense full-attention layers with hidden
size 4096, intermediate size 12288, 64 query/KV heads, head dimension 64, and
five projected target hidden streams. Target embedding and LM head are frozen.

DFlash2 adds:

- identity-initialized two-tap dynamic causal convolution around every
  attention and MLP sublayer, with channel group size 16;
- a rank-256, top-16 pairwise path selector;
- teacher-forced ground-truth predecessor token IDs during selector training.

For every non-anchor block position:

1. The base loss is exact full-vocabulary CE with depth weight
   `exp(-depth/gamma)`, where gamma is 4 for B8 and 7 for B16.
2. The selector scores the real base top-16 candidate set. Training never
   injects the target into that set.
3. Selector CE is applied only when the target already occurs in top-16. A miss
   contributes neither selector numerator nor selector denominator.
4. The optimized loss is
   `base_ce.mean + selector_ce_on_hits.mean`; the two terms are normalized
   independently and have coefficient 1.0.
5. `unary_recall` is the unweighted top-16 target-hit rate over valid supervised
   positions and is diagnostic only.

The same candidate set is therefore used in training and inference.

## Supported recipes

| Method | Physical block size | Proposed tokens | Gamma |
|---|---:|---:|---:|
| DFlash2 | 8 | 7 | 4 |
| DFlash2 | 16 | 15 | 7 |

The current GLM experiment recipe uses 64 deterministic anchors per sample,
three epochs, LR `6e-4`, 1000 optimizer-step warmup, cosine decay, per-rank
batch size 1, gradient accumulation 8, BF16, HCCL, and FSDP2. These experiment
hyperparameters are recorded separately from the official DFlash2 loss
semantics in `semantic_config.json`.

## Launch

```bash
cd glm-5.3-flash

MASK_TOKEN_ID=154821 \
NPROC_PER_NODE=8 \
bash scripts/train_drafter.sh \
  --method dflash2 \
  --block-size 16 \
  --hidden-cache /path/to/glm53_hidden_cache \
  --target-io /path/to/glm53_target_io \
  --output-dir /path/to/dflash2-b16 \
  --checkpoint-every 10000
```

Use `--block-size 8` for the B8 recipe. Production training accepts only NPU,
HCCL, BF16, and FSDP2 through the launcher.

## Outputs and resume

The output directory contains:

- `semantic_config.json`: immutable data/model/tokenizer/recipe contract;
- `optimizer_metrics.jsonl`: idempotent train and held-out metrics;
- `checkpoints/step-XXXXXXXX`: sharded model, optimizer, scheduler, RNG, and
  sampler state;
- `candidate-capability.json`: final candidate capability record.

Resume requires the same hidden-cache identity, target-I/O checksum, mask/tokenizer
identity, model configuration, method, block size, and recipe semantics.

## Verification

CPU contract and objective tests:

```bash
PYTHONPATH=$PWD/src python -m pytest -q \
  tests/test_glm53_training_models.py \
  tests/test_glm53_training_objectives.py \
  tests/test_glm53_training_offline.py \
  tests/test_glm53_training_entrypoints.py
```

Before a full Ascend run, also verify on the real 910B A2 image:

1. BF16 forward/backward and bool-mask SDPA;
2. HCCL/FSDP2 gradient accumulation;
3. checkpoint save/resume parity;
4. representative peak HBM headroom;
5. non-zero selector denominator and plausible `unary_recall`.

CPU tests validate code semantics only and do not certify the vendor runtime or
the later vLLM-Ascend DFlash2 serving adapter.
