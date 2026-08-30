# Ascend 910B A2 runbook

This runbook is the production handoff for GLM-5.3-Flash Stage A, Stage B and
offline drafter training. It assumes a vendor image in which CANN, PyTorch,
`torch_npu`, and Ascend SGLang are already mutually compatible. Do not install
generic PyPI PyTorch into that image.

## 1. Preflight

```bash
python - <<'PY'
import torch
import torch_npu
import sglang

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("sglang", sglang.__version__)
print("device_count", torch.npu.device_count())
for index in range(torch.npu.device_count()):
    print(index, torch.npu.get_device_name(index))
print("CANN", torch_npu.npu.get_cann_version("CANN"))
PY
```

Production Stage B accepts only a live device name in the Ascend 910B family.
The locked SGLang build must expose its PyTorch `ModelRunner`, Ascend attention
backend, exact-token response IDs, hidden-capture hook, and TP-aware global
`compute_logits` API. Missing APIs fail closed; do not bypass them.

## 2. Stage A

```bash
MODEL_PATH=/models/GLM-5.3-Flash-BF16 \
DATASET=/data/vibe_coding_630k \
OUTPUT_JSONL=$PWD/outputs/glm53/trajectories.jsonl \
TP_SIZE=16 \
bash scripts/run_stage_a_trajectories.sh
```

First run with `MAX_SAMPLES=10`. Confirm that the manifest is frozen and every
record contains exact prompt/response IDs, flattened IDs, a loss mask, tool
events, and workspace metadata. A smoke-unverified artifact is never valid for
production training.

## 3. Stage B

```bash
MODEL_PATH=/models/GLM-5.3-Flash-BF16 \
TRAJECTORY_JSONL=$PWD/outputs/glm53/trajectories.jsonl \
OUTPUT_DIR=$PWD/outputs/glm53_hidden_cache \
TP_SIZE=16 \
bash scripts/run_stage_b_hidden.sh
```

Start with `MAX_SAMPLES=10`. A successful production manifest must report:

- `status=frozen` and `production_eligible=true`;
- logical layers `[1,11,22,32,42]` and physical indices `[2,12,23,33,43]`;
- passing per-row auxiliary-tap and final-logit parity;
- live Ascend 910B, SGLang, torch-npu and CANN evidence;
- immutable index, stream and provenance checksums.

## 4. Target I/O

```bash
MODEL_PATH=/models/GLM-5.3-Flash-BF16 \
CACHE_DIR=$PWD/outputs/glm53_hidden_cache \
TARGET_IO_DIR=$PWD/outputs/glm53_target_io \
bash scripts/extract_target_io.sh
```

This step rejects quantized, tied, biased or logit-transformed checkpoints and
binds the output to the hidden-cache identity.

## 5. Training

Resolve the mask ID from the exact frozen tokenizer, then run one recipe:

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

Other valid pairs are `dflash/8`, `dflash2/8`, `dflash2/16`, and `dspark/8`.
The value `154821` is not a tunable placeholder: target-I/O extraction verifies
that it is the exact special `[MASK]` token in the frozen GLM-5.3 tokenizer,
and training binds the same tokenizer fingerprint and target-I/O artifact.

Inspect `semantic_config.json` before a production run. It records the durable
Stage A route mixture, deterministic train/held-out split, long-window cycling
policy, and exact mask identity. During training, monitor
`optimizer_metrics.jsonl`; each optimizer step is conflict-safe across resume
and reports additive losses/denominators, DFlash2 unary recall, or DSpark
TV/confidence components. For DFlash2, `unary_recall` is the real
base top-k hit rate and the selector denominator contains hits only. Validation
uses a fixed held-out window and
does not contribute gradients.
Run a short job first and retain evidence for:

1. BF16 forward/backward with bool SDPA at a 4096-token window;
2. HCCL all-reduce and FSDP2 accumulation;
3. save/resume parity for model, optimizer, scheduler, RNG and sampler cursor;
4. representative peak HBM below physical capacity.

The aggregate evidence checker is `scripts/run_ascend_training_gate.sh`.

## 6. What is not yet authorized

Do not report speculative acceptance length, TPS, or speedup from these
checkpoints yet. The current training artifact deliberately records
`runtime_attested=false`. Formal evaluation requires a separate GLM-5.3 state
rollback implementation and parity proof for all KDA/linear-attention and
short-convolution states after partial block acceptance.
