# Ascend 910B execution runbook

This runbook is the operational checklist for the unified GLM-5.2 DFlash,
DFlash2 and DSpark pipeline. Local tests do not replace the **real 910B** gates.

## 1. Pin the runtime identity

Record the immutable GLM-5.2 BF16 weight fingerprint, model revision, tokenizer
fingerprint, CANN version, torch-npu version and SGLang version. Stage A,
Stage B and target token-I/O extraction must use this same identity. Do not use
quantized target weights for a BF16 experiment.

The BF16 target is roughly 1.5 TB before KV/activation headroom. A typical
deployment uses at least 32 x 64-GB 910B NPUs across two nodes. Reuse the Stage
A service, stop it, then start Stage B; do not keep duplicate target replicas.

## 2. Build frozen sampled trajectories

```bash
MODEL_PATH=/shared/models/GLM-5.2-bf16 \
ENDPOINT=http://glm52-sglang-service:30000 \
ENDPOINT_MANIFEST=/shared/identity/glm52-endpoint.json \
SERVED_MODEL_NAME=GLM-5.2 \
WORKSPACE_MAP=/shared/data/workspace_map.jsonl \
WORKSPACE_CACHE=/shared/cache/vibe-workspaces \
OPEN_SWE_STORE=/shared/data/open_swe_original.sqlite \
OUTPUT_JSONL=/shared/out/trajectories-shard-0.jsonl \
bash scripts/run_stage_a_trajectories.sh
```

For the pinned SGLang 0.5.16 image, first run
`scripts/apply_sglang_v0516_token_ids_patch.sh` inside the image and restart the
service. Stage A's one-token capability probe must return both prompt and
sampled response IDs before any real episode starts. The endpoint manifest is
an operator attestation, not cryptographic proof of the remote weight bytes.

Require a frozen manifest, zero unresolved errors, stable IDs and the intended
sampling parameters. A `MAX_SAMPLES` output is intentionally partial and may
only be used for gates.

## 3. Capture schema-v2 hidden streams

Run on both nodes, changing only `NODE_RANK`:

```bash
MODEL_PATH=/shared/models/GLM-5.2-bf16 \
TRAJECTORY_JSONL=/shared/out/trajectories-shard-0.jsonl \
OUTPUT_DIR=/shared/out/hidden-v2-shard-0 \
TP_SIZE=32 EP_SIZE=32 NNODES=2 NODE_RANK=0 \
DIST_INIT_ADDR=<rank-0-host> NCCL_PORT=29500 \
bash scripts/run_stage_b_hidden.sh
```

Expected logical taps are `[1,20,38,56,75]`; the cache must contain both
`aux_hidden_states.bin` and `target_final_hidden.bin`. Each sample is captured
with one teacher-forced target forward.

## 4. Extract token I/O, then calibrate and enforce numerical parity

```bash
MODEL_PATH=/shared/models/GLM-5.2-bf16 \
TARGET_IO_DIR=/shared/out/glm52-target-io \
bash scripts/extract_glm52_io.sh
```

The artifact must declare identity logits, no bias/scaling/softcap, BF16 source
dtypes and matching model/tokenizer fingerprints.

On one short fixed token sequence, save:

1. three repeated direct-forward captures;
2. one deliberately shifted-layer capture;
3. one deliberately pre-final-norm capture.

Each `.pt` contains `aux_hidden_states`, `target_final_hidden`, `target_logits`,
`input_ids` and `layer_ids`; all five files use exactly the same fixture IDs.
Run `tools/calibrate_hidden_capture_gate.py` with all six identity
fields. Keep the emitted JSON under experiment control; production validation
must not regenerate it.

Then run:

```bash
python tools/validate_hidden_cache.py \
  --cache-dir /shared/gate/hidden-v2 \
  --reference-pt /shared/gate/direct-1.pt \
  --parity-gate /shared/gate/hidden-parity-gate.json \
  --runtime-identity-json /shared/gate/runtime-identity.json \
  --target-io-dir /shared/out/glm52-target-io
```

Proceed only when cosine, max-absolute and mean-absolute errors pass and both
negative controls remain outside the calibrated bounds. This writes the fixed
`hidden-v2/parity_attestation.json`; training refuses a cache without it and
recomputes its cache and target-I/O bindings before allocating the drafter.

## 5. Run local and two-rank training gates

```bash
PY=/path/to/vendor/python bash scripts/smoke_no_model.sh

CACHE_DIR=/shared/gate/hidden-v2-frozen \
TARGET_IO_DIR=/shared/out/glm52-target-io \
MASK_TOKEN_ID=<verified-id> \
OUTPUT_DIR=/shared/gate/train-2rank \
bash scripts/gate_train_2rank_910b.sh
```

Require finite loss/gradients, unchanged frozen target I/O, complete checkpoint
markers and exact uninterrupted-vs-resume output parity.

## 6. Train one method

```bash
METHOD=dspark \
BLOCK_SIZE=8 \
CACHE_DIR=/shared/out/hidden-v2-frozen \
TARGET_IO_DIR=/shared/out/glm52-target-io \
OUTPUT_DIR=/shared/out/glm52-dspark \
MASK_TOKEN_ID=<verified-id> PAD_TOKEN_ID=<verified-id> \
NUM_NPUS=8 NNODES=1 NODE_RANK=0 \
bash scripts/train_glm52_drafter_910b.sh
```

Use `METHOD=dflash`, `dflash2`, or `dspark`. DFlash and DFlash2 support
`BLOCK_SIZE=8` and `16`; run both settings separately. DSpark supports only
`BLOCK_SIZE=8`, whose layout is one anchor plus seven proposed tokens. Its
default recipe is three epochs, `lr=6e-4`, gamma 4, rank-256 vanilla Markov,
Markov-aware confidence, and `0.1 CE + 0.9 TV + 1.0 BCE`, where
`TV=0.5*sum_v|p_target(v)-p_draft(v)|`.
For DFlash and DFlash2, B8 also uses gamma 4 while B16 uses gamma 7.
For multiple nodes, use identical
arguments and set `NODE_RANK` uniquely. Resume only from a `COMPLETE` step and
do not change cache identity, method, architecture, optimizer or scheduler.

## 7. Export and serving ABI gate

Training writes `OUTPUT_DIR/export`. Unlike the resumable training checkpoint,
this deployment artifact contains the trained draft plus frozen target
`embed_tokens.weight` and `lm_head.weight`. Verify all hashes in
`export_manifest.json` before loading it. Its config fixes
`sample_from_anchor=false`, so B8/B16 request 7/15 speculative tokens.

Load the exported draft in the exact Ascend serving fork and compare a fixed
batch against offline training with the same token IDs, anchors, positions and
cache fingerprint:

- all methods: common backbone/base logits;
- DFlash2: top-16 candidate IDs/scores and selector scores;
- DSpark: Markov chunk scores and confidence logits.

DFlash and DSpark use the public Speculators config/key contract. DFlash2 is
marked `custom-vllm-ascend-adapter-required`; do not bypass that preflight or
mislabel it as stock DFlash.

## 8. Acceptance and TPS benchmark

Use the same NPU set sequentially, never two co-resident target replicas:

```bash
TARGET_MODEL=/shared/models/GLM-5.2-bf16 \
DRAFTER_EXPORT=/shared/out/glm52-dspark/export \
PROMPTS_JSONL=/shared/eval/fixed-prompts.jsonl \
OUT_DIR=/shared/eval/glm52-dspark-b8 \
TP_SIZE=16 MAX_SAMPLES=100 MAX_TOKENS=2048 \
bash scripts/eval_vllm_ascend.sh
```

The output contains `baseline.json`, `speculative.json`, server logs, and
`comparison.json`. Acceptance is derived from vLLM Prometheus counters using
the official bonus-inclusive definition. TPS is measured from returned
completion-token counts and wall time. For temperature zero, exact output
parity is a hard gate. The launcher intentionally does not enable approximate
Block Verify or Entropy Verify.

The local test suite checks config/state round-trip and benchmark arithmetic;
it cannot prove that a particular CANN/vLLM-Ascend image contains the required
GLM DSpark support. Record the image digest and package versions with every
hardware result.

## Legacy v1 policy

Legacy v1 has only auxiliary hidden states. It can be opened only with the
explicit legacy reader for diagnosis. Every aligned training CLI rejects it;
never fabricate the missing final hidden or mix v1 and v2 shards.
