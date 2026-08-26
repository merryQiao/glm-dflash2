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
4. **Offline training and deployment export.** All methods use the same cache
   rows, stable sample IDs, deterministic anchors, common backbone,
   optimizer/checkpoint framework and absolute-position contract. The final
   `export/` combines the trained draft with the frozen target embedding and
   LM head required by a serving runtime.

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
- **DSpark:** the plain common backbone plus a rank-256 vanilla
  Markov head `Embedding(V,256) -> Linear(256,V)` and a Markov-aware confidence
  head. Target distributions are reconstructed from
  `target_final_hidden @ frozen_lm_head.T`; its exact chunked loss is
  `0.1*CE + 0.9*TV + 1.0*BCE`, where
  `TV=0.5*sum_v|p_target(v)-p_draft(v)|`.

For DSpark, target token position `p` uses final hidden position `p-1`. At
depth zero the predecessor is the clean anchor; later depths use the
teacher-forced previous target. The LM-head matmul stays BF16 and logits/loss
normalization use FP32.

The DSpark confidence soft target is `1 - TV`; the unified
GLM-5.2 five-layer recipe is three epochs, learning rate `6e-4`, gamma 4, and loss
`0.1*CE + 0.9*TV + 1.0*confidence_BCE`. DFlash/DFlash2 retain three epochs
and learning rate `6e-4`; gamma is 4 for B8 and 7 for B16.

## Quick local verification

```bash
PY=/path/to/python bash scripts/smoke_no_model.sh
```

This runs the unit suite, creates and validates a schema-v2 mock cache, and
performs one finite optimizer update for DFlash, DFlash2 and DSpark.

## Production commands

### Stage A — rollout trajectories

For a large GLM-5.2 deployment, reuse its SGLang endpoint so this driver does
not need to launch the multi-node service itself:

```bash
cd /path/to/glm-dflash2

MODEL_PATH=/shared/models/GLM-5.2 \
ENDPOINT=http://glm52-sglang-service:30000 \
SERVED_MODEL_NAME=GLM-5.2 \
WORKSPACE_MAP=/shared/data/workspace_map.jsonl \
WORKSPACE_CACHE=/shared/cache/vibe-workspaces \
OPEN_SWE_STORE=/shared/data/open_swe_original.sqlite \
OUTPUT_JSONL=/shared/out/trajectories-shard-0-of-1.jsonl \
bash scripts/run_stage_a_trajectories.sh
```

Stage A defaults to `WORKERS=8` trajectory workers but permits only
`MAX_RUNNING_REQUESTS=2` concurrent model HTTP calls. Tool execution, Git,
containers, and workspace preparation remain concurrent while the shared
request semaphore bounds target KV/cache pressure even for an external
endpoint. A local SGLang server additionally receives
`--max-running-requests 2 --max-total-tokens 131072`. The latter two settings
must also be configured on an independently launched external SGLang service;
the client semaphore limits requests from this driver but cannot constrain
other clients of that service.

Start conservatively on a new Ascend deployment:

```bash
WORKERS=4 MAX_RUNNING_REQUESTS=1 MAX_TOTAL_TOKENS=131072 \
MAX_SAMPLES=50 ... bash scripts/run_stage_a_trajectories.sh
```

Then use the default `8/2` profile only after checking peak HBM. Do not copy
the Qwen3.8 reference's `12/8` profile to GLM-5.2 BF16 without a representative
long-context load test. `EPISODE_RETRIES=2` retries an isolated workspace
episode; only the main thread commits JSONL records and the error ledger, so a
retry cannot produce duplicate committed IDs.

`MODEL_PATH` is still required locally for the exact tokenizer/chat template.
An external OpenAI endpoint exposes its served model name but not a weight
digest, so the manifest records `weight_identity_verified=false`. Operational
deployment must bind that endpoint to the same immutable model revision as
`MODEL_PATH`; a local temporary server is fingerprint-verifiable.
Repository-backed rows without a materialized workspace are hard errors; they
are never silently converted into invented tool traces.

The Stage A route behavior follows the specified SpecForge
`vibe_coding_qwen38.py` pipeline:

- repo references use cached Git mirrors and disposable worktrees;
- executable-repo references use disposable containers;
- `file_before_change` creates an isolated temporary repository;
- Open-SWE prefix rows restore complete original trajectories from a read-only
  SQLite store instead of inventing missing tool observations.

Build that store once if the selected shard contains Open-SWE rows:

```bash
PYTHONPATH=src python tools/prepare_open_swe_trajectories.py \
  --dataset data/vibe_coding_630k \
  --output outputs/open_swe_original.sqlite
```

Then set `OPEN_SWE_STORE` and, if desired, `WORKSPACE_CACHE` in the Stage A
command. Container execution remains sandboxed; host tests are disabled unless
`ALLOW_HOST_TESTS=1` is explicitly set.

Independent full model replicas can data-shard by stable selected-row index:

```bash
DATA_SHARD_COUNT=2 DATA_SHARD_INDEX=0 ... bash scripts/run_stage_a_trajectories.sh
DATA_SHARD_COUNT=2 DATA_SHARD_INDEX=1 ... bash scripts/run_stage_a_trajectories.sh
```

Without `ENDPOINT`, the script launches a temporary local SGLang server using
the GLM parsers `reasoning=glm45` and `tool-call=glm47`, then shuts it down
before returning.

The default generation policy is sampling, not greedy. To override it for an
explicit ablation, set `TEMPERATURE`, `TOP_P`, and `TOP_K`; the exact values are
part of the resume manifest and cannot silently change within a shard.

For a production BF16 multi-node GLM service, the earlier `ENDPOINT=...` form
is preferred: Stage A is only an OpenAI client and does not duplicate the
target model.

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

Every successful training run writes a resumable `step-N/` training checkpoint
and one deployment artifact under `export/`:

```text
export/
  config.json
  config.py
  model.safetensors       # draft + frozen embed_tokens + frozen lm_head
  export_manifest.json    # checksums, target identity, runtime compatibility
```

The exported proposal count is `block_size - 1`: the physical block's first
position is the known anchor and is never sampled. DFlash and DSpark exports
use the public Speculators key/config contract. DFlash2 is exported exactly for
round-trip and provenance, but is explicitly marked
`custom-vllm-ascend-adapter-required`; stock vLLM-Ascend must not load it as
ordinary DFlash.

Run a target-only versus speculative benchmark sequentially on the same 910B
devices:

```bash
TARGET_MODEL=/shared/models/GLM-5.2-bf16 \
DRAFTER_EXPORT=/shared/out/glm52-dspark/export \
PROMPTS_JSONL=/shared/eval/fixed-prompts.jsonl \
OUT_DIR=/shared/eval/glm52-dspark-b8 \
TP_SIZE=16 MAX_SAMPLES=100 MAX_TOKENS=2048 \
bash scripts/eval_vllm_ascend.sh
```

The launcher does not co-locate the baseline and speculative servers. It reads
vLLM's `spec_decode_num_drafts`, `spec_decode_num_draft_tokens`, and
`spec_decode_num_accepted_tokens` Prometheus counters. Mean acceptance length
uses vLLM's bonus-inclusive convention
`1 + accepted_tokens / drafts`; TPS is actual completion tokens divided by
wall time. Greedy evaluation requires exact target-only/speculative output
parity. Block Verify and Entropy Verify are intentionally not enabled because
they can change output tokens.

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

- The real 910B hidden parity, two-rank FSDP2 resume, export load, and serving
  ABI gates must still be run on the actual CANN/torch-npu/SGLang/vLLM-Ascend
  deployment stack.
- Stock vLLM-Ascend serving is wired only for DFlash and DSpark. DFlash2 still
  needs a method-specific proposer adapter and is blocked by the benchmark
  launcher until that adapter exists.
- `MASK_TOKEN_ID` is mandatory and must come from the actual tokenizer/runtime;
  EOS or PAD must not be substituted.
- Quantized target token-I/O artifacts, mismatched model revisions, ambiguous
  hidden taps and pre-final-norm substitutes fail closed.
