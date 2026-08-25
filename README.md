# GLM-5.2 DFlash2: two-pass SGLang data pipeline

服务器迁移与 AI 接手请先阅读 [`AI_HANDOFF.md`](AI_HANDOFF.md)。

This repository prepares `cbyzju/vibe_coding_630k` for a GLM-5.2 DFlash2
drafter on Ascend 910B. GLM-5.2 weights and generated caches are external.

The production path is deliberately two-pass:

1. **Stage A — sampled agent trajectory construction.** SGLang serves GLM-5.2
   with the model's production sampling policy by default
   (`temperature=1.0`, `top_p=0.95`, top-k disabled). Normal
   rows run multi-round coding-agent rollouts with real bounded workspace tools
   and optional web tools. The two Open-SWE prefix routes deliberately restore
   the original complete trajectories, matching the reference
   `vibe_coding_qwen38.py` behavior. Each record is validated and rendered to
   canonical `input_ids` plus a DFlash token-position `loss_mask`.
2. **Stage B — hidden extraction.** The rollout server has exited. A fresh
   SGLang internal `ModelRunner` teacher-forces the immutable **sampled** token
   IDs and
   captures logical GLM layers `1,20,38,56,75` in one prefill. Rank 0 writes a
   resumable packed BF16 cache.

This avoids keeping two GLM replicas resident and avoids the ambiguity of
trying to extract arbitrary intermediate layers through SGLang's public HTTP
hidden-state response.

## Data contract

Stage A stores the complete canonical message list, ordered tool schemas,
actual tool events/results, structured reasoning/tool calls, generation start
message index, source/model metadata, per-round SGLang prompt token IDs (and
response IDs if a vendor extension exposes them), final rendered `input_ids`,
and `loss_mask`.

`loss_mask[i] = 1` means token position `i` belongs to an assistant turn
generated in this rollout. It is **not shifted left** like ordinary AR CE:
SpecForge DFlash samples anchors and labels at target-token positions. System,
source history, user and tool-observation tokens are zero; the last token is
always zero.

Stage A asks SGLang for each round's prompt IDs, re-renders the round with the
same tokenizer, ordered tools, and chat-template kwargs, and requires exact
equality. The standard SGLang OpenAI chat endpoint does not expose generated
IDs, so assistant turns are frozen by canonical replay after structured
reasoning/tool-call parsing. It also enforces a hard `MAX_SEQUENCE_TOKENS`;
no silent truncation is allowed.

Per-sample failures are appended and fsynced to
`<trajectory>.errors.jsonl`; generation continues with later samples. A full
shard is marked `frozen` only when every owned ID is committed and no unresolved
error remains. `MAX_SAMPLES` produces an explicit `partial` manifest instead
of masquerading as a complete training shard.

## Packed hidden cache

Each cache has:

```text
manifest.json
index.jsonl
segment-00000/
  input_ids.bin       # little-endian int64
  loss_mask.bin       # uint8
  hidden_states.bin   # little-endian raw BF16 words
```

For each sample, all streams are written and fsynced before its index line is
written and fsynced. The index line is the commit point. The manifest can be
rebuilt from the index after interruption. Each stream slice has SHA-256.

`PackedHiddenDataset` returns:

- `input_ids`: `[T]`, int64;
- `loss_mask`: `[T]`, bool;
- archival `layer_hidden_states`: `[T,5,6144]`, BF16;
- SpecForge-ready `hidden_states`: `[T,30720]`, BF16.

The collator right-pads variable-length examples without shifting the mask.

## 0. Environment

Use the official Ascend SGLang image documented for GLM-5.2, currently
`quay.io/ascend/sglang:cann9.0.0-910b-v0.5.16`, together with the host Ascend
driver/firmware and the documented `/dev/davinci*`, manager and HDC mounts.
Do not install generic CUDA wheels over a working vendor environment. Source
the image's CANN/ATB environment before either stage.

The launchers explicitly select `--device npu --attention-backend ascend`.
`MOE_A2A_BACKEND=deepep DEEPEP_MODE=auto` can be enabled in the official image;
they are exposed rather than forced so a vendor image without DeepEP still has
a functional fallback. The internal runner requires a SGLang build containing:

- `benchmark.one_batch.load_model` or the legacy
  `bench_one_batch.load_model` layout;
- `CaptureHiddenMode.FULL`;
- GLM-5.2/DeepSeek `set_eagle3_layers_to_capture` (or a vendor-equivalent
  `set_dflash_layers_to_capture` alias);
- `LogitsProcessorOutput.hidden_states` packed auxiliary states.

The development reference was SGLang commit
`f2ef826f0caefc40e33aa676124bba80b9092887`. A vendor Ascend fork may use a
different commit, but the real gate below is mandatory and its version is
recorded in `manifest.json`.

GLM-5.2 is a large MoE target. This project targets the **BF16 checkpoint**.
The official deployment guide reports roughly 1.51 TB for BF16 weights, so on
64 GB 910B cards use at least `TP_SIZE=32 NNODES=2` (plus activation/KV-cache
headroom). Quantized target I/O is not accepted by the offline trainer.

These are model-weight requirements, before KV cache and captured hidden-state
memory. Do not attempt BF16 TP16 on 16 x 64 GB cards.

Install only the CPU/data dependencies into the vendor image:

```bash
pip install -r requirements-data.txt
```

## 1. Download/validate source data

```bash
bash scripts/download_dataset.sh
bash scripts/smoke_no_model.sh
```

## 2. Stage A — rollout trajectories

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
target model. A single-host example is only appropriate when that host has
enough aggregate NPU memory:

```bash
MODEL_PATH=/shared/models/GLM-5.2-bf16 \
OUTPUT_JSONL=/shared/out/trajectories.jsonl \
TP_SIZE=32 \
MOE_A2A_BACKEND=deepep DEEPEP_MODE=auto \
bash scripts/run_stage_a_trajectories.sh
```

## 3. Stage B — selected-layer hidden states

BF16 two-node TP+EP example (run once per node, changing `NODE_RANK`):

```bash
MODEL_PATH=/shared/models/GLM-5.2 \
TRAJECTORY_JSONL=/shared/out/trajectories-shard-0-of-1.jsonl \
OUTPUT_DIR=/shared/out/hidden-shard-0-of-1 \
TP_SIZE=32 EP_SIZE=32 NNODES=2 NODE_RANK=0 \
DIST_INIT_ADDR=<rank-0-host> NCCL_PORT=29500 \
MOE_A2A_BACKEND=deepep DEEPEP_MODE=auto \
bash scripts/run_stage_b_hidden.sh
```

For multi-node TP, run the same command on every node with a shared
`DIST_INIT_ADDR`, `NCCL_PORT`, `NNODES`, and different `NODE_RANK`. Only global
TP rank 0 writes. DP and PP are rejected by this first extractor; TP plus EP is
supported. Chunked prefill and graph capture are forcibly disabled so every
trajectory is one exact teacher-forced prefill.

For BF16 on two 16-card 910B nodes, launch the same command on both nodes with
`TP_SIZE=32 EP_SIZE=32 NNODES=2`, a shared `DIST_INIT_ADDR`/`NCCL_PORT`, and
`NODE_RANK=0` or `1`. `NCCL_PORT` is the historical SGLang CLI field name; the
Ascend runtime still uses HCCL underneath.

The convenience wrapper runs local Stage A and Stage B sequentially:

```bash
MODEL_PATH=/shared/models/GLM-5.2 \
TRAJECTORY_JSONL=/shared/out/trajectories.jsonl \
HIDDEN_OUTPUT_DIR=/shared/out/hidden \
bash scripts/build_two_pass_cache.sh
```

## 4. Validate

```bash
python tools/validate_hidden_cache.py \
  --cache-dir /shared/out/hidden-shard-0-of-1 \
  --expected-layer-ids 1,20,38,56,75 \
  --full-scan
```

Before scaling beyond 1–2 samples on 910B, also compare every captured layer
slice against a direct reference forward/hook on the same short token IDs.
The required output is finite BF16 `[T,5,6144]`, and the SGLang physical
capture mapping should be `[2,21,39,57,76]` for this GLM implementation.
Shape-only validation is insufficient.

## 5. Offline DFlash2 training (target backbone is not loaded)

Training consumes the frozen packed cache plus a small token-I/O artifact.  It
does **not** instantiate GLM-5.2: the only target tensors present are the frozen
`embed_tokens.weight` and `lm_head.weight`.  Extract them once from dense
BF16/FP16/FP32 safetensors (quantized ModelSlim tensors are rejected):

```bash
MODEL_PATH=/shared/models/GLM-5.2-bf16 \
TARGET_IO_DIR=/shared/out/glm52-target-io \
bash scripts/extract_glm52_io.sh
```

The fixed first recipe is a Qwen3-shaped five-layer draft at hidden width 6144,
block size 16, 64 uniformly sampled anchors, gamma 7, two-tap group-16 dynamic
convolution, and a rank-256/top-16 DFlash2 selector.  On one eight-NPU 910B
node:

```bash
CACHE_DIR=/shared/out/hidden-frozen \
TARGET_IO_DIR=/shared/out/glm52-target-io \
OUTPUT_DIR=/shared/out/glm52-dflash2-b16 \
MASK_TOKEN_ID=<real-glm-mask-id> \
PAD_TOKEN_ID=154820 \
NUM_NPUS=8 \
bash scripts/train_glm52_dflash2_910b.sh
```

For multi-node training, launch the same command on every node with the same
`NNODES`, `MASTER_ADDR` and `MASTER_PORT`, and set `NODE_RANK` uniquely.  For
example, a two-node job uses `NNODES=2 NODE_RANK=0` on the master and
`NNODES=2 NODE_RANK=1` on the peer.  `NUM_NPUS` is the per-node process count.

The default optimizer follows the DFlash2 recipe: AdamW at `6e-4`,
`betas=(0.9,0.95)`, zero weight decay, 1,000 warmup optimizer steps and cosine
decay. Resume rejects changes to the cache/target-I/O identity or any
scheduler-, sampling-, batching- or architecture-relevant setting instead of
silently treating them as an exact continuation.

`MASK_TOKEN_ID` is mandatory and is never inferred from EOS or padding.  Set
`RESUME=/shared/out/.../step-N` to restore model, optimizer, scheduler, data
cursor, framework RNG and anchor RNG.  Checkpoints are written only after an
optimizer step.  The final `export/` is draft-only and deliberately excludes
the frozen target I/O tensors, which the serving runtime must share with its
target.

Do not install a generic PyPI PyTorch over the Ascend image.  Install the
vendor-matched PyTorch/torch-npu/CANN tuple first, then the small Python-only
set in `requirements-train.txt`.

Before a full run, execute both local and hardware gates:

```bash
PY=/path/to/vendor/python bash scripts/smoke_train_no_npu.sh

CACHE_DIR=/shared/gate/hidden-frozen \
TARGET_IO_DIR=/shared/out/glm52-target-io \
MASK_TOKEN_ID=<real-glm-mask-id> \
bash scripts/gate_train_2rank_910b.sh
```

The two-rank gate runs both an uninterrupted two-step job and a one-step
save/resume job. It only writes `"passed": true` after exports are bitwise
equal and finite, logged losses/gradients are finite, checkpoints are complete,
and the frozen target-I/O file hash is unchanged. Serving compatibility is a
separate gate: capture `backbone_logits`, top-k IDs/scores, pair scores and the
final selected path from both this trainer and the target SGLang-on-Ascend
runtime, then run `tools/compare_sglang_runtime.py`.  Until both gates pass on
the deployment stack, the artifact is training-ready but not claimed to be
serving-compatible.

This hardware gate is the remaining compatibility boundary. Public SGLang
serving of GLM-5.2 on Ascend is supported, but Stage B intentionally uses the
internal standalone `ModelRunner` because the HTTP API cannot return five
arbitrary intermediate layers. The bridge supports both known
`ForwardBatch.init_new` signatures; an untested future vendor fork may still
need a small adapter. A successful `/health` response from Stage A alone does
not validate Stage B hidden capture.

A disposable two-sample hardware gate is:

```bash
MAX_SAMPLES=2 TRAJECTORY_JSONL=/shared/gate/trajectories.jsonl \
HIDDEN_OUTPUT_DIR=/shared/gate/hidden \
MODEL_PATH=/shared/models/GLM-5.2 bash scripts/build_two_pass_cache.sh

python tools/validate_hidden_cache.py \
  --cache-dir /shared/gate/hidden \
  --allow-building-cache --full-scan
```

Keep this gate in a separate output directory. Its Stage-A manifest is
`partial` and its hidden cache remains `building`; neither can be mistaken for
the 630K production cache.

## Storage warning

Five BF16 layers at hidden size 6144 cost 61,440 bytes/token. At 630K samples,
an average of 1,000 tokens is about **38.7 TB**; 2,000 tokens is about
**77.4 TB**, before filesystem/index overhead. Capacity and retention policy
must be settled before a full extraction.

Very long individual trajectories are also a memory risk in Stage B because
the correctness path disables chunked prefill. Start with the 1–2 sample gate,
then measure the longest selected trajectory before scaling. Supporting 131K
tokens under a tight memory budget would require a separately validated
KV-cached chunked extractor; the current code does not silently approximate it.

The repository exposes only the SGLang two-pass production path. The obsolete
vLLM response-only generator and its separate output schema were removed to
avoid accidentally producing a cache that Stage B cannot consume.
