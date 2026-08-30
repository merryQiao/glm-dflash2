# Qwen3-Omni Stage B + Stage C for Ascend

This branch is a self-contained, downloadable package for the **Thinker text
stream** of `Qwen3-Omni-30B-A3B-Instruct`:

- **Stage B** replays immutable Stage A trajectories with vLLM-Ascend and
  archives target hidden states plus exact multimodal position IDs.
- **Stage C** trains an embedding/head-free DFlash, DFlash2, or DSpark
  drafter offline on Ascend 910B A2/A3. The 30B target is not loaded during
  training.

Stage A trajectory generation, inference profiling, Talker/Code2Wav training,
GLM code, model weights, datasets, caches, and historical experiments are
intentionally not included. Downloading this GitHub branch therefore produces
only the code needed after trajectories already exist.

Read [`docs/STAGE_B_STAGE_C_GUIDE.md`](docs/STAGE_B_STAGE_C_GUIDE.md) first.
It explains the artifact contracts, the Stage B changes made for Stage C,
every supported objective, exact commands, and remaining hardware gates.

## Pipeline

```text
immutable Stage A trajectory shards
  prompt_token_ids + response_token_ids + multimodal messages
                    |
                    v
Stage B: exact-token teacher-forced Thinker replay (vLLM-Ascend)
  input_ids                       [T]
  loss_mask                       [T]
  target_hidden_states            [T, 5, 2048]
  target_last_hidden_states       [T, 2048]
  position_ids                    [T, 3]
                    |
                    +--> extract frozen target embedding + lm_head once
                    |
                    v
Stage C: BF16 + HCCL + FSDP2 offline training
  DFlash B8/B16 | DFlash2 B8/B16 | DSpark B8
```

## Quick start

Install only the lightweight Python dependencies into the official Ascend
runtime. Do **not** replace that image's PyTorch, torch-npu, vLLM, or
vLLM-Ascend packages.

```bash
pip install -r requirements.txt
```

Edit model, trajectory, output, revision, and runtime fields in
`configs/generate_thinker_data.yaml`, then extract the v3 hidden cache:

```bash
ASCEND_HARDWARE=a2 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
CONFIG=$PWD/configs/generate_thinker_data.yaml \
bash scripts/extract_thinker_hidden_ascend.sh
```

Extract the two frozen target-I/O tensors and bind them to that cache:

```bash
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
HIDDEN_CACHE_DIR=/path/to/thinker_hidden \
TARGET_IO_DIR=/path/to/thinker_target_io \
bash scripts/extract_thinker_target_io.sh
```

Run all five CPU functional routes:

```bash
bash scripts/smoke_stage_c_training.sh
```

Launch production Stage C training:

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

## Verification scope

The package has CPU contract tests and real forward/backward/AdamW smoke
coverage for all five routes. Production readiness still requires a real A2/A3
run proving the target hidden connector, HCCL/FSDP2, BF16 SDPA, HBM fit, and a
checkpoint save/resume cycle. The code fails closed on v2 caches, identity
mismatches, quantized or incompatible target I/O, and incomplete checkpoints.
