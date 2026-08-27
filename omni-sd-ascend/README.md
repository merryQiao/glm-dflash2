# Qwen3-Omni Thinker data pipeline for Ascend 910B A2/A3

This directory contains the production-oriented, two-stage data path for a
Qwen3-Omni speculative drafter. It handles the **Thinker text stream only**;
Talker and Code2Wav are deliberately not loaded.

1. Stage A uses a vLLM-Ascend TP/EP engine to sample responses and stores the
   engine's exact prompt/response token IDs.
2. Stage B replays `prompt_token_ids + response_token_ids` as one teacher-forced
   prompt in vLLM's native `extract_hidden_states` mode. It never samples a new
   training target.
3. Every output shard is atomic and content-addressed. Resume rejects mixed
   model revisions, runtime settings, sampling policies, and corrupt files.

## Runtime baseline

Use an official Qwen3-Omni-capable vLLM-Ascend image, currently:

```text
quay.io/ascend/vllm-ascend:v0.23.0
```

Do not reinstall `torch`, `torch-npu`, `vllm`, or `vllm-ascend` with pip inside
that image. Install only the auxiliary packages:

```bash
python -m pip install -r requirements-data.txt
```

The model is BF16. `quantization: ascend` is rejected by configuration
validation rather than silently changing the experiment.

## Required configuration edits

Edit `configs/generate_thinker_data.yaml` before any model is allocated:

- set immutable `model.revision` and `model.processor_revision` commits;
- set the local model and accepted-condition paths;
- set `runtime.hardware` to `a2` or `a3`;
- make `runtime.tensor_parallel_size` equal the number of chips exposed to the
  process;
- choose output and connector scratch paths on reliable storage.

The supplied hidden layer IDs are `[1, 12, 24, 36, 47]` for the 48-layer
Thinker. The extractor additionally requests synthetic layer ID `48`, which
vLLM defines as the raw post-decoder/pre-final-norm state. The pipeline reads
only `thinker.model.norm.weight` from the same immutable checkpoint and applies
Qwen3-Omni's official RMSNorm order offline. The cache therefore stores:

```text
target_hidden_states       [tokens, 5, 2048]
target_last_hidden_states  [tokens, 2048]
```

The second tensor has the explicit semantics
`post_final_norm_lm_head_input`; an unnormalised decoder output is never used as
a substitute.

## A2/A3 smoke gate

First prepare a small accepted-condition file containing at least one text,
image, audio, and video condition. Point a copy of the YAML at this file and
set its exact `expected_conditions` count. Then run on the destination host.

A2 example (four chips):

```bash
cd /path/to/omni-sd-ascend
ASCEND_HARDWARE=a2 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
CONFIG=$PWD/configs/smoke_a2.yaml \
bash scripts/smoke_ascend.sh
```

A3 example (two chips):

```bash
ASCEND_HARDWARE=a3 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 \
CONFIG=$PWD/configs/smoke_a3.yaml \
bash scripts/smoke_ascend.sh
```

`HCCL_OP_EXPANSION_MODE=AIV` is enabled only for A3. The smoke run must create
`ASCEND_SMOKE_ATTESTATION.json`; missing modalities, token mismatches, NaN/Inf,
or missing final-normalized semantics abort the run.

## Full production run

One Python process owns the full TP/EP engine. **Do not use torchrun.**

```bash
ASCEND_HARDWARE=a2 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
CONFIG=$PWD/configs/generate_thinker_data.yaml \
bash scripts/generate_thinker_trajectories_ascend.sh

ASCEND_HARDWARE=a2 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
CONFIG=$PWD/configs/generate_thinker_data.yaml \
bash scripts/extract_thinker_hidden_ascend.sh
```

For multiple independent engines, expose disjoint chip sets and assign shards
with `WORKER_ID=0..N-1` and `NUM_WORKERS=N`. Each process still owns its own
complete TP group.

## Hard runtime gate

CPU contract tests pass without Ascend hardware, but that does **not** prove
the destination image supports the full extractor path. Before a production
run, the A2/A3 smoke must establish that its pinned vLLM-Ascend build supports:

- `Qwen3OmniMoeThinkerForConditionalGeneration` in core vLLM;
- `method=extract_hidden_states` for this multimodal target;
- `ExampleHiddenStatesConnector` on NPU (including its async host-copy path);
- custom save paths and `[tokens, layers, hidden]` connector layout.

Upstream vLLM documents that chunked prefill is incompatible with hidden-state
extraction, so both engines force it off. If the NPU connector fails, stop at
the smoke gate and patch that version's connector; do not fall back silently to
retokenized responses or mislabeled final hidden states.

## Local verification

```bash
python -m pytest -q
python -m py_compile $(find src scripts/data -name '*.py' -type f)
for script in scripts/*.sh; do bash -n "$script"; done
```
