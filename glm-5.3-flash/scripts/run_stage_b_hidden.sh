#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the local GLM-5.3-Flash-BF16 checkpoint}
TRAJECTORY_JSONL=${TRAJECTORY_JSONL:?set TRAJECTORY_JSONL to a frozen Stage A shard}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/glm53_hidden_cache}
TP_SIZE_VALUE=${TP_SIZE:-16}
NNODES_VALUE=${NNODES:-1}

if (( TP_SIZE_VALUE % NNODES_VALUE != 0 )); then
  echo "TP_SIZE must be divisible by NNODES" >&2
  exit 2
fi
if (( NNODES_VALUE > 1 )); then
  : "${DIST_INIT_ADDR:?set DIST_INIT_ADDR for multi-node Stage B}"
fi

args=(
  "$ROOT/tools/extract_hidden_sglang.py"
  --model-path "$MODEL_PATH"
  --trajectory-jsonl "$TRAJECTORY_JSONL"
  --output-dir "$OUTPUT_DIR"
  --trust-remote-code
  --dtype "${DTYPE:-bfloat16}"
  --device "${DEVICE:-npu}"
  --attention-backend "${ATTENTION_BACKEND:-ascend}"
  --model-runner torch
  --tp-size "$TP_SIZE_VALUE"
  --ep-size "${EP_SIZE:-$TP_SIZE_VALUE}"
  --dp-size 1
  --pp-size 1
  --nnodes "$NNODES_VALUE"
  --node-rank "${NODE_RANK:-0}"
  --dist-init-addr "${DIST_INIT_ADDR:-127.0.0.1}"
  --dist-port "${DIST_PORT:-29500}"
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.90}"
  --capture-layer-ids "${CAPTURE_LAYER_IDS:-1,11,22,32,42}"
  --max-segment-gib "${MAX_SEGMENT_GIB:-64}"
)

if [[ -n "${MAX_SAMPLES:-}" ]]; then args+=(--max-samples "$MAX_SAMPLES"); fi
if [[ "${ALLOW_SMOKE_UNVERIFIED:-0}" == 1 ]]; then
  args+=(--allow-smoke-unverified --smoke-max-samples "${SMOKE_MAX_SAMPLES:-50}")
fi
if [[ -n "${QUANTIZATION:-}" ]]; then args+=(--quantization "$QUANTIZATION"); fi
if [[ -n "${LOAD_FORMAT:-}" ]]; then args+=(--load-format "$LOAD_FORMAT"); fi
if [[ -n "${MOE_A2A_BACKEND:-}" ]]; then args+=(--moe-a2a-backend "$MOE_A2A_BACKEND"); fi
if [[ -n "${DEEPEP_MODE:-}" ]]; then args+=(--deepep-mode "$DEEPEP_MODE"); fi

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "${args[@]}" "$@"
