#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the local GLM-5.2 checkpoint}
TRAJECTORY_JSONL=${TRAJECTORY_JSONL:?set TRAJECTORY_JSONL to a frozen Stage A shard}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/glm52_hidden_cache}
TP_SIZE_VALUE=${TP_SIZE:-32}
NNODES_VALUE=${NNODES:-1}

if (( TP_SIZE_VALUE % NNODES_VALUE != 0 )); then
  echo "TP_SIZE must be divisible by NNODES" >&2
  exit 2
fi
if (( NNODES_VALUE > 1 )); then
  : "${DIST_INIT_ADDR:?set DIST_INIT_ADDR for multi-node Stage B}"
  : "${NCCL_PORT:?set NCCL_PORT for multi-node Stage B}"
fi

args=(
  "$ROOT/tools/extract_hidden_sglang.py"
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --dtype "${DTYPE:-bfloat16}"
  --device "${DEVICE:-npu}"
  --attention-backend "${ATTENTION_BACKEND:-ascend}"
  --tp-size "$TP_SIZE_VALUE"
  --ep-size "${EP_SIZE:-$TP_SIZE_VALUE}"
  --nnodes "$NNODES_VALUE"
  --node-rank "${NODE_RANK:-0}"
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.90}"
  --trajectory-jsonl "$TRAJECTORY_JSONL"
  --output-dir "$OUTPUT_DIR"
  --capture-layer-ids "${CAPTURE_LAYER_IDS:-1,20,38,56,75}"
  --max-segment-gib "${MAX_SEGMENT_GIB:-64}"
  --storage-safety-factor "${STORAGE_SAFETY_FACTOR:-1.05}"
)

if [[ -n "${DIST_INIT_ADDR:-}" ]]; then
  args+=(--dist-init-addr "$DIST_INIT_ADDR")
fi
if [[ -n "${NCCL_PORT:-}" ]]; then
  args+=(--nccl-port "$NCCL_PORT")
fi
if [[ -n "${QUANTIZATION:-}" ]]; then
  args+=(--quantization "$QUANTIZATION")
fi
if [[ -n "${LOAD_FORMAT:-}" ]]; then
  args+=(--load-format "$LOAD_FORMAT")
fi
if [[ -n "${MOE_A2A_BACKEND:-}" ]]; then
  args+=(--moe-a2a-backend "$MOE_A2A_BACKEND")
fi
if [[ -n "${DEEPEP_MODE:-}" ]]; then
  args+=(--deepep-mode "$DEEPEP_MODE")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "${ALLOW_PARTIAL_TRAJECTORIES:-0}" == 1 ]]; then
  args+=(--allow-partial-trajectories)
fi
if [[ "${SKIP_STORAGE_PREFLIGHT:-0}" == 1 ]]; then
  args+=(--skip-storage-preflight)
fi

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "${args[@]}" "$@"
