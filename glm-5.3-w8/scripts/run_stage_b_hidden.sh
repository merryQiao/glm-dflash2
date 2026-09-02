#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
: "${MODEL_PATH:?set MODEL_PATH to the ModelSlim GLM-5.3 W8A8 checkpoint}"
: "${TRAJECTORY_JSONL:?set TRAJECTORY_JSONL to the immutable Stage-A JSONL}"
: "${TARGET_IO_DIR:?set TARGET_IO_DIR to the extracted BF16 target I/O}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for the packed hidden cache}"
SCRATCH_ROOT=${SCRATCH_ROOT:-${OUTPUT_DIR}.connector_tmp}
TP_SIZE=${TP_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_SHARD_MIB=${MAX_SHARD_MIB:-512}

extra=()
[[ -n "${MAX_SAMPLES:-}" ]] && extra+=(--max-samples "$MAX_SAMPLES")

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" "$ROOT/tools/extract_hidden_vllm_ascend.py" \
  --trajectory-jsonl "$TRAJECTORY_JSONL" \
  --model-path "$MODEL_PATH" \
  --target-io "$TARGET_IO_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --scratch-root "$SCRATCH_ROOT" \
  --tp-size "$TP_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-shard-mib "$MAX_SHARD_MIB" \
  "${extra[@]}" "$@"
