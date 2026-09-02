#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
: "${MODEL_PATH:?set MODEL_PATH to the ModelSlim GLM-5.3 W4A8 checkpoint}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for the frozen BF16 target I/O artifact}"

exec "$PY" "$ROOT/tools/extract_target_io.py" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
