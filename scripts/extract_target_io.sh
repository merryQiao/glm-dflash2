#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the local GLM-5.3-Flash-BF16 checkpoint}
CACHE_DIR=${CACHE_DIR:?set CACHE_DIR to the immutable GLM-5.3 Stage B cache}
TARGET_IO_DIR=${TARGET_IO_DIR:-$ROOT/outputs/glm53_target_io}

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
  "$ROOT/tools/extract_target_io.py" \
  --model-path "$MODEL_PATH" \
  --hidden-cache-dir "$CACHE_DIR" \
  --output-dir "$TARGET_IO_DIR" \
  "$@"
