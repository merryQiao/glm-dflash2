#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the dense BF16/FP16/FP32 GLM-5.2 checkpoint}
TARGET_IO_DIR=${TARGET_IO_DIR:-$ROOT/outputs/glm52_target_io}

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$ROOT/tools/extract_target_io.py" \
  --model-dir "$MODEL_PATH" \
  --output-dir "$TARGET_IO_DIR"
