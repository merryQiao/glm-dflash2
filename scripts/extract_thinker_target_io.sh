#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
: "${MODEL_PATH:?set MODEL_PATH to Qwen3-Omni-30B-A3B-Instruct BF16}"
: "${HIDDEN_CACHE_DIR:?set HIDDEN_CACHE_DIR to the verified Stage B v3 cache}"
TARGET_IO_DIR=${TARGET_IO_DIR:-${ROOT_DIR}/outputs/thinker_target_io}

PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" \
  "${ROOT_DIR}/tools/extract_thinker_target_io.py" \
  --model-path "${MODEL_PATH}" \
  --hidden-cache-dir "${HIDDEN_CACHE_DIR}" \
  --output-dir "${TARGET_IO_DIR}" \
  "$@"
