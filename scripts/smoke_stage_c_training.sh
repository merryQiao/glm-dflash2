#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

for route in dflash:8 dflash:16 dflash2:8 dflash2:16 dspark:8; do
  method=${route%%:*}
  block_size=${route##*:}
  "${PY}" "${ROOT_DIR}/tools/train_thinker_drafter.py" \
    --method "${method}" \
    --block-size "${block_size}" \
    --device cpu \
    --backend gloo \
    --strategy single \
    --tiny-smoke
done
