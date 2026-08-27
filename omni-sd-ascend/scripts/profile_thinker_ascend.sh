#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/common_ascend.sh"

PY="${PY:-python}"
CONFIG="${CONFIG:-${ROOT}/configs/generate_thinker_data.yaml}"
exec "${PY}" -u "${ROOT}/inference_qwen3-omni.py" \
  --config "${CONFIG}" "$@"
