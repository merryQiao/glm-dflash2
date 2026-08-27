#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/common_ascend.sh"
# Device ownership is exclusively controlled by ASCEND_RT_VISIBLE_DEVICES.

PY="${PY:-python}"
CONFIG="${CONFIG:-${ROOT}/configs/generate_thinker_data.yaml}"
exec "${PY}" -u "${ROOT}/scripts/data/generate_thinker_data_vllm.py" \
  --config "${CONFIG}" "$@"
