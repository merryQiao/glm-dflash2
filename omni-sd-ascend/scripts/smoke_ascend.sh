#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/common_ascend.sh"

PY="${PY:-python}"
CONFIG="${CONFIG:-${ROOT}/configs/generate_thinker_data.yaml}"

"${PY}" -u "${ROOT}/scripts/data/generate_thinker_data_vllm.py" \
  --config "${CONFIG}" --max-shards 1
"${PY}" -u "${ROOT}/scripts/data/generate_thinker_hidden.py" \
  --config "${CONFIG}" --max-shards 1
"${PY}" -u "${ROOT}/scripts/data/attest_ascend_smoke.py" --config "${CONFIG}"
