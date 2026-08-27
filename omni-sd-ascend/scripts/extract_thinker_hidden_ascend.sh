#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/common_ascend.sh"

PY="${PY:-python}"
CONFIG="${CONFIG:-${ROOT}/configs/generate_thinker_data.yaml}"
exec "${PY}" -u "${ROOT}/scripts/data/generate_thinker_hidden.py" \
  --config "${CONFIG}" "$@"
