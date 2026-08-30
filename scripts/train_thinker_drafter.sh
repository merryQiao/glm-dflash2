#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=common_ascend.sh
source "${ROOT_DIR}/scripts/common_ascend.sh"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

NPROC_PER_NODE=${NPROC_PER_NODE:-$(tr ',' '\n' <<<"${ASCEND_RT_VISIBLE_DEVICES}" | wc -l)}
PY=${PY:-python}

exec "${PY}" -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE}" \
  "${ROOT_DIR}/tools/train_thinker_drafter.py" \
  --device npu \
  --backend hccl \
  --strategy fsdp2 \
  "$@"
