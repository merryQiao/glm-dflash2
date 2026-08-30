#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

: "${MASK_TOKEN_ID:?Set MASK_TOKEN_ID to the target tokenizer mask token ID}"
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

exec python -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE}" \
  "${ROOT_DIR}/tools/train_drafter_offline.py" \
  --device npu \
  --backend hccl \
  --strategy fsdp2 \
  --dtype bfloat16 \
  --mask-token-id "${MASK_TOKEN_ID}" \
  "$@"
