#!/usr/bin/env bash

set -euo pipefail

: "${ASCEND_RT_VISIBLE_DEVICES:?set ASCEND_RT_VISIBLE_DEVICES to the allocated chips}"
ASCEND_HARDWARE="${ASCEND_HARDWARE:-a2}"
case "${ASCEND_HARDWARE}" in
  a2|a3) ;;
  *) echo "ASCEND_HARDWARE must be a2 or a3" >&2; exit 2 ;;
esac

# AIV is the documented A3 HCCL expansion mode. Do not force it on A2.
if [[ "${ASCEND_HARDWARE}" == "a3" ]]; then
  export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
fi
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

device_count="$(tr ',' '\n' <<<"${ASCEND_RT_VISIBLE_DEVICES}" | sed '/^[[:space:]]*$/d' | wc -l)"
if [[ "${device_count}" -le 0 ]]; then
  echo "ASCEND_RT_VISIBLE_DEVICES is empty" >&2
  exit 2
fi
if [[ -n "${TP_SIZE:-}" && "${device_count}" -ne "${TP_SIZE}" ]]; then
  echo "visible device count (${device_count}) != TP_SIZE (${TP_SIZE})" >&2
  exit 2
fi
