#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET=${1:?usage: apply_patch.sh /path/to/vllm-ascend [--dry-run]}
MODE=${2:-}
[[ -d "$TARGET/.git" ]] || { echo "not a vLLM-Ascend git tree: $TARGET" >&2; exit 2; }

expected=$(awk -F= '$1=="vllm_ascend_commit" {print $2}' "$ROOT/VERSION")
actual=$(git -C "$TARGET" rev-parse HEAD)
if [[ "$expected" == PIN_ON_ASCEND_HOST ]]; then
  echo "VERSION is not pinned; record the validated vLLM-Ascend commit first" >&2
  exit 2
fi
[[ "$actual" == "$expected" ]] || { echo "vLLM-Ascend commit mismatch: $actual != $expected" >&2; exit 2; }
if [[ "$MODE" == --dry-run ]]; then
  echo "compatible vLLM-Ascend tree: $actual"
  exit 0
fi
echo "DFlash2 uses method=custom_class; install by adding this repository to PYTHONPATH." >&2
