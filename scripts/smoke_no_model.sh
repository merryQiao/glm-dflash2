#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
SMOKE_DIR=${SMOKE_DIR:-$ROOT/outputs/smoke_no_model}
rm -rf "$SMOKE_DIR"

PYTHONPATH="$ROOT/src" "$PY" -m unittest discover -s "$ROOT/tests" -v
PYTHONPATH="$ROOT/src" "$PY" "$ROOT/tools/mock_two_pass_smoke.py" --output-dir "$SMOKE_DIR"
PYTHONPATH="$ROOT/src" "$PY" "$ROOT/tools/validate_hidden_cache.py" \
  --cache-dir "$SMOKE_DIR/hidden" \
  --expected-samples 1 \
  --full-scan
bash "$ROOT/scripts/smoke_train_no_npu.sh"
