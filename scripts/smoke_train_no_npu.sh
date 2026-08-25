#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$ROOT/tools/smoke_train_tiny.py"
