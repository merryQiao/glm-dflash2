#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET_ID=${DATASET_ID:-cbyzju/vibe_coding_630k}
DATA_DIR=${DATA_DIR:-$ROOT/data/vibe_coding_630k}
MODELSCOPE_BIN=${MODELSCOPE_BIN:-modelscope}
DATA_REVISION=${DATA_REVISION:-d21155bcfc3dcc1433631500460c3f1cfd45f2be}

"$MODELSCOPE_BIN" download \
  --dataset "$DATASET_ID" \
  --revision "$DATA_REVISION" \
  --local_dir "$DATA_DIR" \
  --max-workers "${DOWNLOAD_WORKERS:-8}"

python "$ROOT/tools/validate_dataset.py" \
  --input-dir "$DATA_DIR/processed" \
  --expected-rows 630000 \
  --expected-files 141
