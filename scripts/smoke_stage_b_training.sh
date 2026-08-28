#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
from tools.train_drafter_offline import run_tiny_smoke

for method, block_size in (
    ("dflash", 8),
    ("dflash", 16),
    ("dflash2", 8),
    ("dflash2", 16),
    ("dspark", 8),
):
    result = run_tiny_smoke(method=method, block_size=block_size)
    if not result["finite_loss"] or result["optimizer_steps"] != 1:
        raise SystemExit(f"tiny smoke failed: {result}")
    print(result)
PY
