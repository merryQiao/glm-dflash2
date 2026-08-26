#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
METHOD=${METHOD:-dflash2}
CACHE_DIR=${CACHE_DIR:?set CACHE_DIR to a frozen 1-2 sample gate cache}
TARGET_IO_DIR=${TARGET_IO_DIR:?set TARGET_IO_DIR}
MASK_TOKEN_ID=${MASK_TOKEN_ID:?set MASK_TOKEN_ID}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/gate_train_2rank_910b}
MASTER_PORT=${MASTER_PORT:-29682}

if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "gate output must be new or empty: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR/direct" "$OUTPUT_DIR/resumed"
TARGET_HASH_BEFORE=$(sha256sum "$TARGET_IO_DIR/model.safetensors" | awk '{print $1}')

run_train() {
  local port=$1
  shift
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" -m torch.distributed.run --nproc_per_node=2 --master_port="$port" \
    "$ROOT/tools/train_drafter_offline.py" \
    --method "$METHOD" \
    --cache-dir "$CACHE_DIR" --target-io-dir "$TARGET_IO_DIR" \
    --mask-token-id "$MASK_TOKEN_ID" --device npu --epochs 2 \
    --grad-accum 1 --save-every 1 --log-every 1 "$@"
}

# Compare an uninterrupted two-step run against one step plus exact resume.
run_train "$MASTER_PORT" --output-dir "$OUTPUT_DIR/direct" --max-steps 2
run_train "$((MASTER_PORT + 1))" --output-dir "$OUTPUT_DIR/resumed" --max-steps 1
run_train "$((MASTER_PORT + 2))" --output-dir "$OUTPUT_DIR/resumed" \
  --resume "$OUTPUT_DIR/resumed/step-1" --max-steps 2

TARGET_HASH_AFTER=$(sha256sum "$TARGET_IO_DIR/model.safetensors" | awk '{print $1}')
GATE_OUTPUT="$OUTPUT_DIR" GATE_METHOD="$METHOD" TARGET_HASH_BEFORE="$TARGET_HASH_BEFORE" \
TARGET_HASH_AFTER="$TARGET_HASH_AFTER" "$PY" - <<'PY'
import json, math, os
from pathlib import Path
import torch
from safetensors.torch import load_file

root = Path(os.environ["GATE_OUTPUT"])
direct = load_file(root / "direct/export/model.safetensors", device="cpu")
resumed = load_file(root / "resumed/export/model.safetensors", device="cpu")
compare = set(direct) == set(resumed)
finite = True
max_error = 0.0
for key in direct:
    finite &= bool(torch.isfinite(direct[key]).all() and torch.isfinite(resumed[key]).all())
    compare &= bool(torch.equal(direct[key], resumed[key]))
    if direct[key].numel():
        max_error = max(max_error, float((direct[key].float() - resumed[key].float()).abs().max()))

logs_finite = True
for name in ("direct", "resumed"):
    for line in (root / name / "train.jsonl").read_text().splitlines():
        record = json.loads(line)
        logs_finite &= all(
            math.isfinite(float(value))
            for value in record.values()
            if isinstance(value, (int, float))
        )

checkpoints = all((root / name / "step-2/COMPLETE").is_file() for name in ("direct", "resumed"))
frozen = os.environ["TARGET_HASH_BEFORE"] == os.environ["TARGET_HASH_AFTER"]
passed = bool(compare and finite and logs_finite and checkpoints and frozen)
payload = {
    "schema": "glm-unified-910b-train-gate-v3",
    "method": os.environ["GATE_METHOD"],
    "passed": passed,
    "uninterrupted_resume_exact": bool(compare),
    "all_export_tensors_finite": bool(finite),
    "all_logged_values_finite": bool(logs_finite),
    "frozen_target_io_hash_unchanged": bool(frozen),
    "max_export_abs_error": max_error,
}
(root / "gate-result.json").write_text(json.dumps(payload, indent=2) + "\n")
if not passed:
    raise SystemExit(1)
PY
