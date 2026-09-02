#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
: "${METHOD:?set METHOD to dspark or dflash2}"
: "${BLOCK_SIZE:?set BLOCK_SIZE to 8 or 16}"
: "${HIDDEN_CACHE:?set HIDDEN_CACHE to a frozen Stage-B cache}"
: "${TARGET_IO_DIR:?set TARGET_IO_DIR to the frozen BF16 target I/O}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for checkpoints and logs}"
: "${MASK_TOKEN_ID:?set MASK_TOKEN_ID from the GLM-5.3 tokenizer}"

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29697}
DEVICE=${DEVICE:-npu}
STRATEGY=${STRATEGY:-fsdp2}
ANCHORS=${ANCHORS:-512}
ANCHOR_CHUNK_SIZE=${ANCHOR_CHUNK_SIZE:-8}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
TRAINING_WINDOW=${TRAINING_WINDOW:-4096}
VOCAB_CHUNK_SIZE=${VOCAB_CHUNK_SIZE:-8192}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1000}
SEED=${SEED:-42}

if [[ "$METHOD" == dspark && "$BLOCK_SIZE" != 8 ]]; then
  echo "DSpark only supports physical block size 8" >&2
  exit 2
fi

extra=()
[[ -n "${RESUME:-}" ]] && extra+=(--resume "$RESUME")
[[ -n "${MAX_STEPS:-}" ]] && extra+=(--max-steps "$MAX_STEPS")
[[ "${GRADIENT_CHECKPOINTING:-0}" == 1 ]] && extra+=(--gradient-checkpointing)

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  "$ROOT/tools/train_offline.py" \
  --method "$METHOD" \
  --block-size "$BLOCK_SIZE" \
  --hidden-cache "$HIDDEN_CACHE" \
  --target-io "$TARGET_IO_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mask-token-id "$MASK_TOKEN_ID" \
  --device "$DEVICE" \
  --strategy "$STRATEGY" \
  --anchors "$ANCHORS" \
  --anchor-chunk-size "$ANCHOR_CHUNK_SIZE" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION" \
  --training-window "$TRAINING_WINDOW" \
  --vocab-chunk-size "$VOCAB_CHUNK_SIZE" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --seed "$SEED" \
  "${extra[@]}" "$@"
