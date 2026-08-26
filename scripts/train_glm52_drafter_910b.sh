#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
METHOD=${METHOD:?set METHOD to dflash, dflash2, or dspark}
BLOCK_SIZE=${BLOCK_SIZE:?set BLOCK_SIZE to 8 or 16}
CACHE_DIR=${CACHE_DIR:?set CACHE_DIR to the aligned schema-v2 hidden cache}
TARGET_IO_DIR=${TARGET_IO_DIR:?set TARGET_IO_DIR to the extracted GLM token I/O}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/glm52_${METHOD}_b${BLOCK_SIZE}}
MASK_TOKEN_ID=${MASK_TOKEN_ID:?set the real tokenizer MASK token id explicitly}
NUM_NPUS=${NUM_NPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29681}

case "$METHOD" in
  dflash|dflash2|dspark) ;;
  *) echo "METHOD must be dflash, dflash2, or dspark" >&2; exit 2 ;;
esac
case "$BLOCK_SIZE" in
  8|16) ;;
  *) echo "BLOCK_SIZE must be 8 or 16" >&2; exit 2 ;;
esac
if [[ "$METHOD" == "dspark" && "$BLOCK_SIZE" != "8" ]]; then
  echo "DSpark requires BLOCK_SIZE=8 (one anchor + seven proposals)" >&2
  exit 2
fi
dspark_lr=3e-4
dspark_epochs=1
dspark_gamma=4
if [[ "$METHOD" == "dspark" ]]; then
  default_lr=$dspark_lr
  default_epochs=$dspark_epochs
  default_gamma=$dspark_gamma
else
  default_lr=6e-4
  default_epochs=3
  default_gamma=7
fi
if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == "127.0.0.1" ]]; then
  echo "MASTER_ADDR must identify rank-0 host when NNODES>1" >&2
  exit 2
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  echo "NODE_RANK must be in [0, NNODES)" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
extra=()
if [[ -n "${RESUME:-}" ]]; then extra+=(--resume "$RESUME"); fi
if [[ -n "${MAX_STEPS:-}" ]]; then extra+=(--max-steps "$MAX_STEPS"); fi

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$NUM_NPUS" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$ROOT/tools/train_drafter_offline.py" \
  --method "$METHOD" \
  --cache-dir "$CACHE_DIR" \
  --target-io-dir "$TARGET_IO_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mask-token-id "$MASK_TOKEN_ID" \
  --pad-token-id "${PAD_TOKEN_ID:-0}" \
  --device npu \
  --epochs "${EPOCHS:-$default_epochs}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-8}" \
  --lr "${LR:-$default_lr}" \
  --beta1 "${BETA1:-0.9}" \
  --beta2 "${BETA2:-0.95}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --warmup-steps "${WARMUP_STEPS:-1000}" \
  --save-every "${SAVE_EVERY:-1000}" \
  --log-every "${LOG_EVERY:-10}" \
  --token-chunk-size "${TOKEN_CHUNK_SIZE:-256}" \
  --vocab-chunk-size "${VOCAB_CHUNK_SIZE:-4096}" \
  --block-size "$BLOCK_SIZE" \
  --num-anchors 64 \
  --gamma "${GAMMA:-$default_gamma}" \
  --selector-rank 256 \
  --selector-top-k 16 \
  --markov-rank 256 \
  --hidden-size 6144 \
  --intermediate-size 12288 \
  --num-draft-layers 5 \
  "${extra[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
