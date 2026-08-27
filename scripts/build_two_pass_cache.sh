#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
TRAJECTORY_JSONL=${TRAJECTORY_JSONL:-$ROOT/outputs/glm52_trajectories/trajectories-shard-0-of-1.jsonl}
HIDDEN_OUTPUT_DIR=${HIDDEN_OUTPUT_DIR:-$ROOT/outputs/glm52_hidden_cache}

echo "DEPRECATED: invoke scripts/generate_trajectories.sh and scripts/extract_hidden_sglang.sh independently" >&2

# Stage A owns and shuts down its temporary HTTP server before returning.
OUTPUT_JSONL="$TRAJECTORY_JSONL" MODEL_PATH="$MODEL_PATH" \
  bash "$ROOT/scripts/run_stage_a_trajectories.sh"

# Stage B starts a fresh internal SGLang ModelRunner and teacher-forces the
# immutable token IDs committed by Stage A.
stage_b_env=(
  "TRAJECTORY_JSONL=$TRAJECTORY_JSONL"
  "OUTPUT_DIR=$HIDDEN_OUTPUT_DIR"
  "MODEL_PATH=$MODEL_PATH"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  stage_b_env+=("ALLOW_PARTIAL_TRAJECTORIES=1")
fi
env "${stage_b_env[@]}" bash "$ROOT/scripts/run_stage_b_hidden.sh"
