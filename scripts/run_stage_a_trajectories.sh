#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the local GLM-5.2 checkpoint/tokenizer}
OUTPUT_JSONL=${OUTPUT_JSONL:-$ROOT/outputs/glm52_trajectories/trajectories-shard-${DATA_SHARD_INDEX:-0}-of-${DATA_SHARD_COUNT:-1}.jsonl}

args=(
  "$ROOT/tools/generate_trajectories.py"
  --dataset "${DATASET:-$ROOT/data/vibe_coding_630k}"
  --model-path "$MODEL_PATH"
  --served-model-name "${SERVED_MODEL_NAME:-GLM-5.2}"
  --output-jsonl "$OUTPUT_JSONL"
  --workspace-cache "${WORKSPACE_CACHE:-$ROOT/outputs/workspace_cache}"
  --open-swe-store "${OPEN_SWE_STORE:-$ROOT/outputs/open_swe_original.sqlite}"
  --container-runtime "${CONTAINER_RUNTIME:-docker}"
  --container-network "${CONTAINER_NETWORK:-none}"
  --tp-size "${TP_SIZE:-32}"
  --device "${DEVICE:-npu}"
  --attention-backend "${ATTENTION_BACKEND:-ascend}"
  --port "${PORT:-30000}"
  --context-length "${CONTEXT_LENGTH:-131072}"
  --workers "${WORKERS:-8}"
  --max-running-requests "${MAX_RUNNING_REQUESTS:-2}"
  --max-total-tokens "${MAX_TOTAL_TOKENS:-131072}"
  --max-sequence-tokens "${MAX_SEQUENCE_TOKENS:-131072}"
  --max-new-tokens "${MAX_NEW_TOKENS:-32768}"
  --max-rounds "${MAX_ROUNDS:-32}"
  --episode-retries "${EPISODE_RETRIES:-2}"
  --retry-backoff-seconds "${RETRY_BACKOFF_SECONDS:-1.0}"
  --temperature "${TEMPERATURE:-1.0}"
  --top-p "${TOP_P:-0.95}"
  --top-k "${TOP_K:--1}"
  --chat-template-kwargs-json "${CHAT_TEMPLATE_KWARGS_JSON:-{\"enable_thinking\":true}}"
  --shard-index "${DATA_SHARD_INDEX:-0}"
  --shard-count "${DATA_SHARD_COUNT:-1}"
)

if [[ -n "${ENDPOINT:-}" ]]; then
  args+=(--endpoint "$ENDPOINT")
fi
if [[ -n "${QUANTIZATION:-}" ]]; then
  args+=(--quantization "$QUANTIZATION")
fi
if [[ -n "${MOE_A2A_BACKEND:-}" ]]; then
  args+=(--moe-a2a-backend "$MOE_A2A_BACKEND")
fi
if [[ -n "${DEEPEP_MODE:-}" ]]; then
  args+=(--deepep-mode "$DEEPEP_MODE")
fi
if [[ -n "${WORKSPACE_MAP:-}" ]]; then
  args+=(--workspace-map "$WORKSPACE_MAP")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "${ALLOW_HOST_TESTS:-0}" == 1 ]]; then
  args+=(--allow-host-tests)
fi
if [[ "${NO_CONTAINER_PULL:-0}" == 1 ]]; then
  args+=(--no-container-pull)
fi
if [[ -n "${CONTAINER_CPUS:-}" ]]; then
  args+=(--container-cpus "$CONTAINER_CPUS")
fi
if [[ -n "${CONTAINER_MEMORY:-}" ]]; then
  args+=(--container-memory "$CONTAINER_MEMORY")
fi

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "${args[@]}" "$@"
