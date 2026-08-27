#!/usr/bin/env bash
set -euo pipefail

: "${TARGET_MODEL:?set TARGET_MODEL to the GLM-5.2 BF16 checkpoint}"
: "${DRAFTER_EXPORT:?set DRAFTER_EXPORT to a training output/export directory}"
: "${PROMPTS_JSONL:?set PROMPTS_JSONL to a fixed JSONL prompt set}"
: "${OUT_DIR:?set OUT_DIR}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
VLLM_BIN=${VLLM_BIN:-vllm}
TP_SIZE=${TP_SIZE:-16}
PORT=${PORT:-8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-GLM-5.2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_SAMPLES=${MAX_SAMPLES:-0}
MAX_TOKENS=${MAX_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
SEED=${SEED:-42}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-2}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-1800}
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"cudagraph_mode":"PIECEWISE"}'}
mkdir -p "${OUT_DIR}"

readarray -t export_fields < <("${PY}" - "${DRAFTER_EXPORT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "export_manifest.json").read_text())
config = json.loads((root / "config.json").read_text())
print(manifest["runtime_compatibility"])
print(manifest["method"])
print(manifest["num_speculative_tokens"])
print(config["speculators_config"]["algorithm"])
PY
)
RUNTIME_COMPATIBILITY=${export_fields[0]}
METHOD=${export_fields[1]}
NUM_SPECULATIVE_TOKENS=${export_fields[2]}
ALGORITHM=${export_fields[3]}
if [[ "${RUNTIME_COMPATIBILITY}" == "custom-glm52-vllm-ascend-adapter-required" ]]; then
  echo "GLM-5.2 runtime adapter has not passed its offline/runtime parity gate; serving is intentionally blocked." >&2
  exit 2
fi
if [[ "${METHOD}" != "${ALGORITHM}" ]]; then
  echo "export method/config mismatch" >&2
  exit 2
fi
if (( NUM_SPECULATIVE_TOKENS > 15 )); then
  echo "vLLM-Ascend requires num_speculative_tokens + 1 <= 16" >&2
  exit 2
fi

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((TP_SIZE - 1)))
  export ASCEND_RT_VISIBLE_DEVICES
fi

SERVER_PID=""
stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

wait_ready() {
  local deadline=$((SECONDS + SERVER_TIMEOUT))
  until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "vLLM server exited before becoming healthy" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "timed out waiting for vLLM server" >&2
      return 1
    fi
    sleep 2
  done
}

run_server() {
  local mode=$1
  local log="${OUT_DIR}/${mode}-server.log"
  local -a command=(
    "${VLLM_BIN}" serve "${TARGET_MODEL}"
    --host 127.0.0.1 --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tensor-parallel-size "${TP_SIZE}"
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --trust-remote-code
    --compilation-config "${COMPILATION_CONFIG}"
  )
  if [[ "${mode}" == "speculative" ]]; then
    local speculative_config
    speculative_config=$("${PY}" - "${METHOD}" "${DRAFTER_EXPORT}" "${NUM_SPECULATIVE_TOKENS}" <<'PY'
import json, sys
print(json.dumps({"method": sys.argv[1], "model": sys.argv[2], "num_speculative_tokens": int(sys.argv[3])}))
PY
)
    command+=(--speculative-config "${speculative_config}")
  fi
  if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    local -a extras
    read -r -a extras <<<"${VLLM_EXTRA_ARGS}"
    command+=("${extras[@]}")
  fi
  "${command[@]}" >"${log}" 2>&1 &
  SERVER_PID=$!
  wait_ready
}

run_benchmark() {
  local mode=$1
  "${PY}" "${ROOT}/tools/benchmark_vllm_ascend.py" run \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "${SERVED_MODEL_NAME}" \
    --prompts-jsonl "${PROMPTS_JSONL}" \
    --output "${OUT_DIR}/${mode}.json" \
    --max-samples "${MAX_SAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --seed "${SEED}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --timeout "${SERVER_TIMEOUT}"
}

# Sequential launches are intentional: baseline and speculative runs use the
# same devices without cross-job contention or target replicas co-residing.
run_server "baseline"
run_benchmark "baseline"
stop_server
run_server "speculative"
run_benchmark "speculative"
stop_server

compare_args=()
if [[ "${TEMPERATURE}" == "0" || "${TEMPERATURE}" == "0.0" ]]; then
  compare_args+=(--require-exact-outputs)
fi
"${PY}" "${ROOT}/tools/benchmark_vllm_ascend.py" compare \
  --baseline "${OUT_DIR}/baseline.json" \
  --speculative "${OUT_DIR}/speculative.json" \
  --output "${OUT_DIR}/comparison.json" \
  "${compare_args[@]}"
