#!/usr/bin/env bash
set -euo pipefail

# Canonical Stage B entry point. TRAJECTORY_JSONL must name a frozen Stage A artifact.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${TRAJECTORY_JSONL:?set TRAJECTORY_JSONL to a frozen Stage A shard}"
exec bash "$ROOT/scripts/run_stage_b_hidden.sh" "$@"
