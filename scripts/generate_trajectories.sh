#!/usr/bin/env bash
set -euo pipefail

# Canonical Stage A entry point. It only writes the frozen trajectory artifact.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ROOT/scripts/run_stage_a_trajectories.sh" "$@"
