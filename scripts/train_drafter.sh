#!/usr/bin/env bash
set -euo pipefail

# Canonical offline-training entry point. CACHE_DIR must be a frozen Stage B cache.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${CACHE_DIR:?set CACHE_DIR to a frozen Stage B cache}"
exec bash "$ROOT/scripts/train_glm52_drafter_910b.sh" "$@"
