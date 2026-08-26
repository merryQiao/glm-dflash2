#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export METHOD=dflash2
exec "$ROOT/scripts/train_glm52_drafter_910b.sh" "$@"
