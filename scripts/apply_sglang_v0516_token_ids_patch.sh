#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
PATCH_FILE="$ROOT/patches/sglang-v0.5.16-chat-token-ids.patch"

readarray -t runtime < <("$PY" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import sglang

print(version("sglang"))
print(Path(sglang.__file__).resolve().parent.parent)
PY
)
version=${runtime[0]}
if [[ "$version" != "0.5.16" ]]; then
  echo "refusing to patch SGLang $version; this patch is pinned to 0.5.16" >&2
  exit 2
fi

SGLANG_ROOT=${SGLANG_ROOT:-${runtime[1]}}
if [[ -f "$SGLANG_ROOT/python/sglang/srt/entrypoints/openai/protocol.py" ]]; then
  strip=1
  protocol="$SGLANG_ROOT/python/sglang/srt/entrypoints/openai/protocol.py"
elif [[ -f "$SGLANG_ROOT/sglang/srt/entrypoints/openai/protocol.py" ]]; then
  strip=2
  protocol="$SGLANG_ROOT/sglang/srt/entrypoints/openai/protocol.py"
else
  echo "cannot locate SGLang protocol.py under $SGLANG_ROOT" >&2
  exit 2
fi

if grep -q 'response_token_ids: Optional\[List\[int\]\]' "$protocol"; then
  echo "SGLang chat token-ID compatibility is already installed"
  exit 0
fi

patch --dry-run --batch --forward -p"$strip" -d "$SGLANG_ROOT" < "$PATCH_FILE"
patch --batch --forward -p"$strip" -d "$SGLANG_ROOT" < "$PATCH_FILE"
echo "installed SGLang 0.5.16 non-stream chat token-ID compatibility"
