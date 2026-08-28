#!/usr/bin/env python3
"""Extract frozen GLM-5.3 embedding and LM-head tensors for offline training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm53_drafters.target_io import extract_target_io  # noqa: E402
from glm53_drafters.hidden_cache import validate_frozen_hidden_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-cache-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_manifest = validate_frozen_hidden_cache(args.hidden_cache_dir)
    manifest = extract_target_io(
        args.model_path,
        args.output_dir,
        cache_manifest=cache_manifest,
    )
    import json

    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
