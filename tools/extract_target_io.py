#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm_dflash2.target_io import extract_target_io  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen GLM token I/O tensors without loading the model")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embed-key")
    parser.add_argument("--lm-head-key")
    args = parser.parse_args()
    manifest = extract_target_io(
        args.model_dir,
        args.output_dir,
        embed_key=args.embed_key,
        lm_head_key=args.lm_head_key,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
