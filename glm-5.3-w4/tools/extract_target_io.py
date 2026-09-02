#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm53_w4.target_io import extract_w4a8_target_io  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export frozen BF16 embedding/lm_head from ModelSlim GLM-5.3 W4A8."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embed-key")
    parser.add_argument("--lm-head-key")
    args = parser.parse_args()
    manifest = extract_w4a8_target_io(
        args.model_path,
        args.output_dir,
        embed_key=args.embed_key,
        lm_head_key=args.lm_head_key,
    )
    print(
        f"exported W4A8 target I/O: vocab={manifest['vocab_size']} "
        f"hidden={manifest['hidden_size']} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
