from __future__ import annotations

import argparse
from pathlib import Path

from omni_stage_c.target_io import extract_target_io


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Qwen3-Omni Thinker embedding and LM head")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--hidden-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-token-id", type=int, default=152063)
    args = parser.parse_args()
    print(extract_target_io(args.model_path, args.hidden_cache_dir, args.output_dir,
                            args.mask_token_id))


if __name__ == "__main__":
    main()
