#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from glm_dflash2.vllm_eval import (
    benchmark_openai_server,
    compare_benchmark_results,
    load_prompts,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Benchmark GLM speculative decoding on vLLM-Ascend")
    sub = root.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--prompts-jsonl", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--max-samples", type=int, default=0)
    run.add_argument("--max-tokens", type=int, default=2048)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--warmup-requests", type=int, default=2)
    run.add_argument("--timeout", type=float, default=1800.0)
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--speculative", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--require-exact-outputs", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "run":
        prompts = load_prompts(args.prompts_jsonl, max_samples=args.max_samples)
        result = benchmark_openai_server(
            base_url=args.base_url,
            model=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            warmup_requests=args.warmup_requests,
            timeout=args.timeout,
        )
    else:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        speculative = json.loads(Path(args.speculative).read_text(encoding="utf-8"))
        result = compare_benchmark_results(
            baseline, speculative, require_exact_outputs=args.require_exact_outputs
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result.get("summary", result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
