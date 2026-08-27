#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from glm_dflash2.vllm_ascend.capability import (
    collect_runtime_identity,
    load_runtime_identity,
)
from glm_dflash2.vllm_ascend.parity import attest_candidate, validate_deploy_attestation


def _write(path: str | Path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect-runtime")
    collect.add_argument("--output", required=True)
    collect.add_argument("--tp-size", type=int, required=True)
    collect.add_argument("--ep-size", type=int, required=True)
    collect.add_argument("--pp-size", type=int, default=1)
    collect.add_argument("--dp-size", type=int, default=1)
    collect.add_argument("--nnodes", type=int, default=1)
    collect.add_argument("--attention-backend", default="ascend")
    collect.add_argument("--model-runner", default="v1")
    collect.add_argument("--graph-mode", default="disabled")
    collect.add_argument("--chunked-prefill", action="store_true")
    collect.add_argument("--prefix-cache", action="store_true")
    attest = sub.add_parser("attest")
    attest.add_argument("--export", required=True)
    attest.add_argument("--runtime-identity", required=True)
    attest.add_argument("--parity-results", required=True)
    validate = sub.add_parser("validate-attestation")
    validate.add_argument("--export", required=True)
    validate.add_argument("--runtime-identity", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "collect-runtime":
        result = collect_runtime_identity(
            tp_size=args.tp_size,
            ep_size=args.ep_size,
            pp_size=args.pp_size,
            dp_size=args.dp_size,
            nnodes=args.nnodes,
            attention_backend=args.attention_backend,
            model_runner=args.model_runner,
            graph_mode=args.graph_mode,
            chunked_prefill=args.chunked_prefill,
            prefix_cache=args.prefix_cache,
        )
        _write(args.output, result)
    elif args.command == "attest":
        runtime = load_runtime_identity(args.runtime_identity, production=True)
        parity = json.loads(Path(args.parity_results).read_text(encoding="utf-8"))
        result = attest_candidate(args.export, runtime_identity=runtime, parity_results=parity)
    else:
        runtime = load_runtime_identity(args.runtime_identity, production=True)
        result = validate_deploy_attestation(args.export, active_runtime=runtime)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
