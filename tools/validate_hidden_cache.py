#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm_dflash2.hidden_cache import PackedHiddenDataset

if __package__:
    from .calibrate_hidden_capture_gate import validate_capture_with_gate
else:
    from calibrate_hidden_capture_gate import validate_capture_with_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--expected-layer-ids", default="1,20,38,56,75")
    parser.add_argument("--full-scan", action="store_true")
    parser.add_argument(
        "--allow-building-cache",
        action="store_true",
        help="Validate a deliberately partial hardware-gate cache.",
    )
    parser.add_argument(
        "--reference-pt",
        type=Path,
        help=(
            "Independent direct-forward capture with input_ids, "
            "aux_hidden_states [T,L,H], target_final_hidden [T,H], and layer_ids"
        ),
    )
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument(
        "--parity-gate",
        type=Path,
        help="Immutable bounds emitted by calibrate_hidden_capture_gate.py.",
    )
    parser.add_argument(
        "--runtime-identity-json",
        type=Path,
        help="Active target/runtime identity; required with --parity-gate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.parity_gate is None) != (args.runtime_identity_json is None):
        parser.error("--parity-gate and --runtime-identity-json must be used together")
    dataset = PackedHiddenDataset(
        args.cache_dir,
        require_frozen=not args.allow_building_cache,
        verify_checksums=True,
    )
    expected_layers = tuple(int(value) for value in args.expected_layer_ids.split(","))
    if dataset.spec.layer_ids != expected_layers:
        raise ValueError(f"layer IDs {dataset.spec.layer_ids} != {expected_layers}")
    if args.expected_samples is not None and len(dataset) != args.expected_samples:
        raise ValueError(f"sample count {len(dataset)} != {args.expected_samples}")
    indices = range(len(dataset)) if args.full_scan else range(min(1, len(dataset)))
    tokens = 0
    for index in indices:
        row = dataset[index]
        for key in ("layer_hidden_states", "target_final_hidden"):
            if not torch.isfinite(row[key].float()).all():
                raise ValueError(f"sample {row['sample_id']} contains non-finite {key}")
        if row["input_ids"].numel() != row["loss_mask"].numel():
            raise ValueError(f"sample {row['sample_id']} token/mask mismatch")
        tokens += int(row["input_ids"].numel())

    reference_status = "not_requested"
    parity_result = None
    if args.reference_pt is not None:
        if not len(dataset):
            raise ValueError("cannot compare a reference against an empty cache")
        reference = torch.load(args.reference_pt, map_location="cpu", weights_only=True)
        row = dataset[0]
        reference_ids = torch.as_tensor(reference["input_ids"], dtype=torch.int64).reshape(-1)
        reference_aux = reference.get("aux_hidden_states", reference.get("hidden_states"))
        if reference_aux is None:
            raise ValueError("reference lacks aux_hidden_states")
        reference_hidden = torch.as_tensor(reference_aux).to(torch.bfloat16)
        if not torch.equal(row["input_ids"], reference_ids):
            raise ValueError("reference input_ids differ from cache sample 0")
        if tuple(reference_hidden.shape) != tuple(row["layer_hidden_states"].shape):
            raise ValueError(
                f"reference hidden shape {tuple(reference_hidden.shape)} differs from "
                f"cache {tuple(row['layer_hidden_states'].shape)}"
            )
        if "layer_ids" in reference and tuple(reference["layer_ids"]) != dataset.spec.layer_ids:
            raise ValueError("reference layer IDs differ from cache")
        reference_final = reference.get("target_final_hidden")
        if reference_final is not None:
            reference_final = torch.as_tensor(reference_final).to(torch.bfloat16)
            if tuple(reference_final.shape) != tuple(row["target_final_hidden"].shape):
                raise ValueError("reference target_final_hidden shape differs from cache")

        if args.parity_gate is not None:
            if reference_final is None:
                raise ValueError("parity gate requires reference target_final_hidden")
            artifact = json.loads(args.parity_gate.read_text(encoding="utf-8"))
            identity = json.loads(args.runtime_identity_json.read_text(encoding="utf-8"))
            parity_result = validate_capture_with_gate(
                reference={
                    "aux_hidden_states": reference_hidden,
                    "target_final_hidden": reference_final,
                },
                candidate={
                    "aux_hidden_states": row["layer_hidden_states"],
                    "target_final_hidden": row["target_final_hidden"],
                },
                artifact=artifact,
                identity=identity,
            )
            if not parity_result["passed"]:
                raise ValueError(f"hidden capture parity gate failed: {parity_result['results']}")
            reference_status = "matched_parity_gate"
        else:
            for layer_index, layer_id in enumerate(dataset.spec.layer_ids):
                if not torch.allclose(
                    row["layer_hidden_states"][:, layer_index].float(),
                    reference_hidden[:, layer_index].float(),
                    atol=args.atol,
                    rtol=args.rtol,
                ):
                    delta = (
                        row["layer_hidden_states"][:, layer_index].float()
                        - reference_hidden[:, layer_index].float()
                    ).abs().max()
                    raise ValueError(
                        f"reference mismatch at logical layer {layer_id}: max_abs={delta.item()}"
                    )
            if reference_final is not None and not torch.allclose(
                row["target_final_hidden"].float(),
                reference_final.float(),
                atol=args.atol,
                rtol=args.rtol,
            ):
                delta = (
                    row["target_final_hidden"].float() - reference_final.float()
                ).abs().max()
                raise ValueError(f"reference final hidden mismatch: max_abs={delta.item()}")
            reference_status = "matched_all_hidden_streams"

    print(
        json.dumps(
            {
                "status": "ok",
                "samples": len(dataset),
                "checked_samples": len(list(indices)),
                "checked_tokens": tokens,
                "layers": list(dataset.spec.layer_ids),
                "hidden_size": dataset.spec.hidden_size,
                "reference": reference_status,
                "parity": parity_result,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
