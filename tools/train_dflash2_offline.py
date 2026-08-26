#!/usr/bin/env python3
"""Deprecated DFlash2-only entrypoint kept for command compatibility."""

from __future__ import annotations

import sys

if __package__:
    from .train_drafter_offline import build_parser as _build_parser
    from .train_drafter_offline import main as _main
    from .train_drafter_offline import _sample_or_dummy_anchors as _sample_anchors
else:
    from train_drafter_offline import build_parser as _build_parser
    from train_drafter_offline import main as _main
    from train_drafter_offline import _sample_or_dummy_anchors as _sample_anchors


def build_parser():
    return _build_parser(default_method="dflash2")


def main(argv: list[str] | None = None) -> int:
    return _main(argv, default_method="dflash2")


def _sample_or_dummy_anchors(batch, trainer):
    """Compatibility helper using the first epoch's deterministic anchors."""

    if "sample_id" not in batch:
        batch = dict(batch)
        batch["sample_id"] = [
            f"legacy-row-{row}" for row in range(int(batch["input_ids"].shape[0]))
        ]
    return _sample_anchors(batch, trainer, epoch=0)


if __name__ == "__main__":
    sys.exit(main())
