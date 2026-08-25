#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from torch import nn

from glm_dflash2.checkpointing import (
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)
from glm_dflash2.dflash2_model import Qwen3DFlash2DraftModel, build_dflash2_config
from glm_dflash2.offline_trainer import OfflineDFlash2Trainer
from glm_dflash2.target_io import FrozenTargetIO


def main() -> None:
    torch.manual_seed(17)
    config = build_dflash2_config(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        target_layer_ids=[0, 1],
        num_target_layers=2,
        block_size=4,
        mask_token_id=16,
        conv_group_size=4,
        selector_rank=4,
        selector_top_k=4,
        sliding_window=16,
    )
    embed = nn.Embedding(17, 8)
    head = nn.Linear(8, 17, bias=False)
    embed.weight.requires_grad_(False)
    head.weight.requires_grad_(False)
    manifest = {"source_model_fingerprint": "smoke", "hidden_size": 8, "vocab_size": 17}
    cache = {
        "spec": {
            "layer_ids": [0, 1],
            "hidden_size": 8,
            "dtype": "bfloat16",
            "mask_semantics": "dflash_target_token",
        },
        "provenance": {"model_fingerprint": "smoke", "logical_layer_ids": [0, 1]},
    }
    trainer = OfflineDFlash2Trainer(
        Qwen3DFlash2DraftModel(config),
        FrozenTargetIO(embed, head, manifest),
        cache_manifest=cache,
        num_anchors=2,
        token_chunk_size=3,
        vocab_chunk_size=5,
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "loss_mask": torch.ones(1, 8, dtype=torch.bool),
        "hidden_states": torch.randn(1, 8, 16),
    }
    anchors = torch.tensor([[0, 3]])
    keep = torch.tensor([[True, True]])
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=2e-3)
    first = float(
        trainer(batch, anchor_positions=anchors, block_keep_mask=keep).loss.detach()
    )
    last = first
    for _ in range(12):
        output = trainer(batch, anchor_positions=anchors, block_keep_mask=keep)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last = float(output.loss.detach())
    if not last < first:
        raise RuntimeError(f"tiny overfit did not improve: first={first} last={last}")
    with tempfile.TemporaryDirectory() as tmp:
        progress = TrainingProgress(12, 12, 0, 1)
        save_training_checkpoint(
            tmp,
            model=trainer.draft_model,
            optimizer=optimizer,
            scheduler=None,
            progress=progress,
            anchor_generator=trainer.anchor_generator,
            at_optimizer_boundary=True,
        )
        restored = load_training_checkpoint(
            tmp,
            model=trainer.draft_model,
            optimizer=optimizer,
            scheduler=None,
            anchor_generator=trainer.anchor_generator,
        )
        if restored != progress:
            raise RuntimeError("checkpoint progress did not round-trip")
    print(json.dumps({"status": "ok", "first_loss": first, "last_loss": last}))


if __name__ == "__main__":
    main()
