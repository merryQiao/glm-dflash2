#!/usr/bin/env python3
from __future__ import annotations

import json

import torch
from torch import nn

from glm_dflash2.dflash2_model import build_dflash2_config
from glm_dflash2.offline_trainer import (
    OfflineDFlash2Trainer,
    OfflineDFlashTrainer,
    OfflineDSparkTrainer,
)
from glm_dflash2.target_io import FrozenTargetIO
from train_drafter_offline import build_method_model


METHODS = ("dflash", "dflash2", "dspark")


def _config():
    return build_dflash2_config(
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
        sliding_window=None,
    )


def _target_io() -> FrozenTargetIO:
    embed = nn.Embedding(17, 8).to(torch.bfloat16)
    head = nn.Linear(8, 17, bias=False).to(torch.bfloat16)
    embed.weight.requires_grad_(False)
    head.weight.requires_grad_(False)
    return FrozenTargetIO(
        embed,
        head,
        {
            "schema": "glm-drafter-target-io-v2",
            "source_model_fingerprint": "smoke",
            "model_revision": "revision",
            "tokenizer_fingerprint": "tokenizer",
            "hidden_size": 8,
            "vocab_size": 17,
            "source_dtypes": {
                "embed_tokens": "torch.bfloat16",
                "lm_head": "torch.bfloat16",
            },
            "logit_transform": "identity",
            "lm_head_bias": False,
        },
    )


def _cache_manifest():
    return {
        "spec": {
            "schema_version": 2,
            "layer_ids": [0, 1],
            "hidden_size": 8,
            "dtype": "bfloat16",
            "mask_semantics": "dflash_target_token",
            "final_hidden_semantics": "post_final_norm_lm_head_input",
        },
        "provenance": {
            "model_fingerprint": "smoke",
            "model_revision": "revision",
            "tokenizer_fingerprint": "tokenizer",
            "vocab_size": 17,
            "target_hidden_dtype": "bfloat16",
            "logical_layer_ids": [0, 1],
        },
    }


def main() -> None:
    torch.manual_seed(17)
    batch = {
        "sample_ids": ["smoke-sample"],
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.bool),
        "loss_mask": torch.ones(1, 6, dtype=torch.bool),
        "hidden_states": torch.randn(1, 6, 16, dtype=torch.bfloat16),
        "target_final_hidden": torch.randn(1, 6, 8, dtype=torch.bfloat16),
    }
    anchors = torch.tensor([[0, 1]])
    keep = torch.tensor([[True, True]])
    results = {}
    trainer_types = {
        "dflash": OfflineDFlashTrainer,
        "dflash2": OfflineDFlash2Trainer,
        "dspark": OfflineDSparkTrainer,
    }
    for method in METHODS:
        model = build_method_model(method, _config(), markov_rank=4).to(torch.bfloat16)
        kwargs = {
            "cache_manifest": _cache_manifest(),
            "num_anchors": 2,
            "token_chunk_size": 4,
            "vocab_chunk_size": 5,
            "global_seed": 42,
        }
        if method == "dflash2":
            kwargs["selector_loss_weight"] = 1.0
        elif method == "dspark":
            kwargs.update(ce_weight=0.1, tv_weight=0.9, confidence_weight=1.0)
        trainer = trainer_types[method](model, _target_io(), **kwargs)
        before = [parameter.detach().clone() for parameter in trainer.parameters()]
        output = trainer(
            batch,
            epoch=0,
            anchor_positions=anchors,
            block_keep_mask=keep,
        )
        if not bool(torch.isfinite(output.loss)):
            raise RuntimeError(f"{method} smoke loss is non-finite")
        output.loss.backward()
        optimizer = torch.optim.SGD(trainer.parameters(), lr=1e-3)
        optimizer.step()
        if not any(
            not torch.equal(old, parameter.detach())
            for old, parameter in zip(before, trainer.parameters())
        ):
            raise RuntimeError(f"{method} smoke optimizer step changed no parameter")
        trainer.assert_frozen_io_unchanged()
        results[method] = {"loss": float(output.loss.detach())}
    print(json.dumps({"status": "ok", "methods": results}, sort_keys=True))


if __name__ == "__main__":
    main()
