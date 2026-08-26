from __future__ import annotations

import unittest

import torch
from torch import nn

from glm_dflash2.dflash2_model import build_dflash2_config
from glm_dflash2.hidden_cache import DFlashHiddenCollator
from glm_dflash2.offline_trainer import (
    OfflineDFlash2Trainer,
    OfflineDFlashTrainer,
    OfflineDSparkTrainer,
)
from glm_dflash2.target_io import FrozenTargetIO
from tools.train_drafter_offline import build_method_model


class UnifiedPipelineTest(unittest.TestCase):
    def _config(self):
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

    def _target_io(self):
        embed = nn.Embedding(17, 8).to(torch.bfloat16)
        head = nn.Linear(8, 17, bias=False).to(torch.bfloat16)
        embed.weight.requires_grad_(False)
        head.weight.requires_grad_(False)
        return FrozenTargetIO(
            embed,
            head,
            {
                "schema": "glm-drafter-target-io-v2",
                "source_model_fingerprint": "tiny",
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

    def _manifest(self):
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
                "model_fingerprint": "tiny",
                "model_revision": "revision",
                "tokenizer_fingerprint": "tokenizer",
                "vocab_size": 17,
                "target_hidden_dtype": "bfloat16",
                "logical_layer_ids": [0, 1],
            },
        }

    def test_same_rows_and_anchors_drive_one_optimizer_step_for_all_methods(self):
        torch.manual_seed(9)
        rows = [
            {
                "sample_id": "stable-sample",
                "input_ids": torch.tensor([1, 2, 3, 4, 5, 6]),
                "loss_mask": torch.ones(6, dtype=torch.bool),
                "hidden_states": torch.randn(6, 16, dtype=torch.bfloat16),
                "target_final_hidden": torch.randn(6, 8, dtype=torch.bfloat16),
            }
        ]
        batch = DFlashHiddenCollator(pad_token_id=0)(rows)
        self.assertEqual(batch["sample_ids"], ["stable-sample"])
        anchors = torch.tensor([[0, 1]])
        keep = torch.tensor([[True, True]])

        trainer_types = {
            "dflash": OfflineDFlashTrainer,
            "dflash2": OfflineDFlash2Trainer,
            "dspark": OfflineDSparkTrainer,
        }
        for method, trainer_type in trainer_types.items():
            with self.subTest(method=method):
                model = build_method_model(method, self._config(), markov_rank=4).to(
                    torch.bfloat16
                )
                kwargs = dict(
                    cache_manifest=self._manifest(),
                    num_anchors=2,
                    token_chunk_size=4,
                    vocab_chunk_size=5,
                    global_seed=42,
                )
                if method == "dflash2":
                    kwargs["selector_loss_weight"] = 1.0
                if method == "dspark":
                    kwargs.update(
                        ce_weight=0.1, tv_weight=0.9, confidence_weight=1.0
                    )
                trainer = trainer_type(model, self._target_io(), **kwargs)
                before = {
                    name: value.detach().clone()
                    for name, value in trainer.draft_model.named_parameters()
                }
                optimizer = torch.optim.SGD(trainer.parameters(), lr=1e-3)
                output = trainer(
                    batch,
                    epoch=0,
                    anchor_positions=anchors,
                    block_keep_mask=keep,
                )
                self.assertTrue(torch.isfinite(output.loss))
                output.loss.backward()
                optimizer.step()
                changed = any(
                    not torch.equal(before[name], value.detach())
                    for name, value in trainer.draft_model.named_parameters()
                )
                self.assertTrue(changed)
                trainer.assert_frozen_io_unchanged()


if __name__ == "__main__":
    unittest.main()
