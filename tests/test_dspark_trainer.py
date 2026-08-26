from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch import nn

from glm_dflash2.dflash2_model import build_dflash2_config
from glm_dflash2.dspark_model import DSparkDraftModel
from glm_dflash2.offline_trainer import (
    OfflineDSparkTrainer,
    gather_dspark_teacher_hidden,
)
from glm_dflash2.target_io import FrozenTargetIO


class DSparkTrainerTest(unittest.TestCase):
    def _fixture(self):
        config = build_dflash2_config(
            vocab_size=13, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, target_layer_ids=[0, 1], num_target_layers=2,
            block_size=4, mask_token_id=12, conv_group_size=4,
            selector_rank=4, selector_top_k=4, sliding_window=None,
        )
        model = DSparkDraftModel(config, markov_rank=4).to(torch.bfloat16)
        embed = nn.Embedding(13, 8).to(torch.bfloat16)
        head = nn.Linear(8, 13, bias=False).to(torch.bfloat16)
        embed.weight.requires_grad_(False)
        head.weight.requires_grad_(False)
        io_manifest = {
            "schema": "glm-drafter-target-io-v2",
            "source_model_fingerprint": "tiny", "model_revision": "revision",
            "tokenizer_fingerprint": "tokenizer", "hidden_size": 8,
            "vocab_size": 13,
            "source_dtypes": {"embed_tokens": "torch.bfloat16", "lm_head": "torch.bfloat16"},
            "logit_transform": "identity", "lm_head_bias": False,
        }
        cache_manifest = {
            "spec": {
                "layer_ids": [0, 1], "hidden_size": 8, "dtype": "bfloat16",
                "mask_semantics": "dflash_target_token", "schema_version": 2,
                "final_hidden_semantics": "post_final_norm_lm_head_input",
            },
            "provenance": {
                "model_fingerprint": "tiny", "model_revision": "revision",
                "tokenizer_fingerprint": "tokenizer", "vocab_size": 13,
                "target_hidden_dtype": "bfloat16", "logical_layer_ids": [0, 1],
            },
        }
        trainer = OfflineDSparkTrainer(
            model,
            FrozenTargetIO(embed, head, io_manifest),
            cache_manifest=cache_manifest,
            num_anchors=2,
            gamma=7.0,
            vocab_chunk_size=5,
            global_seed=9,
        )
        positions = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8, 1)
        batch = {
            "sample_id": ["sample"],
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "loss_mask": torch.ones(1, 8, dtype=torch.bool),
            "hidden_states": torch.randn(1, 8, 16, dtype=torch.bfloat16),
            "target_final_hidden": positions.expand(1, 8, 8).contiguous(),
        }
        return trainer, batch, torch.tensor([[0, 3]]), torch.tensor([[True, True]])

    def test_teacher_hidden_uses_one_position_shift_and_predecessors_are_teacher_forced(self):
        _, batch, anchors, keep = self._fixture()
        teacher, predecessors = gather_dspark_teacher_hidden(
            batch["target_final_hidden"],
            batch["input_ids"],
            anchors,
            keep,
            prediction_depth=3,
        )
        self.assertEqual(teacher[..., 0].tolist(), [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]])
        self.assertEqual(predecessors.tolist(), [[[1, 2, 3], [4, 5, 6]]])

    def test_end_to_end_dspark_consumes_final_hidden_and_updates_all_method_heads(self):
        trainer, batch, anchors, keep = self._fixture()
        output = trainer(batch, anchor_positions=anchors, block_keep_mask=keep)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertGreater(output.loss_weight.item(), 0)
        self.assertGreater(output.valid_tokens.item(), 0)
        output.loss.backward()
        self.assertTrue(any(p.grad is not None for p in trainer.draft_model.layers.parameters()))
        self.assertTrue(any(p.grad is not None for p in trainer.draft_model.markov_head.parameters()))
        self.assertTrue(any(p.grad is not None for p in trainer.draft_model.confidence_head.parameters()))
        self.assertIsNone(trainer.target_lm_head_weight.grad)

    def test_missing_final_hidden_fails_before_training(self):
        trainer, batch, anchors, keep = self._fixture()
        del batch["target_final_hidden"]
        with self.assertRaisesRegex(ValueError, "target_final_hidden"):
            trainer(batch, anchor_positions=anchors, block_keep_mask=keep)

    def test_confidence_head_receives_teacher_forced_predecessors(self):
        trainer, batch, anchors, keep = self._fixture()
        seen = []
        original = trainer.draft_model.confidence_logits

        def remember(hidden, predecessors):
            seen.append(predecessors.detach().clone())
            return original(hidden, predecessors)

        with mock.patch.object(
            trainer.draft_model, "confidence_logits", side_effect=remember
        ):
            trainer(batch, anchor_positions=anchors, block_keep_mask=keep)
        self.assertEqual(len(seen), 1)
        self.assertEqual(
            seen[0].tolist(),
            [[[1, 2, 3], [4, 5, 6]]],
        )


if __name__ == "__main__":
    unittest.main()
