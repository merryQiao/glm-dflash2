from __future__ import annotations

import unittest

import torch
from torch import nn

from glm_dflash2.dflash2_model import Qwen3DFlash2DraftModel, build_dflash2_config
from glm_dflash2.offline_trainer import OfflineDFlash2Trainer
from glm_dflash2.target_io import FrozenTargetIO


class OfflineTrainerTest(unittest.TestCase):
    def _fixture(self):
        torch.manual_seed(8)
        config = build_dflash2_config(
            vocab_size=13,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            target_layer_ids=[0, 1],
            num_target_layers=2,
            block_size=4,
            mask_token_id=12,
            conv_group_size=4,
            selector_rank=4,
            selector_top_k=4,
            sliding_window=16,
        )
        embed = nn.Embedding(13, 8)
        head = nn.Linear(8, 13, bias=False)
        embed.weight.requires_grad_(False)
        head.weight.requires_grad_(False)
        io_manifest = {
            "source_model_fingerprint": "tiny-model",
            "hidden_size": 8,
            "vocab_size": 13,
        }
        target_io = FrozenTargetIO(embed, head, io_manifest)
        cache_manifest = {
            "spec": {
                "layer_ids": [0, 1],
                "hidden_size": 8,
                "dtype": "bfloat16",
                "mask_semantics": "dflash_target_token",
            },
            "provenance": {
                "model_fingerprint": "tiny-model",
                "logical_layer_ids": [0, 1],
            },
        }
        trainer = OfflineDFlash2Trainer(
            Qwen3DFlash2DraftModel(config),
            target_io,
            cache_manifest=cache_manifest,
            num_anchors=2,
            gamma=7.0,
            selector_loss_weight=1.0,
            token_chunk_size=2,
            vocab_chunk_size=5,
            anchor_seed=123,
        )
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "loss_mask": torch.ones(1, 8, dtype=torch.bool),
            "hidden_states": torch.randn(1, 8, 16),
        }
        anchors = torch.tensor([[0, 3]])
        keep = torch.tensor([[True, True]])
        return trainer, batch, anchors, keep

    def test_end_to_end_forward_shapes_metrics_and_frozen_io(self):
        trainer, batch, anchors, keep = self._fixture()
        before_embed = trainer.target_embed_weight.detach().clone()
        before_head = trainer.target_lm_head_weight.detach().clone()
        output = trainer(batch, anchor_positions=anchors, block_keep_mask=keep)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertGreater(output.valid_tokens.item(), 0)
        self.assertEqual(output.valid_blocks.item(), 2)
        self.assertGreaterEqual(output.candidate_recall.item(), 0)
        self.assertLessEqual(output.candidate_recall.item(), 1)
        output.loss.backward()
        self.assertTrue(any(p.grad is not None for p in trainer.draft_model.parameters()))
        self.assertIsNone(trainer.target_embed_weight.grad)
        self.assertIsNone(trainer.target_lm_head_weight.grad)
        torch.testing.assert_close(trainer.target_embed_weight, before_embed)
        torch.testing.assert_close(trainer.target_lm_head_weight, before_head)
        parameter_names = [name for name, _ in trainer.named_parameters()]
        self.assertTrue(parameter_names)
        self.assertTrue(all(name.startswith("draft_model.") for name in parameter_names))

    def test_one_small_optimizer_step_reduces_same_batch_loss(self):
        trainer, batch, anchors, keep = self._fixture()
        optimizer = torch.optim.SGD(trainer.parameters(), lr=1e-3)
        before = trainer(batch, anchor_positions=anchors, block_keep_mask=keep).loss
        before.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        after = trainer(batch, anchor_positions=anchors, block_keep_mask=keep).loss
        self.assertLess(after.item(), before.item())

    def test_anchor_generator_state_round_trip(self):
        trainer, batch, _, _ = self._fixture()
        state = trainer.anchor_generator.get_state()
        first = trainer(batch).anchor_positions
        trainer.anchor_generator.set_state(state)
        second = trainer(batch).anchor_positions
        self.assertTrue(torch.equal(first, second))

    def test_padded_block_and_masked_positions_keep_loss_finite(self):
        trainer, batch, anchors, keep = self._fixture()
        batch["attention_mask"] = torch.ones_like(batch["loss_mask"])
        batch["loss_mask"][0, 2] = False
        keep = torch.tensor([[True, False]])
        output = trainer(batch, anchor_positions=anchors, block_keep_mask=keep)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        gradients = [p.grad for p in trainer.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_rejects_cache_provenance_mismatch(self):
        trainer, _, _, _ = self._fixture()
        target_io = FrozenTargetIO(
            trainer._target_io.embed_tokens,
            trainer._target_io.lm_head,
            {**trainer._target_io.manifest, "source_model_fingerprint": "other"},
        )
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            OfflineDFlash2Trainer(
                trainer.draft_model,
                target_io,
                cache_manifest=trainer.cache_manifest,
                num_anchors=1,
            )


if __name__ == "__main__":
    unittest.main()
