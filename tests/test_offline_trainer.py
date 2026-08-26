from __future__ import annotations

import unittest

import torch
from torch import nn

from glm_dflash2.dflash2_model import Qwen3DFlash2DraftModel, build_dflash2_config
from glm_dflash2.draft_backbone import DFlashDraftModel
from glm_dflash2.offline_trainer import OfflineDFlash2Trainer, OfflineDFlashTrainer
from glm_dflash2.target_io import FrozenTargetIO


class OfflineTrainerTest(unittest.TestCase):
    def _fixture(self, method: str = "dflash2"):
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
            sliding_window=None,
        )
        embed = nn.Embedding(13, 8).to(torch.bfloat16)
        head = nn.Linear(8, 13, bias=False).to(torch.bfloat16)
        embed.weight.requires_grad_(False)
        head.weight.requires_grad_(False)
        io_manifest = {
            "schema": "glm-drafter-target-io-v2",
            "source_model_fingerprint": "tiny-model",
            "model_revision": "tiny-revision",
            "tokenizer_fingerprint": "tiny-tokenizer",
            "hidden_size": 8,
            "vocab_size": 13,
            "source_dtypes": {
                "embed_tokens": "torch.bfloat16",
                "lm_head": "torch.bfloat16",
            },
            "logit_transform": "identity",
            "lm_head_bias": False,
        }
        target_io = FrozenTargetIO(embed, head, io_manifest)
        cache_manifest = {
            "spec": {
                "layer_ids": [0, 1],
                "hidden_size": 8,
                "dtype": "bfloat16",
                "mask_semantics": "dflash_target_token",
                "schema_version": 2,
                "final_hidden_semantics": "post_final_norm_lm_head_input",
            },
            "provenance": {
                "model_fingerprint": "tiny-model",
                "model_revision": "tiny-revision",
                "tokenizer_fingerprint": "tiny-tokenizer",
                "vocab_size": 13,
                "target_hidden_dtype": "bfloat16",
                "logical_layer_ids": [0, 1],
            },
        }
        model = (
            Qwen3DFlash2DraftModel(config)
            if method == "dflash2"
            else DFlashDraftModel(config)
        ).to(torch.bfloat16)
        trainer_cls = OfflineDFlash2Trainer if method == "dflash2" else OfflineDFlashTrainer
        trainer = trainer_cls(
            model,
            target_io,
            cache_manifest=cache_manifest,
            num_anchors=2,
            gamma=7.0,
            token_chunk_size=2,
            vocab_chunk_size=5,
            global_seed=123,
        )
        batch = {
            "sample_id": ["sample-a"],
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "loss_mask": torch.ones(1, 8, dtype=torch.bool),
            "hidden_states": torch.randn(1, 8, 16, dtype=torch.bfloat16),
            "target_final_hidden": torch.full(
                (1, 8, 8), float("nan"), dtype=torch.bfloat16
            ),
        }
        anchors = torch.tensor([[0, 3]])
        keep = torch.tensor([[True, True]])
        return trainer, batch, anchors, keep

    def test_both_methods_forward_backward_with_frozen_io_and_ignore_final_hidden(self):
        for method in ("dflash", "dflash2"):
            with self.subTest(method=method):
                trainer, batch, anchors, keep = self._fixture(method)
                before_embed = trainer.target_embed_weight.detach().clone()
                before_head = trainer.target_lm_head_weight.detach().clone()
                output = trainer(batch, anchor_positions=anchors, block_keep_mask=keep)
                self.assertTrue(torch.isfinite(output.loss))
                self.assertGreater(output.valid_tokens.item(), 0)
                self.assertEqual(output.valid_blocks.item(), 2)
                output.loss.backward()
                self.assertTrue(
                    any(p.grad is not None for p in trainer.draft_model.parameters())
                )
                self.assertIsNone(trainer.target_embed_weight.grad)
                self.assertIsNone(trainer.target_lm_head_weight.grad)
                torch.testing.assert_close(trainer.target_embed_weight, before_embed)
                torch.testing.assert_close(trainer.target_lm_head_weight, before_head)
                self.assertTrue(
                    all(
                        name.startswith("draft_model.")
                        for name, _ in trainer.named_parameters()
                    )
                )

    def test_plain_dflash_has_no_dynamic_convolution_or_selector(self):
        trainer, _, _, _ = self._fixture("dflash")
        names = tuple(name for name, _ in trainer.draft_model.named_modules())
        self.assertFalse(any("conv" in name for name in names))
        self.assertFalse(hasattr(trainer.draft_model, "candidate_selector"))

    def test_selector_predecessors_are_anchor_then_teacher_previous_tokens(self):
        trainer, batch, anchors, keep = self._fixture("dflash2")
        captured = []

        def remember(_module, args):
            captured.append(args[3].detach().clone())

        handle = trainer.draft_model.candidate_selector.register_forward_pre_hook(remember)
        try:
            trainer(batch, anchor_positions=anchors, block_keep_mask=keep)
        finally:
            handle.remove()
        self.assertEqual(captured[0].tolist(), [[[1, 2, 3], [4, 5, 6]]])

    def test_automatic_anchor_sampling_is_epoch_and_sample_id_deterministic(self):
        trainer, batch, _, _ = self._fixture("dflash2")
        first = trainer(batch, epoch=4).anchor_positions
        second = trainer(batch, epoch=4).anchor_positions
        other_epoch = trainer(batch, epoch=5).anchor_positions
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other_epoch))

    def test_padded_block_and_masked_positions_keep_loss_finite(self):
        trainer, batch, anchors, _ = self._fixture("dflash2")
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
        trainer, _, _, _ = self._fixture("dflash2")
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
