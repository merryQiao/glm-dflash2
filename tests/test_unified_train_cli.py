from __future__ import annotations

import unittest

import torch

from glm_dflash2.dflash2_model import Qwen3DFlash2DraftModel, build_dflash2_config
from glm_dflash2.draft_backbone import DFlashDraftModel
from glm_dflash2.dspark_model import DSparkDraftModel
from tools.train_drafter_offline import (
    _sample_or_dummy_anchors,
    build_method_model,
    build_parser,
    resolve_method_recipe,
    validate_aligned_cache_manifest,
)


class UnifiedTrainCliTest(unittest.TestCase):
    def _config(self):
        return build_dflash2_config(
            vocab_size=11, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, target_layer_ids=[0, 1], num_target_layers=2,
            block_size=4, mask_token_id=10, conv_group_size=4,
            selector_rank=4, selector_top_k=4, sliding_window=None,
        )

    def test_parser_selects_method_but_not_target_or_mutable_architecture(self):
        parser = build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertNotIn("target_model", destinations)
        args = parser.parse_args(
            [
                "--method", "dspark", "--cache-dir", "/cache",
                "--target-io-dir", "/io", "--output-dir", "/out",
                "--mask-token-id", "9",
            ]
        )
        self.assertEqual(args.method, "dspark")
        args = resolve_method_recipe(args)
        self.assertEqual(args.block_size, 8)
        self.assertEqual(args.num_anchors, 64)
        self.assertEqual(args.gamma, 4.0)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.lr, 3e-4)
        self.assertEqual(args.selector_rank, 256)
        self.assertEqual(args.selector_top_k, 16)
        self.assertEqual(args.markov_rank, 256)

    def test_method_recipe_allows_requested_block_matrix_only(self):
        parser = build_parser()
        base = [
            "--cache-dir", "/cache", "--target-io-dir", "/io",
            "--output-dir", "/out", "--mask-token-id", "9",
        ]
        for method in ("dflash", "dflash2"):
            for block_size in (8, 16):
                args = parser.parse_args(
                    ["--method", method, "--block-size", str(block_size), *base]
                )
                self.assertEqual(resolve_method_recipe(args).block_size, block_size)
        args = parser.parse_args(
            ["--method", "dspark", "--block-size", "16", *base]
        )
        with self.assertRaisesRegex(ValueError, "DSpark.*block-size 8"):
            resolve_method_recipe(args)

    def test_method_dispatch_changes_only_method_specific_modules(self):
        expected = {
            "dflash": DFlashDraftModel,
            "dflash2": Qwen3DFlash2DraftModel,
            "dspark": DSparkDraftModel,
        }
        for method, model_type in expected.items():
            with self.subTest(method=method):
                model = build_method_model(method, self._config(), markov_rank=4)
                self.assertIsInstance(model, model_type)
                self.assertEqual(tuple(model.target_layer_ids), (0, 1))
                self.assertIsNone(model.config.sliding_window)
                self.assertEqual(model.config.drafter_method, method)
                self.assertEqual(model.config.position_contract, "absolute_anchor_plus_local")
                self.assertEqual(model.config.physical_block_size, 4)
                self.assertEqual(model.config.num_speculative_tokens, 3)

    def test_all_aligned_methods_reject_legacy_cache(self):
        for method in ("dflash", "dflash2", "dspark"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "schema v2"):
                validate_aligned_cache_manifest(
                    {"spec": {"schema_version": 1}, "aligned_methods_allowed": False},
                    method=method,
                )

    def test_anchor_sampling_consumes_collator_sample_ids(self):
        class Draft:
            block_size = 4

        class Trainer:
            draft_model = Draft()
            global_seed = 42
            num_anchors = 2

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.bool),
            "loss_mask": torch.ones(1, 4, dtype=torch.bool),
            "sample_ids": ["stable-id"],
        }
        anchors, keep, local_has = _sample_or_dummy_anchors(batch, Trainer(), epoch=0)
        self.assertTrue(local_has)
        self.assertEqual(tuple(anchors.shape), tuple(keep.shape))


if __name__ == "__main__":
    unittest.main()
