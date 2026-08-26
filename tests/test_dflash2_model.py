from __future__ import annotations

import unittest

import torch

from glm_dflash2.dflash2_model import (
    CandidateSelector,
    GroupedDynamicCausalConv,
    Qwen3DFlash2DraftModel,
    build_dflash2_config,
    build_glm52_dflash2_config,
)


class DFlash2ModelTest(unittest.TestCase):
    def test_glm52_export_config_is_complete_and_fixed(self):
        config = build_glm52_dflash2_config(vocab_size=154880, mask_token_id=154879)
        self.assertEqual(config.model_type, "qwen3")
        self.assertEqual(config.architectures, ["DFlash2DraftModel"])
        self.assertEqual(config.hidden_size, 6144)
        self.assertEqual(config.intermediate_size, 12288)
        self.assertEqual(config.num_hidden_layers, 5)
        self.assertEqual(config.num_attention_heads, 64)
        self.assertEqual(config.num_key_value_heads, 64)
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.rms_norm_eps, 1e-5)
        self.assertEqual(config.rope_theta, 8_000_000)
        self.assertEqual(config.layer_types, ["full_attention"] * 5)
        self.assertFalse(config.is_causal)
        self.assertFalse(config.use_sliding_window)
        self.assertIsNone(config.sliding_window)
        expected = {
            "block_size": 16,
            "mask_token_id": 154879,
            "target_layer_ids": [1, 20, 38, 56, 75],
            "num_target_layers": 78,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": 256,
            "selector_top_k": 16,
        }
        for key, value in expected.items():
            self.assertEqual(config.dflash_config[key], value)
        self.assertEqual(config.max_position_embeddings, 1_048_576)

    def test_grouped_convolution_is_identity_and_never_crosses_blocks(self):
        conv = GroupedDynamicCausalConv(4, kernel_size=2, group_size=2)
        x = torch.arange(1, 33, dtype=torch.float32).reshape(1, 8, 4)
        identity, dynamic = conv.prepare(x, block_size=4)
        torch.testing.assert_close(identity, x)
        torch.testing.assert_close(conv.finish(x, dynamic, block_size=4), x)
        with torch.no_grad():
            conv.base_kernel.zero_()
            conv.base_kernel[:, 1].fill_(1.0)
        shifted, shifted_dynamic = conv.prepare(x, block_size=4)
        expected = torch.zeros_like(x).reshape(1, 2, 4, 4)
        expected[:, :, 1:] = x.reshape(1, 2, 4, 4)[:, :, :-1]
        torch.testing.assert_close(shifted, expected.reshape_as(x))
        torch.testing.assert_close(
            conv.finish(x, shifted_dynamic, block_size=4), expected.reshape_as(x)
        )

    def test_selector_initially_preserves_unary_logits_and_key_contract(self):
        selector = CandidateSelector(vocab_size=13, hidden_size=8, rank=4, top_k=3)
        unary = torch.randn(2, 5, 3)
        ids = torch.randint(0, 13, (2, 5, 3))
        pred = torch.randint(0, 13, (2, 5))
        scores = selector.pair_scores(torch.randn(2, 5, 8), unary, ids, pred)
        torch.testing.assert_close(scores, unary)
        keys = set(selector.state_dict())
        self.assertIn("predecessor_codebook", keys)
        self.assertIn("successor_codebook", keys)
        self.assertNotIn("predecessor_codebook.weight", keys)

    def test_tiny_forward_shapes_and_backward(self):
        config = build_dflash2_config(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            target_layer_ids=[1, 3],
            num_target_layers=4,
            block_size=4,
            mask_token_id=30,
            conv_group_size=4,
            selector_rank=8,
            selector_top_k=4,
            sliding_window=32,
        )
        model = Qwen3DFlash2DraftModel(config)
        target_hidden = torch.randn(2, 6, 32)
        noise = torch.randn(2, 8, 16, requires_grad=True)
        positions = torch.cat(
            (torch.arange(6).expand(2, -1), torch.tensor([[2, 3, 4, 5, 4, 5, 6, 7]]).expand(2, -1)),
            dim=1,
        )
        mask = torch.zeros(2, 1, 8, 14)
        output = model(
            position_ids=positions,
            attention_mask=mask,
            noise_embedding=noise,
            target_hidden=target_hidden,
            conv_block_size=4,
        )
        self.assertEqual(tuple(output.shape), (2, 8, 16))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(noise.grad).all())
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        self.assertEqual(tuple(model.fc.weight.shape), (16, 32))
        self.assertEqual(tuple(model.layers[0].attention_conv.base_kernel.shape), (2, 2, 16))
        self.assertEqual(tuple(model.candidate_selector.predecessor_codebook.shape), (31, 8))

    def test_rejects_wrong_target_feature_width_and_conv_partition(self):
        config = build_dflash2_config(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            target_layer_ids=[0, 1],
            num_target_layers=2,
            block_size=2,
            mask_token_id=7,
            conv_group_size=4,
            selector_rank=2,
            selector_top_k=2,
            sliding_window=8,
        )
        model = Qwen3DFlash2DraftModel(config)
        with self.assertRaisesRegex(ValueError, "target_hidden"):
            model(
                position_ids=torch.arange(4).unsqueeze(0),
                attention_mask=torch.zeros(1, 1, 2, 4),
                noise_embedding=torch.zeros(1, 2, 8),
                target_hidden=torch.zeros(1, 2, 8),
            )
        with self.assertRaisesRegex(ValueError, "divide"):
            GroupedDynamicCausalConv(7, kernel_size=2, group_size=4)


if __name__ == "__main__":
    unittest.main()
