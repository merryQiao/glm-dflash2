from __future__ import annotations

import unittest

from glm_dflash2.dflash2_model import DFlashAttention, build_dflash2_config
from glm_dflash2.glm_draft_config import GLM52_DRAFT_SPEC


class GLMDraftConfigTest(unittest.TestCase):
    def test_published_glm52_spec_is_fixed(self):
        spec = GLM52_DRAFT_SPEC
        self.assertEqual(spec.target_layer_ids, (1, 20, 38, 56, 75))
        self.assertEqual(spec.target_num_hidden_layers, 78)
        self.assertEqual(spec.hidden_size, 6144)
        self.assertEqual(spec.intermediate_size, 12288)
        self.assertEqual(spec.num_hidden_layers, 5)
        self.assertEqual(spec.num_attention_heads, 64)
        self.assertEqual(spec.num_key_value_heads, 64)
        self.assertEqual(spec.head_dim, 64)
        self.assertEqual(spec.q_projection_size, 4096)
        self.assertEqual(spec.kv_projection_size, 4096)
        self.assertIsNone(spec.sliding_window)
        self.assertEqual(spec.rope_theta, 8_000_000.0)
        self.assertEqual(spec.rms_norm_eps, 1e-5)

    def test_attention_projection_width_is_heads_times_head_dim(self):
        config = build_dflash2_config(
            vocab_size=17,
            hidden_size=12,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            target_layer_ids=[1],
            num_target_layers=3,
            block_size=4,
            mask_token_id=16,
            conv_group_size=4,
            selector_rank=4,
            selector_top_k=4,
            sliding_window=None,
        )
        attention = DFlashAttention(config, layer_idx=0)
        self.assertEqual(tuple(attention.q_proj.weight.shape), (8, 12))
        self.assertEqual(tuple(attention.k_proj.weight.shape), (8, 12))
        self.assertEqual(tuple(attention.v_proj.weight.shape), (8, 12))
        self.assertEqual(tuple(attention.o_proj.weight.shape), (12, 8))


if __name__ == "__main__":
    unittest.main()
