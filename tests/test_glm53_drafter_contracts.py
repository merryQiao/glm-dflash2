from __future__ import annotations

import unittest

from glm53_drafters.contracts import (
    CACHE_CONTRACT,
    DRAFT_CONTRACT,
    METHOD_BLOCK_SIZES,
    TARGET_CONTRACT,
    estimate_cache_bytes,
    validate_method_block,
)


class GLM53DrafterContractsTest(unittest.TestCase):
    def test_target_contract_uses_official_taps_and_hidden_indices(self):
        self.assertEqual(TARGET_CONTRACT.num_layers, 45)
        self.assertEqual(TARGET_CONTRACT.hidden_size, 4096)
        self.assertEqual(TARGET_CONTRACT.vocab_size, 154880)
        self.assertEqual(TARGET_CONTRACT.logical_layer_ids, (1, 11, 22, 32, 42))
        self.assertEqual(TARGET_CONTRACT.hidden_state_indices, (2, 12, 23, 33, 43))
        self.assertEqual(TARGET_CONTRACT.final_hidden_size, 4096)

    def test_draft_contract_is_fixed_five_layer_full_attention_geometry(self):
        self.assertEqual(DRAFT_CONTRACT.num_layers, 5)
        self.assertEqual(DRAFT_CONTRACT.hidden_size, 4096)
        self.assertEqual(DRAFT_CONTRACT.intermediate_size, 12288)
        self.assertEqual(DRAFT_CONTRACT.num_attention_heads, 64)
        self.assertEqual(DRAFT_CONTRACT.num_key_value_heads, 64)
        self.assertEqual(DRAFT_CONTRACT.head_dim, 64)
        self.assertTrue(DRAFT_CONTRACT.full_attention)
        self.assertIsNone(DRAFT_CONTRACT.sliding_window)

    def test_method_block_matrix_contains_exactly_five_variants(self):
        self.assertEqual(
            dict(METHOD_BLOCK_SIZES),
            {"dflash": (8, 16), "dflash2": (8, 16), "dspark": (8,)},
        )
        for method, block_size in (
            ("dflash", 8),
            ("dflash", 16),
            ("dflash2", 8),
            ("dflash2", 16),
            ("dspark", 8),
        ):
            validate_method_block(method, block_size)
        with self.assertRaisesRegex(ValueError, "DSpark.*8"):
            validate_method_block("dspark", 16)
        with self.assertRaisesRegex(ValueError, "unknown method"):
            validate_method_block("other", 8)

    def test_cache_contract_shapes_and_storage_include_both_hidden_streams(self):
        self.assertEqual(CACHE_CONTRACT.schema_version, 2)
        self.assertEqual(CACHE_CONTRACT.row_shapes(3), {
            "input_ids": (3,),
            "loss_mask": (3,),
            "aux_hidden_states": (3, 5, 4096),
            "target_final_hidden": (3, 4096),
        })
        self.assertEqual(CACHE_CONTRACT.bytes_per_token, 6 * 4096 * 2 + 9)
        self.assertEqual(estimate_cache_bytes(10), 10 * (6 * 4096 * 2 + 9))
        with self.assertRaisesRegex(ValueError, "token_count"):
            estimate_cache_bytes(-1)


if __name__ == "__main__":
    unittest.main()
