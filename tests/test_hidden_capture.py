from __future__ import annotations

import unittest

import torch

from glm_dflash2.hidden_capture import CaptureTap, TargetHiddenCapture


class HiddenCaptureTest(unittest.TestCase):
    def test_capture_validates_and_converts_both_streams_together(self):
        taps = (
            CaptureTap("hf", 1, "hidden_states[2]", "post_decoder_block"),
            CaptureTap("hf", 3, "hidden_states[4]", "post_decoder_block"),
        )
        capture = TargetHiddenCapture(
            aux_hidden_states=torch.randn(4, 2, 8, dtype=torch.float32),
            target_final_hidden=torch.randn(4, 8, dtype=torch.float32),
            capture_mapping=taps,
            final_hidden_semantics="post_final_norm_lm_head_input",
        ).cpu_bfloat16()
        self.assertEqual(capture.aux_hidden_states.dtype, torch.bfloat16)
        self.assertEqual(capture.target_final_hidden.dtype, torch.bfloat16)
        self.assertEqual(capture.logical_layer_ids, (1, 3))

    def test_capture_rejects_ambiguous_or_misaligned_final_hidden(self):
        taps = (CaptureTap("hf", 1, "hidden_states[2]", "post_decoder_block"),)
        with self.assertRaisesRegex(ValueError, "post-final-norm"):
            TargetHiddenCapture(
                aux_hidden_states=torch.zeros(2, 1, 4),
                target_final_hidden=torch.zeros(2, 4),
                capture_mapping=taps,
                final_hidden_semantics="decoder_layer_77_output",
            )
        with self.assertRaisesRegex(ValueError, "token dimension"):
            TargetHiddenCapture(
                aux_hidden_states=torch.zeros(2, 1, 4),
                target_final_hidden=torch.zeros(3, 4),
                capture_mapping=taps,
                final_hidden_semantics="post_final_norm_lm_head_input",
            )


if __name__ == "__main__":
    unittest.main()
