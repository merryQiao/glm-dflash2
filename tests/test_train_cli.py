from __future__ import annotations

import unittest

from tools.train_drafter_offline import build_parser, resolve_method_recipe


class TrainCliTest(unittest.TestCase):
    def test_fixed_training_defaults_and_no_target_backbone_argument(self):
        parser = build_parser(default_method="dflash2")
        destinations = {action.dest for action in parser._actions}
        self.assertNotIn("target_model", destinations)
        args = resolve_method_recipe(parser.parse_args(
            [
                "--cache-dir", "/cache",
                "--target-io-dir", "/io",
                "--output-dir", "/out",
                "--mask-token-id", "9",
            ]
        ))
        self.assertEqual(args.block_size, 16)
        self.assertEqual(args.num_anchors, 64)
        self.assertEqual(args.gamma, 7.0)
        self.assertEqual(args.selector_rank, 256)
        self.assertEqual(args.selector_top_k, 16)
        self.assertEqual(args.hidden_size, 6144)
        self.assertEqual(args.intermediate_size, 12288)
        self.assertEqual(args.num_draft_layers, 5)
        self.assertEqual(args.lr, 6e-4)
        self.assertEqual((args.beta1, args.beta2), (0.9, 0.95))
        self.assertEqual(args.weight_decay, 0.0)
        self.assertEqual(args.device, "npu")
        self.assertTrue(args.fsdp2)

    def test_rejects_mutating_fixed_architecture(self):
        parser = build_parser(default_method="dflash2")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--cache-dir", "/cache",
                    "--target-io-dir", "/io",
                    "--output-dir", "/out",
                    "--mask-token-id", "9",
                    "--block-size", "32",
                ]
            )


if __name__ == "__main__":
    unittest.main()
