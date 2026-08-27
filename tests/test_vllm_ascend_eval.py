from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from glm_dflash2.vllm_eval import (
    compare_benchmark_results,
    parse_spec_decode_metrics,
    summarize_spec_decode,
)


class VllmAscendEvalTest(unittest.TestCase):
    def test_prometheus_counters_are_summed_across_labels(self):
        text = """
# HELP vllm:spec_decode_num_drafts_total Number of drafts
vllm:spec_decode_num_drafts_total{engine="0"} 10
vllm:spec_decode_num_drafts_total{engine="1"} 5
vllm:spec_decode_num_draft_tokens_total{engine="0"} 70
vllm:spec_decode_num_draft_tokens_total{engine="1"} 35
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 42
vllm:spec_decode_num_accepted_tokens_total{engine="1"} 18
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 9
"""
        metrics = parse_spec_decode_metrics(text)
        self.assertEqual(metrics["num_drafts"], 15)
        self.assertEqual(metrics["num_draft_tokens"], 105)
        self.assertEqual(metrics["num_accepted_tokens"], 60)

    def test_acceptance_uses_vllm_bonus_token_convention(self):
        before = {"num_drafts": 2.0, "num_draft_tokens": 14.0, "num_accepted_tokens": 5.0}
        after = {"num_drafts": 12.0, "num_draft_tokens": 84.0, "num_accepted_tokens": 45.0}
        value = summarize_spec_decode(before, after)
        self.assertEqual(value["drafts"], 10)
        self.assertEqual(value["accepted_tokens"], 40)
        self.assertEqual(value["draft_tokens"], 70)
        self.assertAlmostEqual(value["mean_acceptance_length"], 5.0)
        self.assertAlmostEqual(value["draft_acceptance_rate"], 40 / 70)

    def test_comparison_reports_speedup_and_greedy_parity(self):
        baseline = {
            "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 42, "max_tokens": 16},
            "fixture_sha256": "fixture",
            "summary": {"completion_tokens": 200, "wall_seconds": 10.0, "tps": 20.0},
            "samples": [{"sample_id": "a", "output_text": "same", "output_token_ids": [1, 2]}],
        }
        speculative = {
            "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 42, "max_tokens": 16},
            "fixture_sha256": "fixture",
            "rejection_mode": "standard",
            "summary": {"completion_tokens": 200, "wall_seconds": 5.0, "tps": 40.0},
            "samples": [{"sample_id": "a", "output_text": "same", "output_token_ids": [1, 2]}],
            "spec_decode": {
                "mean_acceptance_length": 6.0,
                "draft_acceptance_rate": 5 / 7,
                "drafts": 10.0,
                "draft_tokens": 70.0,
                "accepted_tokens": 50.0,
            },
        }
        result = compare_benchmark_results(baseline, speculative, require_exact_outputs=True)
        self.assertEqual(result["speedup"], 2.0)
        self.assertTrue(result["exact_output_match"])
        self.assertEqual(result["mean_acceptance_length"], 6.0)

    def test_comparison_rejects_speculative_run_without_active_draft_metrics(self):
        baseline = {
            "sampling": {"temperature": 0.0},
            "fixture_sha256": "fixture",
            "summary": {"tps": 20.0},
            "samples": [{"sample_id": "a", "output_text": "same", "output_token_ids": [1]}],
        }
        speculative = {
            "sampling": {"temperature": 0.0},
            "fixture_sha256": "fixture",
            "summary": {"tps": 24.0},
            "samples": [{"sample_id": "a", "output_text": "same", "output_token_ids": [1]}],
        }
        with self.assertRaisesRegex(ValueError, "speculative run.*metrics"):
            compare_benchmark_results(
                baseline, speculative, require_exact_outputs=True
            )

    def test_greedy_parity_uses_token_ids_not_text(self):
        baseline = {
            "sampling": {"temperature": 0.0},
            "fixture_sha256": "fixture",
            "summary": {"tps": 20.0},
            "samples": [{"sample_id": "a", "output_text": "same", "output_token_ids": [1]}],
        }
        speculative = {
            "sampling": {"temperature": 0.0},
            "fixture_sha256": "fixture",
            "rejection_mode": "standard",
            "summary": {"tps": 24.0},
            "samples": [{"sample_id": "a", "output_text": "same", "output_token_ids": [2]}],
            "spec_decode": {"drafts": 1, "draft_tokens": 7, "accepted_tokens": 1,
                            "mean_acceptance_length": 2, "draft_acceptance_rate": 1 / 7},
        }
        with self.assertRaisesRegex(ValueError, "token-ID"):
            compare_benchmark_results(baseline, speculative, require_exact_outputs=True)

    def test_launcher_uses_sequential_runs_and_requires_bound_attestation(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts/eval_vllm_ascend.sh").read_text()
        self.assertIn('run_server "baseline"', script)
        self.assertIn('run_server "speculative"', script)
        self.assertIn('validate-attestation', script)
        self.assertIn('deploy_attestation.json', script)
        self.assertIn('--speculative-config', script)
        self.assertIn('ASCEND_RT_VISIBLE_DEVICES', script)


if __name__ == "__main__":
    unittest.main()
