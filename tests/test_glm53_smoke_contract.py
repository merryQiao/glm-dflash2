from __future__ import annotations

import unittest
from types import SimpleNamespace

from tools import generate_trajectories


class GLM53SmokeContractTest(unittest.TestCase):
    def _args(self, **updates):
        values = {
            "endpoint": "http://127.0.0.1:30000",
            "endpoint_manifest": None,
            "allow_unverified_endpoint": True,
            "max_samples": 10,
        }
        values.update(updates)
        return SimpleNamespace(**values)

    def test_unverified_endpoint_is_limited_to_fifty_samples(self):
        generate_trajectories._validate_endpoint_mode(self._args(max_samples=50))
        for max_samples in (None, 0, 51):
            with self.subTest(max_samples=max_samples):
                with self.assertRaisesRegex(ValueError, "1..50"):
                    generate_trajectories._validate_endpoint_mode(
                        self._args(max_samples=max_samples)
                    )

    def test_unverified_mode_and_endpoint_manifest_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            generate_trajectories._validate_endpoint_mode(
                self._args(endpoint_manifest="manifest.json")
            )

    def test_sample_limit_is_total_artifact_budget_across_resume(self):
        self.assertFalse(
            generate_trajectories._sample_budget_exhausted(
                max_samples=50, committed_count=49, attempted_count=0
            )
        )
        self.assertTrue(
            generate_trajectories._sample_budget_exhausted(
                max_samples=50, committed_count=49, attempted_count=1
            )
        )
        self.assertTrue(
            generate_trajectories._sample_budget_exhausted(
                max_samples=50, committed_count=50, attempted_count=0
            )
        )

    def test_unverified_endpoint_can_never_be_frozen_or_production_eligible(self):
        args = self._args()
        self.assertEqual(
            generate_trajectories._terminal_status(
                False, args, committed_count=10, unresolved_errors=0
            ),
            "smoke_unverified",
        )
        self.assertEqual(
            generate_trajectories._terminal_status(
                False, args, committed_count=0, unresolved_errors=1
            ),
            "smoke_failed",
        )
        self.assertFalse(generate_trajectories._production_eligible(args))

    def test_attested_or_local_complete_run_can_be_frozen(self):
        local = self._args(
            endpoint=None,
            endpoint_manifest=None,
            allow_unverified_endpoint=False,
            max_samples=None,
        )
        generate_trajectories._validate_endpoint_mode(local)
        self.assertEqual(generate_trajectories._terminal_status(True, local), "frozen")
        self.assertTrue(generate_trajectories._production_eligible(local))


if __name__ == "__main__":
    unittest.main()
