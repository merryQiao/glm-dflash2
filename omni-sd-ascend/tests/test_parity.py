from __future__ import annotations

import unittest

from omni_sd.parity import ParityError, validate_attestation


class ParityTests(unittest.TestCase):
    def test_attestation_requires_all_modalities_and_exact_tokens(self):
        report = {
            "hardware": "a2",
            "modalities": {name: {"exact_tokens": True, "finite_hidden": True} for name in ("text", "image", "audio", "video")},
            "final_normalized_hidden": True,
        }
        validate_attestation(report)
        report["modalities"]["video"]["exact_tokens"] = False
        with self.assertRaisesRegex(ParityError, "video"):
            validate_attestation(report)


if __name__ == "__main__":
    unittest.main()
