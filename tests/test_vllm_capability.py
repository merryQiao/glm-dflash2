from __future__ import annotations

import unittest

from glm_dflash2.vllm_ascend.capability import (
    normalize_runtime_identity,
    runtime_identities_match,
)


def runtime_identity(**overrides):
    value = {
        "vllm_version": "0.11.0",
        "vllm_commit": "vllm-sha",
        "vllm_ascend_version": "0.11.0rc1",
        "vllm_ascend_commit": "ascend-sha",
        "speculators_version": "0.5.0",
        "speculators_commit": "spec-sha",
        "adapter_revision": "adapter-v1",
        "cann_version": "8.5.0",
        "torch_npu_version": "2.7.1",
        "driver_version": "25.0.rc1",
        "firmware_version": "7.7.0",
        "device_name": "Ascend 910B4",
        "attention_backend": "ascend",
        "model_runner": "v1",
        "tp_size": 16,
        "ep_size": 16,
        "pp_size": 1,
        "dp_size": 1,
        "nnodes": 1,
        "graph_mode": "disabled",
        "chunked_prefill": False,
        "prefix_cache": False,
    }
    value.update(overrides)
    return value


class CapabilityTest(unittest.TestCase):
    def test_production_identity_is_canonical_and_complete(self):
        value = normalize_runtime_identity(runtime_identity(), production=True)
        self.assertEqual(value["tp_size"], 16)
        self.assertEqual(value["device_name"], "Ascend 910B4")

    def test_production_identity_rejects_unknown_or_unsafe_fields(self):
        for field, value in (
            ("vllm_commit", "unknown"),
            ("device_name", ""),
            ("prefix_cache", True),
            ("chunked_prefill", True),
            ("graph_mode", "piecewise"),
            ("pp_size", 2),
            ("dp_size", 2),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize_runtime_identity(runtime_identity(**{field: value}), production=True)

    def test_identity_comparison_names_the_drift(self):
        expected = normalize_runtime_identity(runtime_identity(), production=True)
        actual = normalize_runtime_identity(runtime_identity(cann_version="8.6.0"), production=True)
        matched, drift = runtime_identities_match(expected, actual)
        self.assertFalse(matched)
        self.assertEqual(drift, {"cann_version": {"expected": "8.5.0", "actual": "8.6.0"}})


if __name__ == "__main__":
    unittest.main()
