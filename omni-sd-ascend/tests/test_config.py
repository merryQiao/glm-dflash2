from __future__ import annotations

import copy
import unittest

from tests.helpers import complete_config
from omni_sd.config import ConfigError, validate_config
from omni_sd.ascend_runtime import runtime_identity


class ConfigTests(unittest.TestCase):
    def test_complete_config_is_valid(self):
        validated = validate_config(complete_config())
        self.assertEqual(validated["runtime"]["tensor_parallel_size"], 4)

    def test_missing_runtime_section_is_rejected_before_model_loading(self):
        config = complete_config()
        config.pop("runtime")
        with self.assertRaisesRegex(ConfigError, "runtime"):
            validate_config(config)

    def test_bf16_rejects_ascend_quantization(self):
        config = complete_config()
        config["runtime"]["quantization"] = "ascend"
        with self.assertRaisesRegex(ConfigError, "BF16"):
            validate_config(config)

    def test_layer_ids_must_be_unique_and_in_range(self):
        config = complete_config()
        config["hidden_states"]["layer_ids"] = [1, 1, 48]
        with self.assertRaisesRegex(ConfigError, "layer_ids"):
            validate_config(config)

    def test_runtime_identity_records_device_topology_and_versions(self):
        identity = runtime_identity(
            env={"ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7", "HCCL_OP_EXPANSION_MODE": "AIV"},
            versions={"vllm": "0.23.0", "vllm_ascend": "0.23.0", "torch": "2.8"},
            hardware="a3",
        )
        self.assertEqual(identity["visible_devices"], [4, 5, 6, 7])
        self.assertEqual(identity["hardware"], "a3")
        self.assertEqual(identity["versions"]["vllm_ascend"], "0.23.0")


if __name__ == "__main__":
    unittest.main()
