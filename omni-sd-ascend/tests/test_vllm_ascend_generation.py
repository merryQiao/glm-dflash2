from __future__ import annotations

import unittest

from tests.helpers import complete_config
from omni_sd.vllm_ascend_generation import (
    condition_seed,
    engine_kwargs,
    sampling_kwargs,
)


class GenerationProviderTests(unittest.TestCase):
    def test_engine_owns_tp_and_never_receives_device_ids(self):
        kwargs = engine_kwargs(complete_config())
        self.assertNotIn("device_ids", kwargs)
        self.assertEqual(kwargs["tensor_parallel_size"], 4)
        self.assertTrue(kwargs["enable_expert_parallel"])
        self.assertEqual(kwargs["distributed_executor_backend"], "mp")

    def test_seed_depends_on_condition_not_batch_composition(self):
        config = complete_config()
        first = condition_seed(config, "condition-a")
        second = condition_seed(config, "condition-b")
        self.assertNotEqual(first, second)
        self.assertEqual(first, condition_seed(config, "condition-a"))

    def test_sampling_explicitly_sets_stop_token_ids(self):
        kwargs = sampling_kwargs(complete_config(), "condition-a")
        self.assertEqual(kwargs["stop_token_ids"], [151645])
        self.assertEqual(kwargs["seed"], condition_seed(complete_config(), "condition-a"))


if __name__ == "__main__":
    unittest.main()
