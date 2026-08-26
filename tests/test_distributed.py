from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from glm_dflash2.distributed import (
    configure_accumulation,
    fsdp2_method_modules,
    global_weighted_mean,
    rank_epoch_seed,
    resolve_device_backend,
    reduce_additive_metrics,
)


class _FakeFSDP2:
    def __init__(self):
        self.calls = []

    def set_requires_gradient_sync(self, value, recurse=True):
        self.calls.append(("sync", value, recurse))

    def set_reshard_after_forward(self, value, recurse=True):
        self.calls.append(("reshard", value, recurse))


class DistributedTest(unittest.TestCase):
    def test_fsdp2_method_modules_cover_heads_called_after_backbone_forward(self):
        class Draft(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.candidate_selector = torch.nn.Linear(2, 2)
                self.markov_head = torch.nn.Linear(2, 2)
                self.confidence_head = torch.nn.Linear(2, 1)

        modules = fsdp2_method_modules(Draft())
        self.assertEqual(
            [(name, keep_unsharded) for name, _, keep_unsharded in modules],
            [
                ("candidate_selector", False),
                ("markov_head", True),
                ("confidence_head", False),
            ],
        )

    def test_cpu_backend_and_rank_epoch_seed(self):
        device, backend = resolve_device_backend("cpu", local_rank=0)
        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(backend, "gloo")
        self.assertEqual(rank_epoch_seed(10, 2, 3), rank_epoch_seed(10, 2, 3))
        self.assertNotEqual(rank_epoch_seed(10, 2, 3), rank_epoch_seed(10, 3, 3))
        self.assertNotEqual(rank_epoch_seed(10, 2, 3), rank_epoch_seed(10, 2, 4))

    def test_npu_import_is_lazy_and_actionable(self):
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch(
            "builtins.__import__", side_effect=ImportError("missing torch_npu")
        ):
            with self.assertRaisesRegex(RuntimeError, "torch_npu"):
                resolve_device_backend("npu", local_rank=0)

    def test_accumulation_uses_fsdp2_sync_and_reshard_api(self):
        module = _FakeFSDP2()
        configure_accumulation(module, synchronize=False)
        self.assertEqual(module.calls, [("sync", False, True), ("reshard", False, True)])
        module.calls.clear()
        configure_accumulation(module, synchronize=True)
        self.assertEqual(module.calls, [("sync", True, True), ("reshard", True, True)])

    def test_additive_metrics_are_identity_without_process_group(self):
        metrics = {"numerator": torch.tensor(2.0), "denominator": torch.tensor(3.0)}
        result = reduce_additive_metrics(metrics)
        self.assertEqual(result["numerator"].item(), 2.0)
        self.assertIsNot(result["numerator"], metrics["numerator"])

    def test_global_weighted_mean_compensates_for_fsdp_gradient_averaging(self):
        local_mean = torch.tensor(3.0, requires_grad=True)
        local_weight = torch.tensor(2.0)

        def add_remote_weight(value, op=None):
            del op
            value.add_(6.0)

        with mock.patch("glm_dflash2.distributed.dist.is_initialized", return_value=True), \
             mock.patch("glm_dflash2.distributed.dist.get_world_size", return_value=2), \
             mock.patch("glm_dflash2.distributed.dist.all_reduce", side_effect=add_remote_weight):
            scaled = global_weighted_mean(local_mean, local_weight)
        self.assertEqual(scaled.item(), 1.5)
        scaled.backward()
        self.assertEqual(local_mean.grad.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
