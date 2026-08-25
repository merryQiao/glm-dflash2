from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from glm_dflash2.sglang_hidden_runner import (
    SGLangInternalHiddenRunner,
    initialize_forward_batch,
    req_token_array,
)


class FakeModel:
    def __init__(self):
        self.layers = None

    def set_dflash_layers_to_capture(self, layers):
        self.layers = list(layers)


class FakeGlmModel:
    def __init__(self):
        self.layers = None

    def set_eagle3_layers_to_capture(self, layers):
        self.layers = list(layers)
        if hasattr(self, "model"):
            self.model.layers_to_capture = [int(value) + 1 for value in layers]


class SGLangHiddenRunnerTest(unittest.TestCase):
    def test_req_token_array_uses_sglang_signed_int64_contract(self):
        value = req_token_array([1, 2, 2**40])
        self.assertEqual(value.typecode, "q")
        self.assertEqual(list(value), [1, 2, 2**40])

    def test_forward_batch_bridge_uses_current_schedule_batch_directly(self):
        schedule_batch = object()

        class FakeForwardBatch:
            @staticmethod
            def init_new(batch, runner):
                return batch, runner

        runner = object()
        self.assertEqual(
            initialize_forward_batch(schedule_batch, runner, FakeForwardBatch),
            (schedule_batch, runner),
        )

    def test_forward_batch_bridge_uses_legacy_worker_batch_when_available(self):
        worker_batch = object()
        schedule_batch = SimpleNamespace(
            get_model_worker_batch=lambda: worker_batch
        )

        class FakeForwardBatch:
            @staticmethod
            def init_new(batch, runner):
                return batch, runner

        runner = object()
        self.assertEqual(
            initialize_forward_batch(schedule_batch, runner, FakeForwardBatch),
            (worker_batch, runner),
        )

    def test_forward_batch_bridge_supports_new_ascend_signature(self):
        schedule_batch = object()

        class FakeForwardBatch:
            @staticmethod
            def init_new(
                batch,
                runner,
                *,
                return_hidden_states_before_norm,
            ):
                return batch, runner, return_hidden_states_before_norm

        runner = object()
        self.assertEqual(
            initialize_forward_batch(schedule_batch, runner, FakeForwardBatch),
            (schedule_batch, runner, False),
        )

    def test_constructor_is_lazy_and_configures_explicit_dflash_layers(self):
        model = FakeModel()
        torch_runner = SimpleNamespace(
            model=model,
            model_config=SimpleNamespace(hidden_size=6144),
        )
        wrapped = SimpleNamespace(torch_runner=torch_runner)
        one_batch = types.ModuleType("sglang.benchmark.one_batch")
        one_batch.load_model = lambda *args: (wrapped, "tokenizer")
        benchmark = types.ModuleType("sglang.benchmark")
        sglang = types.ModuleType("sglang")
        sglang.__version__ = "fake-version"
        modules = {
            "sglang": sglang,
            "sglang.benchmark": benchmark,
            "sglang.benchmark.one_batch": one_batch,
        }
        server_args = SimpleNamespace(
            tp_size=16,
            ep_size=16,
            pp_size=1,
            dp_size=1,
            chunked_prefill_size=-1,
        )
        with patch.dict(sys.modules, modules):
            runner = SGLangInternalHiddenRunner(
                server_args=server_args,
                port_args=object(),
                gpu_id=0,
                tp_rank=0,
                logical_layer_ids=(1, 20, 38, 56, 75),
            )
        self.assertEqual(model.layers, [1, 20, 38, 56, 75])
        self.assertEqual(runner.physical_layer_ids, (2, 21, 39, 57, 76))
        self.assertEqual(runner.hidden_size, 6144)
        self.assertEqual(runner.backend_metadata["sglang_version"], "fake-version")

    def test_rejects_a_layer_order_that_cannot_be_identified_from_packed_output(self):
        model = FakeModel()
        direct_runner = SimpleNamespace(
            model=model,
            model_config=SimpleNamespace(hidden_size=6144),
        )
        one_batch = types.ModuleType("sglang.bench_one_batch")
        one_batch.load_model = lambda *args: (direct_runner, "tokenizer")
        sglang = types.ModuleType("sglang")
        sglang.__path__ = []
        modules = {"sglang": sglang, "sglang.bench_one_batch": one_batch}
        server_args = SimpleNamespace(
            tp_size=16,
            ep_size=16,
            pp_size=1,
            dp_size=1,
            chunked_prefill_size=-1,
        )
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(ValueError, "ordered logical layers"):
                SGLangInternalHiddenRunner(
                    server_args=server_args,
                    port_args=object(),
                    gpu_id=0,
                    tp_rank=0,
                    logical_layer_ids=(75, 56, 38, 20, 1),
                )

    def test_constructor_supports_legacy_bench_one_batch_direct_runner(self):
        model = FakeModel()
        direct_runner = SimpleNamespace(
            model=model,
            model_config=SimpleNamespace(hidden_size=6144),
        )
        one_batch = types.ModuleType("sglang.bench_one_batch")
        one_batch.load_model = lambda *args: (direct_runner, "tokenizer")
        sglang = types.ModuleType("sglang")
        sglang.__path__ = []
        modules = {
            "sglang": sglang,
            "sglang.bench_one_batch": one_batch,
        }
        server_args = SimpleNamespace(
            tp_size=16,
            ep_size=16,
            pp_size=1,
            dp_size=1,
            chunked_prefill_size=-1,
        )
        with patch.dict(sys.modules, modules):
            runner = SGLangInternalHiddenRunner(
                server_args=server_args,
                port_args=object(),
                gpu_id=0,
                tp_rank=0,
                logical_layer_ids=(1, 20, 38, 56, 75),
            )
        self.assertIs(runner._runner, direct_runner)
        self.assertEqual(model.layers, [1, 20, 38, 56, 75])

    def test_constructor_uses_existing_glm_eagle3_capture_hook(self):
        model = FakeGlmModel()
        model.model = SimpleNamespace(layers_to_capture=[])
        direct_runner = SimpleNamespace(
            model=model,
            model_config=SimpleNamespace(hidden_size=6144),
        )
        one_batch = types.ModuleType("sglang.bench_one_batch")
        one_batch.load_model = lambda *args: (direct_runner, "tokenizer")
        sglang = types.ModuleType("sglang")
        sglang.__path__ = []
        modules = {"sglang": sglang, "sglang.bench_one_batch": one_batch}
        server_args = SimpleNamespace(
            tp_size=32, ep_size=32, pp_size=1, dp_size=1, chunked_prefill_size=-1
        )
        with patch.dict(sys.modules, modules):
            runner = SGLangInternalHiddenRunner(
                server_args=server_args,
                port_args=object(),
                gpu_id=0,
                tp_rank=0,
                logical_layer_ids=(1, 20, 38, 56, 75),
            )
        self.assertEqual(model.layers, [1, 20, 38, 56, 75])
        self.assertEqual(runner.backend_metadata["capture_hook"], "set_eagle3_layers_to_capture")


if __name__ == "__main__":
    unittest.main()
