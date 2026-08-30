from __future__ import annotations

import subprocess
import sys
import types
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from glm53_drafters.hidden_capture import CaptureAttestation
from glm53_drafters.hidden_capture import _live_cann_version
from glm53_drafters.hidden_capture import validate_ascend_a2_evidence
from glm53_drafters.sglang_hidden_runner import (
    GLM53_DFLASH_LOGICAL_LAYERS,
    SGLangInternalHiddenRunner,
    attest_capture_semantics,
    initialize_forward_batch,
    req_token_array,
    validate_resolved_stage_b_runner,
    validate_stage_b_server_args,
)
from tools.extract_hidden_sglang import _next_source_control


ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __init__(
        self,
        *,
        hook_name: str = "set_dflash_layers_to_capture",
        nested_language_model: bool = False,
    ):
        backbone = nn.Module()
        backbone.norm = nn.Identity()
        backbone.layers = nn.ModuleList(nn.Identity() for _ in range(45))
        backbone.layers_to_capture = []
        self.model = (
            SimpleNamespace(language_model=backbone)
            if nested_language_model
            else backbone
        )
        self._backbone = backbone
        self.layers = None
        if hook_name == "set_eagle3_layers_to_capture":
            self.set_eagle3_layers_to_capture = self._set_layers
        else:
            self.set_dflash_layers_to_capture = self._set_layers

    def _set_layers(self, layers):
        self.layers = list(layers)
        self._backbone.layers_to_capture = [int(layer) + 1 for layer in layers]


class AscendAttentionBackend:
    pass


AscendAttentionBackend.__module__ = "sglang.srt.layers.attention.ascend_backend"


class FakeTorchRunner:
    def __init__(self, model, server_args, *, device="npu:0", attention_backend=None):
        self.model = model
        self.server_args = server_args
        self.device = device
        self.attention_backend = (
            AscendAttentionBackend()
            if attention_backend is None
            else attention_backend
        )
        self.model_config = SimpleNamespace(
            hidden_size=4, vocab_size=7, dtype=torch.bfloat16
        )

    def compute_logits(self, hidden_states):
        weight = torch.arange(28, dtype=torch.float32).reshape(7, 4)
        return hidden_states.float() @ weight.t()


class GLM53SGLangHiddenRunnerTest(unittest.TestCase):
    @staticmethod
    def _server_args(**updates):
        values = {
            "tp_size": 16,
            "ep_size": 16,
            "pp_size": 1,
            "dp_size": 1,
            "nnodes": 1,
            "node_rank": 0,
            "chunked_prefill_size": -1,
            "disable_radix_cache": True,
            "disable_cuda_graph": True,
            "max_running_requests": 1,
            "device": "npu",
            "attention_backend": "ascend",
            "dtype": "bfloat16",
            "model_runner": "torch",
        }
        values.update(updates)
        return SimpleNamespace(**values)

    def _runner(
        self,
        *,
        hook_name="set_dflash_layers_to_capture",
        nested_language_model=False,
        resolved_server_updates=None,
        resolved_device="npu:0",
        resolved_attention_backend=None,
        resolved_dtype=torch.bfloat16,
        comparable_logits=True,
        direct_result=False,
        trusted_runner_identity=True,
        tp_rank=0,
    ):
        model = FakeModel(
            hook_name=hook_name,
            nested_language_model=nested_language_model,
        )
        resolved_server = self._server_args(tp_size=1, ep_size=1)
        for name, value in (resolved_server_updates or {}).items():
            setattr(resolved_server, name, value)
        runner_class = FakeTorchRunner
        if trusted_runner_identity:
            runner_class = type("ModelRunner", (FakeTorchRunner,), {})
            runner_class.__module__ = "sglang.srt.model_executor.model_runner"
        direct = runner_class(
            model,
            resolved_server,
            device=resolved_device,
            attention_backend=resolved_attention_backend,
        )
        if not comparable_logits:
            direct.compute_logits = None
        direct.model_config.dtype = resolved_dtype
        wrapped = direct if direct_result else SimpleNamespace(torch_runner=direct)
        one_batch = types.ModuleType("sglang.benchmark.one_batch")
        one_batch.load_model = lambda *args: (wrapped, "tokenizer")
        benchmark = types.ModuleType("sglang.benchmark")
        sglang = types.ModuleType("sglang")
        sglang.__path__ = []
        sglang.__version__ = "test-version"
        model_runner_module = types.ModuleType(
            "sglang.srt.model_executor.model_runner"
        )
        model_runner_module.ModelRunner = runner_class
        attention_backend_module = types.ModuleType(
            "sglang.srt.layers.attention.ascend_backend"
        )
        attention_backend_module.AscendAttentionBackend = AscendAttentionBackend
        modules = {
            "sglang": sglang,
            "sglang.benchmark": benchmark,
            "sglang.benchmark.one_batch": one_batch,
            "sglang.srt.model_executor.model_runner": model_runner_module,
            "sglang.srt.layers.attention.ascend_backend": attention_backend_module,
        }
        with patch.dict(sys.modules, modules):
            runner = SGLangInternalHiddenRunner(
                server_args=self._server_args(tp_size=1, ep_size=1),
                port_args=object(),
                gpu_id=0,
                tp_rank=tp_rank,
                logical_layer_ids=GLM53_DFLASH_LOGICAL_LAYERS,
            )
        return runner, model

    def test_module_import_is_lazy_for_sglang_and_torch_npu(self):
        command = (
            "import sys; import glm53_drafters.sglang_hidden_runner; "
            "assert 'sglang' not in sys.modules; assert 'torch_npu' not in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", command],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
            check=True,
        )

    def test_stage_b_contract_requires_strict_npu_ascend_single_request(self):
        contract = validate_stage_b_server_args(self._server_args())
        self.assertEqual(contract["device_type"], "npu")
        self.assertEqual(contract["attention_backend"], "ascend")
        self.assertEqual(contract["dtype"], "bfloat16")
        self.assertEqual(contract["capture_mode"], "FULL")
        invalid = (
            ("dp_size", 2, "DP=1"),
            ("pp_size", 2, "PP=1"),
            ("chunked_prefill_size", 4096, "chunked prefill"),
            ("disable_radix_cache", False, "radix"),
            ("disable_cuda_graph", False, "graphs"),
            ("max_running_requests", 2, "one request"),
            ("device", "cpu", "device=npu"),
            ("attention_backend", "torch", "attention_backend=ascend"),
            ("dtype", "float16", "BF16"),
            ("quantization", "int8", "quant"),
        )
        for field, value, message in invalid:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                validate_stage_b_server_args(self._server_args(**{field: value}))

    def test_loaded_runtime_is_revalidated_and_provenance_uses_resolved_objects(self):
        runner, _ = self._runner()
        resolved = runner.backend_metadata["resolved_runtime"]
        requested = runner.backend_metadata["requested_runtime"]
        self.assertEqual(resolved["runner_implementation"], "pytorch")
        self.assertEqual(resolved["runner_selection_path"], "load_model_result.torch_runner")
        self.assertEqual(resolved["device_type"], "npu")
        self.assertEqual(resolved["attention_backend"], "ascend")
        self.assertEqual(resolved["dtype"], "bfloat16")
        self.assertEqual(resolved["runner_class"], "ModelRunner")
        self.assertEqual(requested["model_runner"], "torch")
        self.assertFalse(runner.backend_metadata["ascend_a2_runtime"]["passed"])

        class EagerAttentionBackend:
            pass

        invalid = (
            ({"resolved_device": "cpu"}, "resolved device"),
            (
                {"resolved_attention_backend": EagerAttentionBackend()},
                "resolved attention",
            ),
            ({"resolved_dtype": torch.float16}, "resolved dtype"),
            (
                {"resolved_server_updates": {"disable_radix_cache": False}},
                "radix",
            ),
        )
        for options, message in invalid:
            with self.subTest(options=options), self.assertRaisesRegex(
                ValueError, message
            ):
                self._runner(**options)

        with self.assertRaisesRegex(ValueError, "actual SGLang PyTorch runner"):
            self._runner(trusted_runner_identity=False)

    def test_runtime_class_name_strings_without_imported_class_identity_are_rejected(self):
        model = FakeModel()
        runner_class = type("ModelRunner", (FakeTorchRunner,), {})
        runner_class.__module__ = "sglang.srt.model_executor.model_runner"
        runner = runner_class(model, self._server_args(tp_size=1, ep_size=1))
        with self.assertRaisesRegex(ValueError, "imported.*ModelRunner"):
            validate_resolved_stage_b_runner(
                runner, runner_selection_path="load_model_result.torch_runner"
            )

    def test_a2_evidence_rejects_placeholder_and_non_version_values(self):
        base = {
            "schema": "glm53-ascend-910b-a2-runtime-v1",
            "passed": True,
            "device_name": "Ascend910B2",
            "device_index": 0,
            "device_count": 1,
            "runner_module": "sglang.srt.model_executor.model_runner",
            "runner_class": "ModelRunner",
            "attention_backend_module": "sglang.srt.layers.attention.ascend_backend",
            "attention_backend_class": "AscendAttentionBackend",
            "sglang_version": "0.5.2",
            "torch_npu_version": "2.6.0",
            "cann_version": "8.1.RC1",
        }
        for field, value in (
            ("sglang_version", "not-installed"),
            ("torch_npu_version", "unknown"),
            ("cann_version", "banana"),
            ("cann_version", "not-a-version-3"),
        ):
            with self.subTest(field=field):
                evidence = dict(base, **{field: value})
                with self.assertRaisesRegex(ValueError, "version"):
                    validate_ascend_a2_evidence(evidence)

    def test_cann_provenance_comes_from_torch_npu_runtime_not_environment(self):
        fake_torch_npu = types.ModuleType("torch_npu")
        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}), patch.dict(
            os.environ,
            {
                "ASCEND_TOOLKIT_VERSION": "8.1.RC1",
                "ASCEND_HOME_PATH": "/caller/controlled/toolkit",
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "get_cann_version"):
                _live_cann_version()

        fake_torch_npu.npu = SimpleNamespace(
            get_cann_version=lambda module: "8.1.RC1"
        )
        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            self.assertEqual(_live_cann_version(), "8.1.RC1")

    def test_version_explicit_direct_sglang_model_runner_is_accepted(self):
        runner, _ = self._runner(direct_result=True)
        self.assertEqual(
            runner.backend_metadata["resolved_runtime"]["runner_selection_path"],
            "load_model_result.direct_sglang_model_runner",
        )

    def test_req_tokens_and_forward_batch_bridge_versioned_contracts(self):
        value = req_token_array([1, 2, 154879])
        self.assertEqual(value.typecode, "q")
        with self.assertRaisesRegex(ValueError, "token ID"):
            req_token_array([154880])

        class ForwardBatch:
            @staticmethod
            def init_new(batch, runner, *, return_hidden_states_before_norm):
                return batch, runner, return_hidden_states_before_norm

        self.assertEqual(
            initialize_forward_batch("batch", "runner", ForwardBatch),
            ("batch", "runner", False),
        )

    def test_hook_negotiation_discovers_wrapper_norm_and_exact_physical_taps(self):
        runner, model = self._runner()
        self.assertEqual(model.layers, [1, 11, 22, 32, 42])
        self.assertEqual(runner.physical_layer_ids, (2, 12, 23, 33, 43))
        self.assertEqual(runner.hidden_size, 4)
        self.assertEqual(runner.backend_metadata["capture_hook"], "set_dflash_layers_to_capture")
        self.assertEqual(runner.capture_mapping[0].concrete_tap, "model.model.layers_to_capture[2]")
        self.assertEqual(runner.final_hidden_tap, "model.model.norm.forward_output")

    def test_existing_eagle_hook_is_negotiated_without_monkey_patch(self):
        runner, model = self._runner(hook_name="set_eagle3_layers_to_capture")
        self.assertEqual(model.layers, [1, 11, 22, 32, 42])
        self.assertEqual(runner.backend_metadata["capture_hook"], "set_eagle3_layers_to_capture")

    def test_nested_multimodal_wrapper_resolves_text_backbone_explicitly(self):
        runner, model = self._runner(nested_language_model=True)
        self.assertEqual(model.layers, [1, 11, 22, 32, 42])
        self.assertEqual(
            runner.capture_mapping[0].concrete_tap,
            "model.model.language_model.layers_to_capture[2]",
        )
        self.assertEqual(
            runner.final_hidden_tap,
            "model.model.language_model.norm.forward_output",
        )

    def test_normalized_capture_is_bf16_cpu_and_requires_post_final_norm(self):
        runner, model = self._runner()
        with self.assertRaisesRegex(RuntimeError, "post-final-norm"):
            runner._normalize_capture(torch.zeros(3, 20), token_count=3)
        auxiliary = torch.arange(60, dtype=torch.float32).reshape(3, 5, 4)
        for logical, value in zip(
            runner.logical_layer_ids, auxiliary.unbind(dim=1), strict=True
        ):
            runner._capture_independent_block(logical, value)
        runner._capture_post_norm(model.model.norm(torch.ones(3, 4)))
        native = runner._project_global_logits(torch.ones(1, 4))
        capture = runner._normalize_capture(
            auxiliary.reshape(3, 20),
            token_count=3,
            native_logits=native,
        )
        self.assertEqual(tuple(capture.aux_hidden_states.shape), (3, 5, 4))
        self.assertEqual(capture.aux_hidden_states.dtype, torch.bfloat16)
        self.assertEqual(capture.aux_hidden_states.device.type, "cpu")
        self.assertEqual(tuple(capture.target_final_hidden.shape), (3, 4))
        self.assertIsInstance(capture.attestation, CaptureAttestation)
        self.assertTrue(capture.attestation.passed)

    def test_numeric_attestation_detects_swapped_taps_and_final_logit_mismatch(self):
        auxiliary = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
        independent = tuple(auxiliary[:, index, :].clone() for index in range(5))
        final_hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        native_logits = torch.arange(7, dtype=torch.float32).reshape(1, 7)
        mapping = tuple((value, value + 1) for value in (1, 11, 22, 32, 42))
        passed = attest_capture_semantics(
            auxiliary=auxiliary,
            independent_aux=independent,
            final_hidden=final_hidden,
            projected_logits=native_logits.clone(),
            native_logits=native_logits,
            layer_mapping=mapping,
            native_logits_path="runner.compute_logits/global",
        )
        self.assertTrue(passed.passed)
        swapped = attest_capture_semantics(
            auxiliary=auxiliary,
            independent_aux=(independent[1], independent[0], *independent[2:]),
            final_hidden=final_hidden,
            projected_logits=native_logits,
            native_logits=native_logits,
            layer_mapping=mapping,
            native_logits_path="runner.compute_logits/global",
        )
        self.assertFalse(swapped.passed)
        self.assertIn("aux", swapped.reason)
        wrong_logits = attest_capture_semantics(
            auxiliary=auxiliary,
            independent_aux=independent,
            final_hidden=final_hidden,
            projected_logits=native_logits + 1,
            native_logits=native_logits,
            layer_mapping=mapping,
            native_logits_path="runner.compute_logits/global",
        )
        self.assertFalse(wrong_logits.passed)
        self.assertIn("logit", wrong_logits.reason)

    def test_tp_production_rejects_missing_global_logits_projection_api(self):
        runner, _ = self._runner(comparable_logits=False)
        with self.assertRaisesRegex(RuntimeError, "globally comparable.*logits"):
            runner._project_global_logits(torch.ones(1, 4))

    def test_version_explicit_global_logits_api_may_require_forward_batch(self):
        runner, _ = self._runner()
        seen = object()

        def compute_logits(hidden_states, forward_batch):
            self.assertIs(forward_batch, seen)
            weight = torch.arange(28, dtype=torch.float32).reshape(7, 4)
            return hidden_states.float() @ weight.t()

        runner._runner.compute_logits = compute_logits
        projected = runner._project_global_logits(
            torch.ones(1, 4), forward_batch=seen
        )
        self.assertEqual(tuple(projected.shape), (1, 7))

    def test_nonzero_tp_rank_participates_in_collective_projection_without_claiming_output(self):
        runner, _ = self._runner(tp_rank=1)
        calls = []

        def compute_logits(hidden_states):
            calls.append(tuple(hidden_states.shape))
            return None

        runner._runner.compute_logits = compute_logits
        projected = runner._project_global_logits(
            torch.ones(1, 4), require_output=False
        )
        self.assertIsNone(projected)
        self.assertEqual(calls, [(1, 4)])

    def test_cli_help_and_stage_b_launcher_are_standalone_and_ascend_specific(self):
        tool = ROOT / "tools/extract_hidden_sglang.py"
        launcher = ROOT / "scripts/run_stage_b_hidden.sh"
        subprocess.run(
            [sys.executable, str(tool), "--help"],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        text = launcher.read_text(encoding="utf-8")
        for required in (
            "GLM-5.3-Flash-BF16",
            "--device",
            "npu",
            "--attention-backend",
            "ascend",
            "1,11,22,32,42",
        ):
            self.assertIn(required, text)

    def test_rank_zero_control_keeps_resumed_tp_workers_on_one_input_path(self):
        rows = iter(
            [
                {"id": "done", "input_ids": [1], "loss_mask": [1]},
                {"id": "next", "input_ids": [2], "loss_mask": [1]},
            ]
        )
        first = _next_source_control(rows, done_ids={"done"}, committed=0, max_samples=1)
        second = _next_source_control(rows, done_ids={"done"}, committed=0, max_samples=1)
        third = _next_source_control(rows, done_ids={"done"}, committed=1, max_samples=1)
        self.assertEqual(first, {"action": "skip", "id": "done"})
        self.assertEqual(second["action"], "process")
        self.assertEqual(second["id"], "next")
        self.assertEqual(third, {"action": "stop", "reason": "max_samples"})


if __name__ == "__main__":
    unittest.main()
