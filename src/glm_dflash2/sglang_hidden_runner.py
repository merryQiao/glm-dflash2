from __future__ import annotations

from array import array
from collections.abc import Sequence
import importlib
import inspect
from typing import Any

import torch

from .hidden_capture import CaptureTap, TargetHiddenCapture


GLM52_DFLASH_LOGICAL_LAYERS = (1, 20, 38, 56, 75)


def req_token_array(input_ids: Sequence[int]) -> array:
    """Build the signed-int64 token container required by current SGLang."""

    return array("q", (int(value) for value in input_ids))


def initialize_forward_batch(
    schedule_batch: Any, runner: Any, forward_batch_class: Any
) -> Any:
    """Bridge the legacy and current SGLang standalone-runner contracts."""

    worker_batch = (
        schedule_batch.get_model_worker_batch()
        if hasattr(schedule_batch, "get_model_worker_batch")
        else schedule_batch
    )
    parameters = inspect.signature(forward_batch_class.init_new).parameters
    kwargs = {}
    if "return_hidden_states_before_norm" in parameters:
        kwargs["return_hidden_states_before_norm"] = False
    return forward_batch_class.init_new(worker_batch, runner, **kwargs)


def one_batch_module() -> Any:
    """Load SGLang's standalone runner across its two public module layouts."""

    errors: list[Exception] = []
    for name in ("sglang.benchmark.one_batch", "sglang.bench_one_batch"):
        try:
            return importlib.import_module(name)
        except (ImportError, ModuleNotFoundError) as exc:
            errors.append(exc)
    raise ImportError(
        "SGLang standalone one-batch runner is unavailable; tried "
        "sglang.benchmark.one_batch and sglang.bench_one_batch"
    ) from errors[-1]


class SGLangInternalHiddenRunner:
    """Thin adapter over SGLang's device-aware standalone ModelRunner.

    Imports are intentionally lazy: CPU cache tooling must not import SGLang,
    torch-npu, or initialize a distributed device runtime.
    """

    def __init__(
        self,
        *,
        server_args: Any,
        port_args: Any,
        gpu_id: int,
        tp_rank: int,
        logical_layer_ids: Sequence[int],
    ) -> None:
        one_batch = one_batch_module()
        wrapped, self.tokenizer = one_batch.load_model(
            server_args, port_args, gpu_id, tp_rank
        )
        self._wrapped = wrapped
        self._runner = getattr(wrapped, "torch_runner", wrapped)
        if not hasattr(self._runner, "model"):
            raise RuntimeError("selected SGLang runner is not the PyTorch runner")
        self.tp_rank = int(tp_rank)
        self.logical_layer_ids = tuple(int(value) for value in logical_layer_ids)
        if self.logical_layer_ids != GLM52_DFLASH_LOGICAL_LAYERS:
            raise ValueError(
                "GLM-5.2 DFlash extraction requires ordered logical layers "
                f"{GLM52_DFLASH_LOGICAL_LAYERS}, got {self.logical_layer_ids}"
            )
        model = self._runner.model
        capture_hook = None
        for hook_name in (
            "set_dflash_layers_to_capture",
            "set_eagle3_layers_to_capture",
        ):
            candidate = getattr(model, hook_name, None)
            if callable(candidate):
                candidate(list(self.logical_layer_ids))
                capture_hook = hook_name
                break
        if capture_hook is None:
            raise RuntimeError(
                f"{type(model).__name__} does not expose a supported hidden-layer capture hook"
            )
        self.physical_layer_ids = tuple(value + 1 for value in self.logical_layer_ids)
        capture_owner = getattr(model, "model", None)
        reported_layers = getattr(capture_owner, "layers_to_capture", None)
        if reported_layers is not None and tuple(reported_layers) != self.physical_layer_ids:
            raise RuntimeError(
                "SGLang model configured unexpected physical capture layers: "
                f"{tuple(reported_layers)} != {self.physical_layer_ids}"
            )
        self.hidden_size = int(self._runner.model_config.hidden_size)
        self.capture_mapping = tuple(
            CaptureTap(
                backend_namespace="sglang",
                logical_layer_id=logical,
                concrete_tap=f"model.model.layers_to_capture[{physical}]",
                tap_semantics="post_decoder_block",
            )
            for logical, physical in zip(
                self.logical_layer_ids, self.physical_layer_ids, strict=True
            )
        )
        final_norm = getattr(capture_owner, "norm", None)
        if final_norm is None or not hasattr(final_norm, "register_forward_hook"):
            raise RuntimeError(
                "SGLang GLM model does not expose model.norm for post-final-norm capture"
            )
        self.final_hidden_tap = "model.model.norm.forward_output"
        self._post_norm_hidden: torch.Tensor | None = None
        self._final_hidden_hook = final_norm.register_forward_hook(self._capture_post_norm)
        try:
            import sglang

            version = str(getattr(sglang, "__version__", "unknown"))
        except Exception:
            version = "unknown"
        self.backend_metadata = {
            "backend": "sglang_internal_model_runner",
            "sglang_version": version,
            "model_class": type(model).__name__,
            "tp_size": int(server_args.tp_size),
            "ep_size": int(server_args.ep_size),
            "pp_size": int(server_args.pp_size),
            "dp_size": int(server_args.dp_size),
            "chunked_prefill_size": int(server_args.chunked_prefill_size),
            "capture_mode": "FULL",
            "capture_hook": capture_hook,
            "capture_mapping": [tap.as_tuple() for tap in self.capture_mapping],
            "final_hidden_tap": self.final_hidden_tap,
            "final_hidden_semantics": "post_final_norm_lm_head_input",
        }

    def _capture_post_norm(self, *args: Any) -> None:
        output = args[-1]
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise RuntimeError("SGLang model.norm returned an unsupported hidden tensor")
        self._post_norm_hidden = value

    def _normalize_capture(
        self, auxiliary: torch.Tensor, *, token_count: int
    ) -> TargetHiddenCapture:
        expected_width = len(self.logical_layer_ids) * self.hidden_size
        if tuple(auxiliary.shape) != (token_count, expected_width):
            raise RuntimeError(
                f"captured auxiliary shape {tuple(auxiliary.shape)} != "
                f"{(token_count, expected_width)}"
            )
        if self._post_norm_hidden is None:
            raise RuntimeError("SGLang returned no post-final-norm hidden capture")
        if tuple(self._post_norm_hidden.shape) != (token_count, self.hidden_size):
            raise RuntimeError(
                f"captured final hidden shape {tuple(self._post_norm_hidden.shape)} != "
                f"{(token_count, self.hidden_size)}"
            )
        capture = TargetHiddenCapture(
            aux_hidden_states=auxiliary.reshape(
                token_count, len(self.logical_layer_ids), self.hidden_size
            ),
            target_final_hidden=self._post_norm_hidden,
            capture_mapping=self.capture_mapping,
        ).cpu_bfloat16()
        self._post_norm_hidden = None
        return capture

    @torch.no_grad()
    def extract(self, input_ids: Sequence[int]) -> TargetHiddenCapture:
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        one_batch = one_batch_module()

        if not input_ids:
            raise ValueError("cannot extract an empty sequence")
        self._clear_pools()
        self._post_norm_hidden = None
        sampling = SamplingParams(temperature=0, max_new_tokens=1)
        req = Req(
            rid=0,
            origin_input_text="",
            # Current SGLang's Req mutates/concatenates this sequence and
            # requires its native signed-int64 array contract.  Older builds
            # also accept the array because it implements the list-like API.
            origin_input_ids=req_token_array(input_ids),
            sampling_params=sampling,
        )
        req.logprob_start_len = -1
        if hasattr(req, "set_extend_range"):
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.set_extend_range(0, len(req.origin_input_ids))
        else:
            req.fill_ids = req.origin_input_ids
            req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        page_size = int(getattr(self._runner.server_args, "page_size", 1))
        tree_cache = one_batch.TreeCacheNamespace(
            page_size=page_size,
            device=self._runner.device,
            token_to_kv_pool_allocator=self._runner.token_to_kv_pool_allocator,
        )
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=self._runner.req_to_token_pool,
            token_to_kv_pool_allocator=self._runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self._runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        # ScheduleBatch converts this flag into CaptureHiddenMode.FULL on the
        # ModelWorkerBatch. This is compatible with both legacy
        # ``sglang.bench_one_batch`` and the newer benchmark module layout.
        batch.return_hidden_states = True
        batch.prepare_for_extend()
        one_batch._maybe_prepare_mlp_sync_batch(batch, self._runner)
        if batch.input_ids is None and getattr(batch, "prefill_input_ids_cpu", None) is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
            batch.prefill_input_ids_cpu = None
        forward_batch = initialize_forward_batch(batch, self._runner, ForwardBatch)
        output = self._runner.forward(forward_batch).logits_output.hidden_states
        if output is None:
            raise RuntimeError("SGLang returned no captured hidden states")
        capture = self._normalize_capture(output, token_count=len(input_ids))
        if self.tp_rank == 0:
            result = capture
        else:
            result = TargetHiddenCapture(
                aux_hidden_states=torch.empty(
                    0, len(self.logical_layer_ids), self.hidden_size, dtype=torch.bfloat16
                ),
                target_final_hidden=torch.empty(
                    0, self.hidden_size, dtype=torch.bfloat16
                ),
                capture_mapping=self.capture_mapping,
            )
        self._clear_pools()
        return result

    def _clear_pools(self) -> None:
        if hasattr(self._wrapped, "clear"):
            self._wrapped.clear()
            return
        self._runner.req_to_token_pool.clear()
        self._runner.token_to_kv_pool_allocator.clear()
