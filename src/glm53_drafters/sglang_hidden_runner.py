from __future__ import annotations

from array import array
from collections.abc import Mapping, Sequence
import importlib
import importlib.metadata
import inspect
from typing import Any

import torch

from .contracts import TARGET_CONTRACT, validate_token_ids
from .hidden_capture import (
    ASCEND_A2_ATTESTATION_SCHEMA,
    CaptureAttestation,
    CaptureTap,
    TargetHiddenCapture,
    _live_cann_version,
    validate_ascend_a2_evidence,
    validate_live_ascend_a2_evidence,
)


GLM53_DFLASH_LOGICAL_LAYERS = TARGET_CONTRACT.logical_layer_ids


def _parity_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    delta = (actual.float() - expected.float()).abs()
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    denominator = expected.float().abs().clamp_min(1e-6)
    relative = delta / denominator
    max_rel = float(relative.max().item()) if relative.numel() else 0.0
    return max_abs, max_rel


def attest_capture_semantics(
    *,
    auxiliary: torch.Tensor,
    independent_aux: Sequence[torch.Tensor],
    final_hidden: torch.Tensor,
    projected_logits: torch.Tensor,
    native_logits: torch.Tensor,
    layer_mapping: Sequence[tuple[int, int]],
    native_logits_path: str,
    independent_tap_paths: Sequence[str] | None = None,
    aux_atol: float = 0.02,
    aux_rtol: float = 0.02,
    logits_atol: float = 0.02,
    logits_rtol: float = 0.02,
) -> CaptureAttestation:
    """Numerically prove packed tap order and final-hidden logit semantics."""

    mapping = tuple((int(logical), int(physical)) for logical, physical in layer_mapping)
    logical = tuple(item[0] for item in mapping)
    physical = tuple(item[1] for item in mapping)
    paths = tuple(independent_tap_paths or (f"layers[{item}]" for item in logical))
    if auxiliary.ndim != 3 or final_hidden.ndim != 2:
        raise ValueError("capture parity inputs have invalid hidden-state ranks")
    if len(independent_aux) != auxiliary.shape[1] or len(mapping) != auxiliary.shape[1]:
        raise ValueError("capture parity inputs have incomplete layer mapping")
    independent = torch.stack(tuple(independent_aux), dim=1)
    if tuple(independent.shape) != tuple(auxiliary.shape):
        raise ValueError("independent block-hook tensors differ in shape from packed taps")
    if projected_logits.ndim != 2 or native_logits.ndim != 2:
        raise ValueError("capture parity logits must be rank two")
    projected = projected_logits[-1:].detach().cpu()
    native = native_logits[-1:].detach().cpu()
    if tuple(projected.shape) != tuple(native.shape):
        raise ValueError("projected and native logits have different global shapes")
    aux_abs, aux_rel = _parity_metrics(auxiliary.detach().cpu(), independent.detach().cpu())
    logits_abs, logits_rel = _parity_metrics(projected, native)
    aux_passed = torch.allclose(
        auxiliary.detach().cpu().float(),
        independent.detach().cpu().float(),
        atol=aux_atol,
        rtol=aux_rtol,
    )
    logits_passed = torch.allclose(
        projected.float(), native.float(), atol=logits_atol, rtol=logits_rtol
    )
    reasons = []
    if not aux_passed:
        reasons.append("aux tap mapping/order parity mismatch")
    if not logits_passed:
        reasons.append("final-hidden native logit parity mismatch")
    return CaptureAttestation(
        passed=aux_passed and logits_passed,
        token_count=int(final_hidden.shape[0]),
        logical_layer_ids=logical,
        physical_layer_ids=physical,
        independent_tap_paths=paths,
        native_logits_path=native_logits_path,
        aux_max_abs_error=aux_abs,
        aux_max_rel_error=aux_rel,
        logits_max_abs_error=logits_abs,
        logits_max_rel_error=logits_rel,
        reason="; ".join(reasons),
        aux_atol=aux_atol,
        aux_rtol=aux_rtol,
        logits_atol=logits_atol,
        logits_rtol=logits_rtol,
    )


def _server_value(server_args: Any, name: str, default: Any = None) -> Any:
    value = getattr(server_args, name, default)
    return value.value if hasattr(value, "value") else value


def validate_stage_b_server_args(server_args: Any) -> dict[str, Any]:
    """Fail before model allocation unless deterministic Ascend replay is configured."""

    if int(_server_value(server_args, "dp_size", 1)) != 1:
        raise ValueError("Stage B hidden replay requires DP=1")
    if int(_server_value(server_args, "pp_size", 1)) != 1:
        raise ValueError("Stage B hidden replay requires PP=1")
    if int(_server_value(server_args, "chunked_prefill_size", -1)) != -1:
        raise ValueError("Stage B hidden replay requires chunked prefill to be disabled")
    if not bool(_server_value(server_args, "disable_radix_cache", False)):
        raise ValueError("Stage B hidden replay requires radix/prefix cache to be disabled")
    if not bool(_server_value(server_args, "disable_cuda_graph", False)):
        raise ValueError("Stage B hidden replay requires execution graphs to be disabled")
    if int(_server_value(server_args, "max_running_requests", 1)) != 1:
        raise ValueError("Stage B hidden replay supports exactly one request")
    device = str(_server_value(server_args, "device", "")).lower()
    if device != "npu":
        raise ValueError("production Stage B requires device=npu")
    attention = str(_server_value(server_args, "attention_backend", "")).lower()
    if attention != "ascend":
        raise ValueError("production Stage B requires attention_backend=ascend")
    dtype = str(_server_value(server_args, "dtype", "")).lower()
    if dtype not in {"bfloat16", "bf16"}:
        raise ValueError("production Stage B requires BF16")
    if _server_value(server_args, "quantization", None) not in (None, ""):
        raise ValueError("production Stage B rejects quantization")
    model_runner = str(
        _server_value(
            server_args,
            "model_runner",
            _server_value(server_args, "model_impl", "torch"),
        )
    ).lower()
    if model_runner not in {"torch", "pytorch", "auto"}:
        raise ValueError("Stage B requires the SGLang PyTorch Model Runner")
    return {
        "model_runner": model_runner,
        "capture_mode": "FULL",
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
        "chunked_prefill_size": -1,
        "max_running_requests": 1,
        "device_type": device,
        "attention_backend": attention,
        "dtype": "bfloat16",
        "quantization": None,
        "tp_size": int(_server_value(server_args, "tp_size", 1)),
        "ep_size": int(_server_value(server_args, "ep_size", 1)),
        "pp_size": 1,
        "dp_size": 1,
        "nnodes": int(_server_value(server_args, "nnodes", 1)),
        "node_rank": int(_server_value(server_args, "node_rank", 0)),
    }


def _resolved_dtype(value: Any) -> str:
    normalized = str(value).lower().replace("torch.", "")
    return "bfloat16" if normalized in {"bfloat16", "bf16"} else normalized


def _resolved_device_type(value: Any) -> str:
    return str(value).lower().split(":", 1)[0]


def _resolved_attention_name(value: Any) -> str:
    if isinstance(value, str):
        return value.lower()
    name = type(value).__name__.lower()
    return "ascend" if "ascend" in name else name


def validate_resolved_stage_b_runner(
    runner: Any, *, runner_selection_path: str
) -> dict[str, Any]:
    """Validate post-load runtime objects, not the requested CLI values."""

    supported_paths = {
        "load_model_result.torch_runner",
        "load_model_result.direct_sglang_model_runner",
    }
    if runner_selection_path not in supported_paths:
        raise ValueError("resolved runner is not the SGLang PyTorch runner")
    runner_type = type(runner)
    try:
        runner_module = importlib.import_module(
            "sglang.srt.model_executor.model_runner"
        )
        imported_runner_type = getattr(runner_module, "ModelRunner")
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            "could not resolve the imported SGLang ModelRunner class"
        ) from exc
    if (
        runner_type is not imported_runner_type
        or runner_type.__module__ != "sglang.srt.model_executor.model_runner"
        or runner_type.__name__ != "ModelRunner"
    ):
        raise ValueError(
            "resolved object is not the imported actual SGLang PyTorch runner "
            "(ModelRunner): "
            f"{runner_type.__module__}.{runner_type.__name__}"
        )
    resolved_args = getattr(runner, "server_args", None)
    if resolved_args is None:
        raise ValueError("resolved PyTorch runner exposes no resolved ServerArgs")
    contract = validate_stage_b_server_args(resolved_args)
    device = _resolved_device_type(getattr(runner, "device", ""))
    if device != "npu":
        raise ValueError(f"resolved device must be NPU, got {device!r}")
    attention_object = getattr(
        runner, "attention_backend", getattr(runner, "attn_backend", None)
    )
    attention_type = type(attention_object)
    attention = _resolved_attention_name(attention_object)
    if (
        attention != "ascend"
        or not attention_type.__module__.startswith("sglang.srt.")
        or "ascend" not in attention_type.__name__.lower()
    ):
        raise ValueError(
            f"resolved attention backend must be Ascend, got {attention!r}"
        )
    try:
        backend_module = importlib.import_module(attention_type.__module__)
        imported_attention_type = getattr(backend_module, attention_type.__name__)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            "resolved attention backend is not an imported SGLang Ascend class"
        ) from exc
    if attention_type is not imported_attention_type:
        raise ValueError(
            f"resolved attention backend must be Ascend, got {attention!r}"
        )
    model_config = getattr(runner, "model_config", None)
    dtype = _resolved_dtype(getattr(model_config, "dtype", None))
    if dtype != "bfloat16":
        raise ValueError(f"resolved dtype must be BF16, got {dtype!r}")
    return {
        **contract,
        "runner_implementation": "pytorch",
        "runner_selection_path": runner_selection_path,
        "runner_class": type(runner).__name__,
        "runner_module": type(runner).__module__,
        "model_class": type(runner.model).__name__,
        "attention_backend_module": attention_type.__module__,
        "attention_backend_class": attention_type.__name__,
        "device_type": device,
        "attention_backend": attention,
        "dtype": dtype,
    }


def _probe_ascend_a2_runtime(
    runner: Any,
    *,
    resolved_runtime: Mapping[str, Any],
    gpu_id: int,
) -> tuple[Any | None, dict[str, Any]]:
    """Issue production evidence only from the live torch-npu device probe."""

    try:
        importlib.import_module("torch_npu")
        npu = getattr(torch, "npu", None)
        if npu is None or not hasattr(npu, "get_device_name"):
            raise RuntimeError("torch_npu did not register torch.npu device APIs")
        device_text = str(getattr(runner, "device", f"npu:{gpu_id}"))
        device_index = (
            int(device_text.split(":", 1)[1])
            if ":" in device_text
            else int(gpu_id)
        )
        device_name = str(npu.get_device_name(device_index))
        device_count = int(npu.device_count())
        cann = _cann_identity()
        if cann.get("cann_error"):
            raise RuntimeError(
                "CANN runtime version attestation failed: " + cann["cann_error"]
            )
        evidence = {
            "schema": ASCEND_A2_ATTESTATION_SCHEMA,
            "passed": True,
            "device_name": device_name,
            "device_index": device_index,
            "device_count": device_count,
            "runner_module": str(resolved_runtime["runner_module"]),
            "runner_class": str(resolved_runtime["runner_class"]),
            "attention_backend_module": str(
                resolved_runtime["attention_backend_module"]
            ),
            "attention_backend_class": str(
                resolved_runtime["attention_backend_class"]
            ),
            "sglang_version": _distribution_version("sglang"),
            "torch_npu_version": _distribution_version("torch-npu"),
            "cann_version": str(cann["cann_version"]),
        }
        validate_ascend_a2_evidence(evidence)
        validate_live_ascend_a2_evidence(evidence)
        return evidence, evidence
    except Exception as exc:
        return None, {
            "schema": ASCEND_A2_ATTESTATION_SCHEMA,
            "passed": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _cann_identity() -> dict[str, str]:
    try:
        return {
            "cann_version_source": "torch_npu.npu.get_cann_version(CANN)",
            "cann_version": _live_cann_version(),
        }
    except Exception as exc:
        return {
            "cann_version_source": "torch_npu.npu.get_cann_version(CANN)",
            "cann_version": "unavailable",
            "cann_error": f"{type(exc).__name__}: {exc}",
        }


def req_token_array(input_ids: Sequence[int]) -> array:
    return array("q", validate_token_ids(input_ids))


def initialize_forward_batch(
    schedule_batch: Any, runner: Any, forward_batch_class: Any
) -> Any:
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
    errors: list[Exception] = []
    for name in ("sglang.benchmark.one_batch", "sglang.bench_one_batch"):
        try:
            return importlib.import_module(name)
        except (ImportError, ModuleNotFoundError) as exc:
            errors.append(exc)
    raise ImportError(
        "SGLang standalone one-batch runner is unavailable; tried both layouts"
    ) from errors[-1]


def _hidden_size(model_config: Any) -> int:
    direct = getattr(model_config, "hidden_size", None)
    if direct is not None:
        return int(direct)
    text_config = getattr(model_config, "text_config", None)
    if text_config is not None:
        nested = (
            text_config.get("hidden_size")
            if isinstance(text_config, dict)
            else getattr(text_config, "hidden_size", None)
        )
        if nested is not None:
            return int(nested)
    raise RuntimeError("SGLang model config does not expose text hidden_size")


def _vocab_size(model_config: Any) -> int:
    direct = getattr(model_config, "vocab_size", None)
    if direct is not None:
        return int(direct)
    text_config = getattr(model_config, "text_config", None)
    if text_config is not None:
        nested = (
            text_config.get("vocab_size")
            if isinstance(text_config, dict)
            else getattr(text_config, "vocab_size", None)
        )
        if nested is not None:
            return int(nested)
    raise RuntimeError("SGLang model config does not expose text vocab_size")


def _text_backbone(model: Any) -> tuple[Any, str]:
    wrapped_model = getattr(model, "model", None)
    candidates = (
        (
            getattr(wrapped_model, "language_model", None),
            "model.model.language_model",
        ),
        (getattr(model, "language_model", None), "model.language_model"),
        (wrapped_model, "model.model"),
        (model, "model"),
    )
    for candidate, path in candidates:
        if candidate is not None and getattr(candidate, "norm", None) is not None:
            return candidate, path
    raise RuntimeError("GLM-5.3 wrapper path does not expose the text backbone final norm")


class SGLangInternalHiddenRunner:
    """Lazy adapter over SGLang's internal device-aware PyTorch ModelRunner."""

    def __init__(
        self,
        *,
        server_args: Any,
        port_args: Any,
        gpu_id: int,
        tp_rank: int,
        logical_layer_ids: Sequence[int],
    ) -> None:
        requested_contract = validate_stage_b_server_args(server_args)
        one_batch = one_batch_module()
        wrapped, self.tokenizer = one_batch.load_model(
            server_args, port_args, gpu_id, tp_rank
        )
        self._wrapped = wrapped
        self._runner = getattr(wrapped, "torch_runner", None)
        runner_selection_path = "load_model_result.torch_runner"
        if self._runner is None:
            runner_type = type(wrapped)
            if (
                runner_type.__module__ == "sglang.srt.model_executor.model_runner"
                and runner_type.__name__ == "ModelRunner"
            ):
                self._runner = wrapped
                runner_selection_path = (
                    "load_model_result.direct_sglang_model_runner"
                )
            else:
                raise ValueError(
                    "resolved runner is not the SGLang PyTorch runner: "
                    "load_model result has neither torch_runner nor the supported "
                    "sglang.srt.model_executor.model_runner.ModelRunner identity"
                )
        if not hasattr(self._runner, "model"):
            raise RuntimeError("selected SGLang runner is not the PyTorch runner")
        resolved_runtime = validate_resolved_stage_b_runner(
            self._runner, runner_selection_path=runner_selection_path
        )
        (
            self.ascend_a2_attestation,
            ascend_a2_runtime,
        ) = _probe_ascend_a2_runtime(
            self._runner,
            resolved_runtime=resolved_runtime,
            gpu_id=gpu_id,
        )
        self.tp_rank = int(tp_rank)
        self.logical_layer_ids = tuple(int(value) for value in logical_layer_ids)
        if self.logical_layer_ids != GLM53_DFLASH_LOGICAL_LAYERS:
            raise ValueError(
                "GLM-5.3 extraction requires ordered logical layers "
                f"{GLM53_DFLASH_LOGICAL_LAYERS}, got {self.logical_layer_ids}"
            )
        model = self._runner.model
        capture_hook = None
        for name in ("set_dflash_layers_to_capture", "set_eagle3_layers_to_capture"):
            candidate = getattr(model, name, None)
            if callable(candidate):
                candidate(list(self.logical_layer_ids))
                capture_hook = name
                break
        if capture_hook is None:
            raise RuntimeError(
                f"{type(model).__name__} exposes no supported hidden capture hook"
            )
        self.physical_layer_ids = tuple(layer + 1 for layer in self.logical_layer_ids)
        backbone, backbone_path = _text_backbone(model)
        reported = getattr(backbone, "layers_to_capture", None)
        if reported is None or tuple(int(value) for value in reported) != (
            self.physical_layer_ids
        ):
            raise RuntimeError(
                "SGLang did not prove the expected physical capture taps: "
                f"{reported!r} != {self.physical_layer_ids}"
            )
        self.hidden_size = _hidden_size(self._runner.model_config)
        self.capture_mapping = tuple(
            CaptureTap(
                "sglang",
                logical,
                f"{backbone_path}.layers_to_capture[{physical}]",
                "post_decoder_block",
            )
            for logical, physical in zip(
                self.logical_layer_ids, self.physical_layer_ids, strict=True
            )
        )
        layers = getattr(backbone, "layers", None)
        if layers is None or not hasattr(layers, "__len__"):
            raise RuntimeError(
                "resolved GLM-5.3 text backbone exposes no concrete decoder layers; "
                "cannot attest auxiliary tap semantics"
            )
        if len(layers) <= max(self.logical_layer_ids):
            raise RuntimeError(
                "resolved GLM-5.3 decoder layer collection is too short for "
                f"{self.logical_layer_ids}"
            )
        self.independent_tap_paths = tuple(
            f"{backbone_path}.layers[{logical}].forward_output"
            for logical in self.logical_layer_ids
        )
        self._independent_aux: dict[int, torch.Tensor] = {}
        self._independent_hooks = []
        for logical in self.logical_layer_ids:
            module = layers[logical]
            if not hasattr(module, "register_forward_hook"):
                raise RuntimeError(
                    f"decoder layer {logical} cannot install an independent parity hook"
                )

            def capture_block(
                unused_module: Any,
                unused_inputs: Any,
                output: Any,
                *,
                layer_id: int = logical,
            ) -> None:
                self._capture_independent_block(layer_id, output)

            self._independent_hooks.append(module.register_forward_hook(capture_block))
        final_norm = backbone.norm
        if not hasattr(final_norm, "register_forward_hook"):
            raise RuntimeError("GLM-5.3 final norm cannot install a forward hook")
        self.final_hidden_tap = f"{backbone_path}.norm.forward_output"
        self._post_norm_hidden: torch.Tensor | None = None
        self._final_hidden_hook = final_norm.register_forward_hook(
            self._capture_post_norm
        )
        try:
            sglang = importlib.import_module("sglang")
            version = str(getattr(sglang, "__version__", "unknown"))
        except Exception:
            version = "unknown"
        self.backend_metadata = {
            "backend": "sglang_internal_model_runner",
            "sglang_version": version,
            "torch_version": str(torch.__version__),
            "torch_npu_version": _distribution_version("torch-npu"),
            **_cann_identity(),
            "model_class": type(model).__name__,
            "runner_class": type(self._runner).__name__,
            **resolved_runtime,
            "requested_runtime": requested_contract,
            "resolved_runtime": resolved_runtime,
            "ascend_a2_runtime": ascend_a2_runtime,
            "capture_hook": capture_hook,
            "logical_layer_ids": list(self.logical_layer_ids),
            "physical_layer_ids": list(self.physical_layer_ids),
            "capture_mapping": [tap.as_tuple() for tap in self.capture_mapping],
            "final_hidden_tap": self.final_hidden_tap,
            "final_hidden_semantics": "post_final_norm_lm_head_input",
            "independent_tap_paths": list(self.independent_tap_paths),
        }

    def _capture_independent_block(self, logical_layer_id: int, output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise RuntimeError(
                f"decoder layer {logical_layer_id} returned an unsupported parity tensor"
            )
        self._independent_aux[int(logical_layer_id)] = value

    def _capture_post_norm(self, *args: Any) -> None:
        output = args[-1]
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise RuntimeError("GLM-5.3 final norm returned an unsupported tensor")
        self._post_norm_hidden = value

    def _project_global_logits(
        self,
        final_hidden: torch.Tensor,
        *,
        forward_batch: Any | None = None,
        require_output: bool = True,
    ) -> torch.Tensor | None:
        """Use only a supported full-vocabulary logits API; never a raw TP shard."""

        api_path = (
            f"{type(self._runner).__module__}.{type(self._runner).__name__}"
            ".compute_logits/all_tp_ranks"
        )
        method = getattr(self._runner, "compute_logits", None)
        if not callable(method):
            raise RuntimeError(
                "locked SGLang runtime exposes no globally comparable native logits "
                f"projection API; {api_path}: unavailable"
            )
        expected_vocab = _vocab_size(self._runner.model_config)
        parameters = inspect.signature(method).parameters.values()
        required = [
            parameter
            for parameter in parameters
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        call_args: tuple[Any, ...] = (final_hidden,)
        if len(required) == 2 and required[1].name in {
            "forward_batch",
            "batch",
        } and forward_batch is not None:
            call_args = (final_hidden, forward_batch)
        elif len(required) > 1:
            raise RuntimeError(
                f"{api_path} requires unsupported runtime arguments"
            )
        try:
            result = method(*call_args)
        except Exception as exc:
            raise RuntimeError(
                f"{api_path} failed: {type(exc).__name__}: {exc}"
            ) from exc
        logits = getattr(result, "next_token_logits", result)
        self._native_logits_path = api_path
        if not require_output:
            if (
                isinstance(logits, torch.Tensor)
                and logits.ndim == 2
                and logits.shape == (final_hidden.shape[0], expected_vocab)
            ):
                return logits
            return None
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise RuntimeError(f"{api_path} did not return rank-two logits")
        if logits.shape[0] != final_hidden.shape[0]:
            raise RuntimeError(f"{api_path} token dimension differs")
        if logits.shape[1] != expected_vocab:
            raise RuntimeError(
                f"{api_path} width {logits.shape[1]} is not global vocab "
                f"{expected_vocab}"
            )
        return logits

    def _normalize_capture(
        self,
        auxiliary: torch.Tensor,
        *,
        token_count: int,
        native_logits: torch.Tensor | None = None,
        forward_batch: Any | None = None,
        projected_logits: torch.Tensor | None = None,
    ) -> TargetHiddenCapture:
        if auxiliary.ndim == 2:
            expected = (token_count, len(self.logical_layer_ids) * self.hidden_size)
            if tuple(auxiliary.shape) != expected:
                raise RuntimeError(
                    f"captured auxiliary shape {tuple(auxiliary.shape)} != {expected}"
                )
            auxiliary = auxiliary.reshape(
                token_count, len(self.logical_layer_ids), self.hidden_size
            )
        expected_3d = (
            token_count,
            len(self.logical_layer_ids),
            self.hidden_size,
        )
        if tuple(auxiliary.shape) != expected_3d:
            raise RuntimeError(
                f"captured auxiliary shape {tuple(auxiliary.shape)} != {expected_3d}"
            )
        if self._post_norm_hidden is None:
            raise RuntimeError("SGLang returned no post-final-norm hidden capture")
        if tuple(self._post_norm_hidden.shape) != (token_count, self.hidden_size):
            raise RuntimeError("captured post-final-norm hidden shape is invalid")
        missing = [
            logical
            for logical in self.logical_layer_ids
            if logical not in self._independent_aux
        ]
        if missing:
            raise RuntimeError(
                "independent decoder block hooks did not fire for logical layers "
                f"{missing}; cannot attest auxiliary tap semantics"
            )
        if native_logits is None:
            raise RuntimeError(
                "SGLang returned no globally comparable native logits for parity"
            )
        if projected_logits is None:
            projected_logits = self._project_global_logits(
                self._post_norm_hidden[-1:], forward_batch=forward_batch
            )
        if projected_logits is None:
            raise RuntimeError("rank zero received no global projected logits")
        attestation = attest_capture_semantics(
            auxiliary=auxiliary,
            independent_aux=tuple(
                self._independent_aux[logical] for logical in self.logical_layer_ids
            ),
            final_hidden=self._post_norm_hidden,
            projected_logits=projected_logits,
            native_logits=native_logits[-1:],
            layer_mapping=tuple(
                zip(
                    self.logical_layer_ids,
                    self.physical_layer_ids,
                    strict=True,
                )
            ),
            native_logits_path=self._native_logits_path,
            independent_tap_paths=self.independent_tap_paths,
        )
        capture = TargetHiddenCapture(
            aux_hidden_states=auxiliary,
            target_final_hidden=self._post_norm_hidden,
            capture_mapping=self.capture_mapping,
            attestation=attestation,
        ).cpu_bfloat16()
        self._post_norm_hidden = None
        self._independent_aux.clear()
        return capture

    @torch.no_grad()
    def extract(self, input_ids: Sequence[int]) -> TargetHiddenCapture:
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        if not input_ids:
            raise ValueError("cannot extract an empty sequence")
        one_batch = one_batch_module()
        self._clear_pools()
        self._post_norm_hidden = None
        self._independent_aux.clear()
        req = Req(
            rid=0,
            origin_input_text="",
            origin_input_ids=req_token_array(input_ids),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.logprob_start_len = -1
        if hasattr(req, "set_extend_range"):
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.set_extend_range(0, len(req.origin_input_ids))
        else:
            req.fill_ids = req.origin_input_ids
            req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        tree_cache = one_batch.TreeCacheNamespace(
            page_size=int(getattr(self._runner.server_args, "page_size", 1)),
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
        batch.return_hidden_states = True
        batch.prepare_for_extend()
        one_batch._maybe_prepare_mlp_sync_batch(batch, self._runner)
        if (
            batch.input_ids is None
            and getattr(batch, "prefill_input_ids_cpu", None) is not None
        ):
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        forward_batch = initialize_forward_batch(batch, self._runner, ForwardBatch)
        logits_output = self._runner.forward(forward_batch).logits_output
        output = logits_output.hidden_states
        if output is None:
            raise RuntimeError("SGLang returned no captured auxiliary hidden states")
        native_logits = getattr(logits_output, "next_token_logits", None)
        if self._post_norm_hidden is None:
            raise RuntimeError("SGLang returned no post-final-norm hidden capture")
        projected_logits = self._project_global_logits(
            self._post_norm_hidden[-1:],
            forward_batch=forward_batch,
            require_output=self.tp_rank == 0,
        )
        if self.tp_rank == 0:
            capture = self._normalize_capture(
                output,
                token_count=len(input_ids),
                native_logits=native_logits,
                forward_batch=forward_batch,
                projected_logits=projected_logits,
            )
        else:
            self._post_norm_hidden = None
            self._independent_aux.clear()
            capture = None
        self._clear_pools()
        if self.tp_rank == 0:
            if capture is None:
                raise RuntimeError("rank zero capture was not produced")
            return capture
        return TargetHiddenCapture(
            aux_hidden_states=torch.empty(
                0,
                len(self.logical_layer_ids),
                self.hidden_size,
                dtype=torch.bfloat16,
            ),
            target_final_hidden=torch.empty(
                0, self.hidden_size, dtype=torch.bfloat16
            ),
            capture_mapping=self.capture_mapping,
        )

    def _clear_pools(self) -> None:
        if hasattr(self._wrapped, "clear"):
            self._wrapped.clear()
            return
        self._runner.req_to_token_pool.clear()
        self._runner.token_to_kv_pool_allocator.clear()
