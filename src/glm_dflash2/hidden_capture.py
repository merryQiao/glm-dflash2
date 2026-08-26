from __future__ import annotations

from dataclasses import dataclass

import torch


FINAL_HIDDEN_SEMANTICS = "post_final_norm_lm_head_input"


@dataclass(frozen=True)
class CaptureTap:
    backend_namespace: str
    logical_layer_id: int
    concrete_tap: str
    tap_semantics: str

    def __post_init__(self) -> None:
        if not self.backend_namespace or not self.concrete_tap or not self.tap_semantics:
            raise ValueError("capture tap metadata cannot be empty")
        if self.logical_layer_id < 0:
            raise ValueError("logical layer ID must be non-negative")

    def as_tuple(self) -> tuple[str, int, str, str]:
        return (
            self.backend_namespace,
            self.logical_layer_id,
            self.concrete_tap,
            self.tap_semantics,
        )


@dataclass(frozen=True)
class TargetHiddenCapture:
    aux_hidden_states: torch.Tensor
    target_final_hidden: torch.Tensor
    capture_mapping: tuple[CaptureTap, ...]
    final_hidden_semantics: str = FINAL_HIDDEN_SEMANTICS

    def __post_init__(self) -> None:
        if self.aux_hidden_states.ndim != 3:
            raise ValueError("aux_hidden_states must have shape [tokens, layers, hidden]")
        if self.target_final_hidden.ndim != 2:
            raise ValueError("target_final_hidden must have shape [tokens, hidden]")
        if self.aux_hidden_states.shape[0] != self.target_final_hidden.shape[0]:
            raise ValueError("auxiliary and final token dimensions differ")
        if self.aux_hidden_states.shape[2] != self.target_final_hidden.shape[1]:
            raise ValueError("auxiliary and final hidden widths differ")
        if len(self.capture_mapping) != self.aux_hidden_states.shape[1]:
            raise ValueError("capture mapping does not match auxiliary layer dimension")
        logical = tuple(tap.logical_layer_id for tap in self.capture_mapping)
        if logical != tuple(sorted(logical)) or len(logical) != len(set(logical)):
            raise ValueError("capture mapping logical layers must be unique and ordered")
        if self.final_hidden_semantics != FINAL_HIDDEN_SEMANTICS:
            raise ValueError("target final hidden is not the post-final-norm LM-head input")
        if not bool(torch.isfinite(self.aux_hidden_states).all()):
            raise ValueError("auxiliary hidden states contain NaN or Inf")
        if not bool(torch.isfinite(self.target_final_hidden).all()):
            raise ValueError("target final hidden contains NaN or Inf")

    @property
    def logical_layer_ids(self) -> tuple[int, ...]:
        return tuple(tap.logical_layer_id for tap in self.capture_mapping)

    def cpu_bfloat16(self) -> "TargetHiddenCapture":
        return TargetHiddenCapture(
            aux_hidden_states=self.aux_hidden_states.detach().to(
                device="cpu", dtype=torch.bfloat16
            ),
            target_final_hidden=self.target_final_hidden.detach().to(
                device="cpu", dtype=torch.bfloat16
            ),
            capture_mapping=self.capture_mapping,
            final_hidden_semantics=self.final_hidden_semantics,
        )
