from __future__ import annotations

import pytest

from glm53_w8.distributed import configure_accumulation


class _FSDP2Probe:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, bool]] = []

    def set_requires_gradient_sync(self, value: bool, *, recurse: bool = True) -> None:
        self.calls.append(("sync", bool(value), recurse))

    def set_reshard_after_backward(self, value: bool, *, recurse: bool = True) -> None:
        self.calls.append(("backward", bool(value), recurse))


def test_configure_accumulation_uses_standard_fsdp2_backward_reshard_api() -> None:
    model = _FSDP2Probe()
    configure_accumulation(model, synchronize=False)
    configure_accumulation(model, synchronize=True)
    assert model.calls == [
        ("sync", False, True),
        ("backward", False, True),
        ("sync", True, True),
        ("backward", True, True),
    ]


def test_configure_accumulation_rejects_incomplete_fsdp2_api() -> None:
    class _Incomplete:
        def set_requires_gradient_sync(self, value: bool, *, recurse: bool = True) -> None:
            del value, recurse

    with pytest.raises(TypeError, match="incomplete FSDP2 accumulation API"):
        configure_accumulation(_Incomplete(), synchronize=True)
