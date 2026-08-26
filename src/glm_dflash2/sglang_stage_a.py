from __future__ import annotations

import json
import os
import fcntl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .jsonl import repair_truncated_jsonl


@dataclass(frozen=True)
class SGLangServerConfig:
    python: str
    model_path: Path
    served_model_name: str = "GLM-5.2"
    host: str = "127.0.0.1"
    port: int = 30000
    tp_size: int = 16
    dtype: str = "bfloat16"
    device: str = "npu"
    attention_backend: str = "ascend"
    context_length: int = 131072
    mem_fraction_static: float = 0.9
    max_running_requests: int = 1
    max_total_tokens: int | None = 131072
    reasoning_parser: str = "glm45"
    tool_call_parser: str = "glm47"
    quantization: str | None = None
    moe_a2a_backend: str | None = None
    deepep_mode: str | None = None
    extra_args: tuple[str, ...] = ()


def build_server_command(config: SGLangServerConfig) -> list[str]:
    """Build the GLM-5.2 SGLang command without shell interpolation."""

    command = [
        config.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(config.model_path),
        "--served-model-name",
        config.served_model_name,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tp-size",
        str(config.tp_size),
        "--dtype",
        config.dtype,
        "--device",
        config.device,
        "--attention-backend",
        config.attention_backend,
        "--trust-remote-code",
        "--reasoning-parser",
        config.reasoning_parser,
        "--tool-call-parser",
        config.tool_call_parser,
        "--context-length",
        str(config.context_length),
        "--mem-fraction-static",
        str(config.mem_fraction_static),
        "--max-running-requests",
        str(config.max_running_requests),
        "--log-level-http",
        "warning",
    ]
    if config.max_total_tokens is not None:
        command.extend(("--max-total-tokens", str(config.max_total_tokens)))
    if config.quantization:
        command.extend(("--quantization", config.quantization))
    if config.moe_a2a_backend:
        command.extend(("--moe-a2a-backend", config.moe_a2a_backend))
    if config.deepep_mode:
        command.extend(("--deepep-mode", config.deepep_mode))
    command.extend(config.extra_args)
    return command


def owns_source_index(source_index: int, *, shard_index: int, shard_count: int) -> bool:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if source_index < 0:
        raise ValueError("source_index must be non-negative")
    return source_index % shard_count == shard_index


class CommittedJsonlWriter:
    """Append complete JSONL records; flush+fsync is the commit point."""

    def __init__(
        self,
        path: Path,
        *,
        truncate: bool = False,
        repair_tail: bool = True,
        lock: bool = True,
    ) -> None:
        self.path = Path(path)
        self.truncate = truncate
        self.repair_tail = repair_tail
        self.lock = lock
        self._handle = None
        self._lock = None

    def __enter__(self) -> "CommittedJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock:
            self._lock = self.path.with_suffix(self.path.suffix + ".lock").open("a+b")
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self._lock.close()
                self._lock = None
                raise RuntimeError(f"another writer owns {self.path}") from exc
        if self.repair_tail and not self.truncate:
            repair_truncated_jsonl(self.path)
        self._handle = self.path.open("w" if self.truncate else "a", encoding="utf-8")
        return self

    def append(self, record: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("writer is not open")
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
        self._handle.write(line + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None


class AttemptErrorLedger:
    """Append-only per-sample failures with explicit later resolution."""

    def __init__(self, path: Path, *, truncate: bool = False) -> None:
        self.path = Path(path)
        self.truncate = bool(truncate)
        self._handle = None
        self._unresolved: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> "AttemptErrorLedger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.truncate:
            self.path.write_text("", encoding="utf-8")
        repair_truncated_jsonl(self.path)
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sample_id = str(row["id"])
                    if row.get("status") == "error":
                        self._unresolved[sample_id] = row
                    elif row.get("status") == "resolved":
                        self._unresolved.pop(sample_id, None)
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    @property
    def unresolved_ids(self) -> frozenset[str]:
        return frozenset(self._unresolved)

    def _append(self, row: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("error ledger is not open")
        self._handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def record_error(self, *, sample_id: str, source_index: int, error: BaseException) -> None:
        row = {
            "id": str(sample_id),
            "source_index": int(source_index),
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "time_unix": time.time(),
        }
        self._append(row)
        self._unresolved[str(sample_id)] = row

    def resolve(self, *, sample_id: str, source_index: int) -> None:
        sample_id = str(sample_id)
        if sample_id not in self._unresolved:
            return
        self._append(
            {
                "id": sample_id,
                "source_index": int(source_index),
                "status": "resolved",
                "time_unix": time.time(),
            }
        )
        self._unresolved.pop(sample_id, None)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
