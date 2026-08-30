from __future__ import annotations

import os
import socket
from pathlib import Path


def repair_truncated_jsonl(path: Path, *, chunk_size: int = 1024 * 1024) -> int:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("rb+") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return 0
        cursor = size
        keep = 0
        while cursor > 0:
            start = max(0, cursor - chunk_size)
            handle.seek(start)
            chunk = handle.read(cursor - start)
            last_newline = chunk.rfind(b"\n")
            if last_newline >= 0:
                keep = start + last_newline + 1
                break
            cursor = start
        handle.truncate(keep)
        handle.flush()
        os.fsync(handle.fileno())
        return size - keep


class OutputShardLock:
    """Fail fast when two generators target the same response shard."""

    def __init__(self, output_path: Path):
        self.path = output_path.with_suffix(output_path.suffix + ".lock")
        self._handle = None

    def __enter__(self) -> "OutputShardLock":
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"output shard is already locked: {self.path}") from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is None:
            return
        import fcntl

        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
