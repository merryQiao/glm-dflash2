"""Small atomic I/O helpers shared by Thinker data preparation scripts."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


class ParquetBatchWriter:
    """Write compressed Parquet batches and expose the result atomically."""

    def __init__(
        self, path: str | Path, schema: pa.Schema, batch_size: int = 4096
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        self.schema = schema
        self.batch_size = batch_size
        self.rows: list[dict[str, Any]] = []
        self.writer = pq.ParquetWriter(
            self.temporary,
            schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        self.count = 0
        self.closed = False

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def extend(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
        self.count += len(self.rows)
        self.rows.clear()

    def close(self, commit: bool = True) -> None:
        if self.closed:
            return
        try:
            if commit:
                self.flush()
            self.writer.close()
            if commit:
                os.replace(self.temporary, self.path)
            else:
                self.temporary.unlink(missing_ok=True)
        finally:
            self.closed = True

    def __enter__(self) -> "ParquetBatchWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(commit=exc_type is None)


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write a JSON document without exposing a partially written file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
