"""Content-addressed artifact records used by resumable jobs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path, *, relative_to: str | Path) -> dict[str, Any]:
    target = Path(path)
    root = Path(relative_to)
    return {
        "path": str(target.relative_to(root)),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def verify_artifact_record(record: Mapping[str, Any], *, root: str | Path) -> Path:
    path = Path(root) / str(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"artifact checksum/size mismatch: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"artifact checksum mismatch: {path}")
    return path
