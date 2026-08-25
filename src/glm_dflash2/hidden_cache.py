from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .jsonl import repair_truncated_jsonl


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_compact_json(value) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class HiddenCacheSpec:
    layer_ids: tuple[int, ...]
    hidden_size: int
    dtype: str = "bfloat16"
    input_dtype: str = "int64"
    mask_semantics: str = "dflash_target_token"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.layer_ids or any(layer < 0 for layer in self.layer_ids):
            raise ValueError("layer_ids must contain non-negative logical layer IDs")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("layer_ids must be unique")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.dtype != "bfloat16" or self.input_dtype != "int64":
            raise ValueError("the packed format requires BF16 hidden and int64 IDs")


class PackedHiddenWriter:
    """Crash-recoverable, single-writer packed hidden-state cache."""

    def __init__(
        self,
        root: str | Path,
        *,
        spec: HiddenCacheSpec,
        max_segment_bytes: int = 64 << 30,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.spec = spec
        self.max_segment_bytes = int(max_segment_bytes)
        self.provenance = dict(provenance or {})
        if self.max_segment_bytes < 1:
            raise ValueError("max_segment_bytes must be positive")
        self._lock = self._index = None
        self._files: dict[str, Any] = {}
        self._segment = 0
        self._segment_bytes = 0
        self._manifest: dict[str, Any] = {}
        self._sample_ids: set[str] = set()

    def __enter__(self) -> "PackedHiddenWriter":
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / ".writer.lock").open("a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another writer owns {self.root}") from exc
        index_path = self.root / "index.jsonl"
        repair_truncated_jsonl(index_path)
        existing = rebuild_manifest_from_index(self.root, spec=self.spec, write=False)
        existing_rows = _read_index(self.root)
        self._sample_ids = {str(row["sample_id"]) for row in existing_rows}
        if len(self._sample_ids) != len(existing_rows):
            raise ValueError("hidden cache index contains duplicate sample_id values")
        _recover_unindexed_stream_tails(self.root, existing_rows)
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_spec = previous.get("spec")
            if previous_spec:
                normalized_spec = dict(previous_spec)
                normalized_spec["layer_ids"] = tuple(normalized_spec["layer_ids"])
                if HiddenCacheSpec(**normalized_spec) != self.spec:
                    raise ValueError("hidden cache spec differs from existing manifest")
            previous_provenance = previous.get("provenance")
            if previous_provenance and previous_provenance != self.provenance:
                raise ValueError("hidden cache provenance differs from existing manifest")
            if previous.get("status") == "frozen":
                raise ValueError("cannot append to a frozen hidden cache")
        self._manifest = {
            "schema_version": 1,
            "status": "building",
            "spec": asdict(self.spec),
            "samples": existing["samples"],
            "total_tokens": existing["total_tokens"],
            "segments": existing["segments"],
            "provenance": self.provenance,
        }
        _atomic_json(manifest_path, self._manifest)
        self._index = index_path.open("a", encoding="utf-8")
        if existing["segments"]:
            self._segment = max(int(value) for value in existing["segments"])
            segment_dir = self.root / f"segment-{self._segment:05d}"
            self._segment_bytes = sum(
                (segment_dir / name).stat().st_size if (segment_dir / name).exists() else 0
                for name in ("input_ids.bin", "loss_mask.bin", "hidden_states.bin")
            )
        self._open_segment(self._segment)
        return self

    def _open_segment(self, segment: int) -> None:
        for handle in self._files.values():
            handle.close()
        segment_dir = self.root / f"segment-{segment:05d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        self._files = {
            "input_ids": (segment_dir / "input_ids.bin").open("ab"),
            "loss_mask": (segment_dir / "loss_mask.bin").open("ab"),
            "hidden_states": (segment_dir / "hidden_states.bin").open("ab"),
        }
        _fsync_dir(segment_dir)
        _fsync_dir(self.root)
        self._segment = segment
        if str(segment) not in self._manifest["segments"]:
            self._manifest["segments"].append(str(segment))

    def append(
        self,
        *,
        sample_id: str,
        source_index: int,
        input_ids: torch.Tensor | Sequence[int],
        loss_mask: torch.Tensor | Sequence[int | bool],
        hidden_states: torch.Tensor,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._index is None:
            raise RuntimeError("writer is not open")
        ids = torch.as_tensor(input_ids, dtype=torch.int64).reshape(-1).cpu()
        mask = torch.as_tensor(loss_mask, dtype=torch.bool).reshape(-1).cpu()
        hidden = torch.as_tensor(hidden_states).detach().cpu()
        expected = (ids.numel(), len(self.spec.layer_ids), self.spec.hidden_size)
        if tuple(hidden.shape) != expected:
            raise ValueError(f"hidden_states shape {tuple(hidden.shape)} != {expected}")
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError("hidden_states contain NaN or Inf")
        if mask.numel() != ids.numel():
            raise ValueError("loss_mask length differs from input_ids")
        if not sample_id:
            raise ValueError("sample_id is empty")
        if sample_id in self._sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")

        ids_raw = ids.numpy().astype("<i8", copy=False).tobytes(order="C")
        mask_raw = mask.numpy().astype("u1", copy=False).tobytes(order="C")
        hidden_raw = (
            hidden.to(torch.bfloat16)
            .contiguous()
            .view(torch.uint16)
            .numpy()
            .astype("<u2", copy=False)
            .tobytes(order="C")
        )
        sample_bytes = len(ids_raw) + len(mask_raw) + len(hidden_raw)
        if self._segment_bytes and self._segment_bytes + sample_bytes > self.max_segment_bytes:
            self._open_segment(self._segment + 1)
            self._segment_bytes = 0

        offsets = {name: handle.tell() for name, handle in self._files.items()}
        for name, raw in (
            ("input_ids", ids_raw),
            ("loss_mask", mask_raw),
            ("hidden_states", hidden_raw),
        ):
            handle = self._files[name]
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        record = {
            "sample_id": sample_id,
            "source_index": int(source_index),
            "segment": self._segment,
            "tokens": int(ids.numel()),
            "offsets": offsets,
            "nbytes": {
                "input_ids": len(ids_raw),
                "loss_mask": len(mask_raw),
                "hidden_states": len(hidden_raw),
            },
            "sha256": {
                "input_ids": _sha256(ids_raw),
                "loss_mask": _sha256(mask_raw),
                "hidden_states": _sha256(hidden_raw),
            },
            "metadata": dict(metadata or {}),
        }
        self._index.write(_compact_json(record) + "\n")
        self._index.flush()
        os.fsync(self._index.fileno())
        self._segment_bytes += sample_bytes
        self._manifest["samples"] += 1
        self._manifest["total_tokens"] += int(ids.numel())
        self._sample_ids.add(sample_id)
        _atomic_json(self.root / "manifest.json", self._manifest)
        return record

    def freeze(self) -> None:
        if self._index is None:
            raise RuntimeError("writer is not open")
        self._manifest["status"] = "frozen"
        _atomic_json(self.root / "manifest.json", self._manifest)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        if self._index is not None:
            self._index.close()
            self._index = None
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None


def _read_index(root: Path) -> list[dict[str, Any]]:
    path = root / "index.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt index line {number}: {exc}") from exc
            rows.append(row)
    return rows


def _recover_unindexed_stream_tails(
    root: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Truncate only bytes beyond committed index entries and check boundaries."""

    committed: dict[tuple[int, str], int] = {}
    last_rows: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        segment = int(row["segment"])
        last_rows[segment] = row
        for name in ("input_ids", "loss_mask", "hidden_states"):
            end = int(row["offsets"][name]) + int(row["nbytes"][name])
            key = (segment, name)
            if int(row["offsets"][name]) < committed.get(key, 0):
                raise ValueError(f"overlapping/out-of-order index range for {key}")
            committed[key] = end
    segments = {int(path.name.split("-")[-1]) for path in root.glob("segment-*") if path.is_dir()}
    segments.update(segment for segment, _ in committed)
    for segment in segments:
        segment_dir = root / f"segment-{segment:05d}"
        for name in ("input_ids", "loss_mask", "hidden_states"):
            path = segment_dir / f"{name}.bin"
            expected = committed.get((segment, name), 0)
            actual = path.stat().st_size if path.exists() else 0
            if actual < expected:
                raise ValueError(
                    f"committed index extends past {path}: expected {expected}, found {actual}"
                )
            if actual > expected:
                with path.open("rb+") as handle:
                    handle.truncate(expected)
                    handle.flush()
                    os.fsync(handle.fileno())
        last = last_rows.get(segment)
        if last is None:
            continue
        for name in ("input_ids", "loss_mask", "hidden_states"):
            path = segment_dir / f"{name}.bin"
            offset = int(last["offsets"][name])
            length = int(last["nbytes"][name])
            with path.open("rb") as handle:
                handle.seek(offset)
                raw = handle.read(length)
            if _sha256(raw) != str(last["sha256"][name]):
                raise ValueError(
                    f"checksum mismatch in last committed sample {last['sample_id']} stream {name}"
                )


def rebuild_manifest_from_index(
    root: str | Path,
    *,
    spec: HiddenCacheSpec | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    rows = _read_index(root)
    old_path = root / "manifest.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    resolved_spec = asdict(spec) if spec is not None else old.get("spec")
    value = {
        "schema_version": 1,
        "status": old.get("status", "building"),
        "spec": resolved_spec,
        "samples": len(rows),
        "total_tokens": sum(int(row["tokens"]) for row in rows),
        "segments": sorted({str(int(row["segment"])) for row in rows}, key=int),
        "provenance": old.get("provenance", {}),
    }
    if write:
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(old_path, value)
    return value


class PackedHiddenDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        require_frozen: bool = True,
        verify_checksums: bool = False,
    ) -> None:
        self.root = Path(root)
        self.verify_checksums = bool(verify_checksums)
        self.manifest = rebuild_manifest_from_index(self.root, write=False)
        if require_frozen and self.manifest.get("status") != "frozen":
            raise ValueError("hidden cache is not frozen")
        raw_spec = self.manifest.get("spec")
        if not raw_spec:
            raise ValueError("hidden cache manifest has no spec")
        raw_spec["layer_ids"] = tuple(raw_spec["layer_ids"])
        self.spec = HiddenCacheSpec(**raw_spec)
        self.rows = _read_index(self.root)

    def __len__(self) -> int:
        return len(self.rows)

    def _slice(self, row: Mapping[str, Any], name: str, dtype: str) -> np.ndarray:
        path = self.root / f"segment-{int(row['segment']):05d}" / f"{name}.bin"
        itemsize = np.dtype(dtype).itemsize
        count = int(row["nbytes"][name]) // itemsize
        value = np.memmap(
            path,
            mode="r",
            dtype=dtype,
            offset=int(row["offsets"][name]),
            shape=(count,),
        )
        if self.verify_checksums and _sha256(value.tobytes(order="C")) != str(
            row["sha256"][name]
        ):
            raise ValueError(
                f"checksum mismatch for sample {row['sample_id']} stream {name}"
            )
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        tokens = int(row["tokens"])
        ids = torch.from_numpy(np.array(self._slice(row, "input_ids", "<i8"), copy=True))
        mask = torch.from_numpy(np.array(self._slice(row, "loss_mask", "u1"), copy=True)).bool()
        hidden_u16 = torch.from_numpy(
            np.array(self._slice(row, "hidden_states", "<u2"), copy=True)
        )
        hidden = hidden_u16.view(torch.bfloat16).reshape(
            tokens, len(self.spec.layer_ids), self.spec.hidden_size
        )
        return {
            "sample_id": str(row["sample_id"]),
            "source_index": int(row["source_index"]),
            "input_ids": ids,
            "loss_mask": mask,
            "layer_hidden_states": hidden,
            # SpecForge DFlash consumes this flattened field as the
            # ``hidden_states=...`` argument to its draft model.
            "hidden_states": hidden.flatten(1),
            "metadata": row.get("metadata", {}),
        }


class DFlashHiddenCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("cannot collate an empty batch")
        max_tokens = max(int(row["input_ids"].numel()) for row in rows)
        hidden_width = int(rows[0]["hidden_states"].shape[-1])
        batch = len(rows)
        input_ids = torch.full((batch, max_tokens), self.pad_token_id, dtype=torch.int64)
        attention_mask = torch.zeros((batch, max_tokens), dtype=torch.bool)
        loss_mask = torch.zeros((batch, max_tokens), dtype=torch.bool)
        hidden_states = torch.zeros((batch, max_tokens, hidden_width), dtype=torch.bfloat16)
        for index, row in enumerate(rows):
            length = int(row["input_ids"].numel())
            if int(row["hidden_states"].shape[-1]) != hidden_width:
                raise ValueError("hidden_states widths differ within a batch")
            input_ids[index, :length] = row["input_ids"]
            attention_mask[index, :length] = True
            loss_mask[index, :length] = row["loss_mask"]
            hidden_states[index, :length] = row["hidden_states"]
        return {
            "sample_ids": [str(row["sample_id"]) for row in rows],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "hidden_states": hidden_states,
        }
