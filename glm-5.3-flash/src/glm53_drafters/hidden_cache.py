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

from .contracts import TARGET_CONTRACT, validate_loss_mask, validate_token_ids
from .hidden_capture import (
    CAPTURE_ATTESTATION_SCHEMA,
    FINAL_HIDDEN_SEMANTICS,
    CaptureAttestation,
    validate_ascend_a2_evidence,
    validate_live_ascend_a2_evidence,
)


def _compact_json(value: Any, *, canonical: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=canonical,
        separators=(",", ":"),
        default=str,
    )


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


CaptureMappingTuple = tuple[str, int, str, str]


@dataclass(frozen=True)
class HiddenCacheSpec:
    layer_ids: tuple[int, ...]
    hidden_size: int
    dtype: str = "bfloat16"
    input_dtype: str = "int64"
    mask_semantics: str = "dflash_target_token"
    schema_version: int = 2
    capture_mapping: tuple[CaptureMappingTuple, ...] = ()
    target_num_hidden_layers: int = 45
    vocab_size: int = TARGET_CONTRACT.vocab_size
    final_hidden_size: int | None = None
    final_hidden_dtype: str = "bfloat16"
    final_hidden_semantics: str = FINAL_HIDDEN_SEMANTICS

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("aligned GLM-5.3 hidden cache requires schema v2")
        if not self.layer_ids or any(layer < 0 for layer in self.layer_ids):
            raise ValueError("layer_ids must contain non-negative logical layer IDs")
        if tuple(sorted(set(self.layer_ids))) != self.layer_ids:
            raise ValueError("layer_ids must be unique and ordered")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.dtype != "bfloat16" or self.input_dtype != "int64":
            raise ValueError("packed cache requires BF16 hidden and int64 IDs")
        if len(self.capture_mapping) != len(self.layer_ids):
            raise ValueError("schema v2 needs one capture mapping per logical layer")
        mapped = tuple(int(item[1]) for item in self.capture_mapping)
        if mapped != self.layer_ids:
            raise ValueError("capture mapping logical layer order differs from layer_ids")
        if any(
            len(item) != 4 or not item[0] or not item[2] or not item[3]
            for item in self.capture_mapping
        ):
            raise ValueError("capture mapping entries are incomplete")
        final_size = self.hidden_size if self.final_hidden_size is None else int(
            self.final_hidden_size
        )
        object.__setattr__(self, "final_hidden_size", final_size)
        if final_size != self.hidden_size:
            raise ValueError("auxiliary and final hidden widths must match")
        if self.final_hidden_dtype != "bfloat16":
            raise ValueError("schema v2 final hidden must be BF16")
        if self.final_hidden_semantics != FINAL_HIDDEN_SEMANTICS:
            raise ValueError("schema v2 final hidden semantics are invalid")
        if self.target_num_hidden_layers <= max(self.layer_ids):
            raise ValueError("logical layer IDs exceed target decoder depth")
        if self.vocab_size != TARGET_CONTRACT.vocab_size:
            raise ValueError(
                f"schema v2 requires target vocabulary {TARGET_CONTRACT.vocab_size}"
            )


def _spec_from_json(raw: Mapping[str, Any]) -> HiddenCacheSpec:
    value = dict(raw)
    value["layer_ids"] = tuple(int(item) for item in value["layer_ids"])
    value["capture_mapping"] = tuple(
        (str(item[0]), int(item[1]), str(item[2]), str(item[3]))
        for item in value.get("capture_mapping", ())
    )
    return HiddenCacheSpec(**value)


STREAMS = (
    "input_ids",
    "loss_mask",
    "aux_hidden_states",
    "target_final_hidden",
)


def _identity(spec: HiddenCacheSpec, provenance: Mapping[str, Any]) -> str:
    payload = {
        "schema": "glm53-hidden-cache-identity-v1",
        "spec": asdict(spec),
        "provenance": provenance,
    }
    return _sha256(_compact_json(payload, canonical=True).encode("utf-8"))


def _repair_truncated_index(path: Path) -> None:
    if not path.exists():
        return
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return
    boundary = raw.rfind(b"\n")
    repaired = raw[: boundary + 1] if boundary >= 0 else b""
    with path.open("wb") as handle:
        handle.write(repaired)
        handle.flush()
        os.fsync(handle.fileno())


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
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt index line {number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"index line {number} is not an object")
            rows.append(value)
    return rows


def _attestation_record(value: CaptureAttestation | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, CaptureAttestation):
        return value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("capture attestation must be structured evidence")
    raw = dict(value)
    for key in ("logical_layer_ids", "physical_layer_ids", "independent_tap_paths"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return CaptureAttestation(**raw).to_dict()


def _attestation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    passed = failed = missing = 0
    for row in rows:
        evidence = row.get("attestation")
        digest.update(str(row.get("sample_id", "")).encode("utf-8") + b"\0")
        digest.update(_compact_json(evidence, canonical=True).encode("utf-8") + b"\n")
        if evidence is None:
            missing += 1
        elif isinstance(evidence, Mapping) and evidence.get("passed") is True:
            passed += 1
        else:
            failed += 1
    required = len(rows)
    return {
        "schema": CAPTURE_ATTESTATION_SCHEMA,
        "required_rows": required,
        "passed_rows": passed,
        "failed_rows": failed,
        "missing_rows": missing,
        "all_rows_passed": required > 0 and passed == required,
        "rows_sha256": digest.hexdigest(),
    }


def _validate_attestation_for_row(
    evidence: Mapping[str, Any] | None,
    *,
    tokens: int,
    spec: HiddenCacheSpec,
) -> None:
    if evidence is None:
        return
    if int(evidence["token_count"]) != tokens:
        raise ValueError("capture attestation token count differs from cache row")
    if tuple(evidence["logical_layer_ids"]) != spec.layer_ids:
        raise ValueError("capture attestation logical layer order differs")
    expected_physical = tuple(layer + 1 for layer in spec.layer_ids)
    if tuple(evidence["physical_layer_ids"]) != expected_physical:
        raise ValueError("capture attestation physical layer mapping differs")


def _recover_streams(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    committed: dict[tuple[int, str], int] = {}
    for row in rows:
        if set(row.get("nbytes", {})) != set(STREAMS):
            raise ValueError("index stream set differs from schema v2")
        segment = int(row["segment"])
        for name in STREAMS:
            offset = int(row["offsets"][name])
            end = offset + int(row["nbytes"][name])
            key = (segment, name)
            if offset < committed.get(key, 0):
                raise ValueError(f"overlapping index range for {key}")
            committed[key] = end
    segments = {
        int(path.name.rsplit("-", 1)[1])
        for path in root.glob("segment-*")
        if path.is_dir()
    }
    segments.update(segment for segment, _ in committed)
    for segment in segments:
        directory = root / f"segment-{segment:05d}"
        for name in STREAMS:
            path = directory / f"{name}.bin"
            expected = committed.get((segment, name), 0)
            actual = path.stat().st_size if path.exists() else 0
            if actual < expected:
                raise ValueError(f"committed index extends past {path}")
            if actual > expected:
                with path.open("rb+") as handle:
                    handle.truncate(expected)
                    handle.flush()
                    os.fsync(handle.fileno())


def rebuild_manifest_from_index(
    root: str | Path,
    *,
    spec: HiddenCacheSpec | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    rows = _read_index(root)
    path = root / "manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    raw_spec = asdict(spec) if spec is not None else old.get("spec")
    provenance = old.get("provenance", {})
    cache_identity = old.get("cache_identity")
    if cache_identity and raw_spec:
        computed_identity = _identity(_spec_from_json(raw_spec), provenance)
        if cache_identity != computed_identity:
            raise ValueError("cache identity differs from manifest spec/provenance")
    value = {
        "schema_version": 2,
        "status": old.get("status", "building"),
        "production_eligible": old.get("production_eligible") is True,
        "cache_identity": cache_identity,
        "spec": raw_spec,
        "samples": len(rows),
        "total_tokens": sum(int(row["tokens"]) for row in rows),
        "segments": sorted({str(int(row["segment"])) for row in rows}, key=int),
        "provenance": provenance,
        "attestation": _attestation_summary(rows),
        "seal": old.get("seal"),
    }
    if write:
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, value)
    return value


class PackedHiddenWriter:
    """Crash-recoverable, single-writer packed schema-v2 cache."""

    def __init__(
        self,
        root: str | Path,
        *,
        spec: HiddenCacheSpec,
        provenance: Mapping[str, Any] | None = None,
        ascend_a2_attestation: Mapping[str, Any] | None = None,
        max_segment_bytes: int = 64 << 30,
    ) -> None:
        self.root = Path(root)
        self.spec = spec
        self.provenance = json.loads(_compact_json(dict(provenance or {})))
        self._ascend_a2_attestation = (
            dict(ascend_a2_attestation)
            if ascend_a2_attestation is not None
            else None
        )
        if ascend_a2_attestation is not None:
            evidence = dict(ascend_a2_attestation)
            validate_ascend_a2_evidence(evidence)
            if self.provenance.get("ascend_a2_runtime") != evidence:
                raise ValueError(
                    "cache provenance differs from runtime-issued Ascend 910B A2 evidence"
                )
        self.cache_identity = _identity(spec, self.provenance)
        self.max_segment_bytes = int(max_segment_bytes)
        if self.max_segment_bytes < 1:
            raise ValueError("max_segment_bytes must be positive")
        self._lock = self._index = None
        self._files: dict[str, Any] = {}
        self._segment = 0
        self._segment_bytes = 0
        self._sample_ids: set[str] = set()
        self._manifest: dict[str, Any] = {}
        self._production_requested = self.provenance.get("production_eligible") is True
        self._attestation_digest = hashlib.sha256()
        self._attestation_counts = {"required": 0, "passed": 0, "failed": 0, "missing": 0}

    def _track_attestation(self, sample_id: str, evidence: Mapping[str, Any] | None) -> None:
        self._attestation_digest.update(sample_id.encode("utf-8") + b"\0")
        self._attestation_digest.update(
            _compact_json(evidence, canonical=True).encode("utf-8") + b"\n"
        )
        self._attestation_counts["required"] += 1
        if evidence is None:
            self._attestation_counts["missing"] += 1
        elif evidence.get("passed") is True:
            self._attestation_counts["passed"] += 1
        else:
            self._attestation_counts["failed"] += 1

    def _current_attestation_summary(self) -> dict[str, Any]:
        counts = self._attestation_counts
        return {
            "schema": CAPTURE_ATTESTATION_SCHEMA,
            "required_rows": counts["required"],
            "passed_rows": counts["passed"],
            "failed_rows": counts["failed"],
            "missing_rows": counts["missing"],
            "all_rows_passed": (
                counts["required"] > 0 and counts["passed"] == counts["required"]
            ),
            "rows_sha256": self._attestation_digest.hexdigest(),
        }

    def __enter__(self) -> "PackedHiddenWriter":
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / ".writer.lock").open("a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another writer owns {self.root}") from exc
        index_path = self.root / "index.jsonl"
        _repair_truncated_index(index_path)
        rows = _read_index(self.root)
        for row in rows:
            evidence = _attestation_record(row.get("attestation"))
            _validate_attestation_for_row(
                evidence, tokens=int(row.get("tokens", -1)), spec=self.spec
            )
            self._track_attestation(str(row.get("sample_id", "")), evidence)
        self._sample_ids = {str(row["sample_id"]) for row in rows}
        if len(self._sample_ids) != len(rows):
            raise ValueError("hidden cache index contains duplicate sample IDs")
        _recover_streams(self.root, rows)
        existing = rebuild_manifest_from_index(self.root, spec=self.spec, write=False)
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("cache_identity") not in (None, self.cache_identity):
                raise ValueError("cache identity differs from existing manifest")
            if previous.get("spec") and _spec_from_json(previous["spec"]) != self.spec:
                raise ValueError("cache identity/spec differs from existing manifest")
            if previous.get("provenance") and previous["provenance"] != self.provenance:
                raise ValueError("cache identity/provenance differs from existing manifest")
            if previous.get("status") in {"frozen", "smoke_unverified", "incomplete"}:
                raise ValueError("cannot append to a sealed hidden cache")
        self._manifest = {
            "schema_version": 2,
            "status": "building",
            "production_eligible": False,
            "cache_identity": self.cache_identity,
            "spec": asdict(self.spec),
            "samples": existing["samples"],
            "total_tokens": existing["total_tokens"],
            "segments": existing["segments"],
            "provenance": self.provenance,
            "attestation": self._current_attestation_summary(),
            "seal": None,
        }
        _atomic_json(manifest_path, self._manifest)
        self._index = index_path.open("a", encoding="utf-8")
        if existing["segments"]:
            self._segment = max(int(item) for item in existing["segments"])
            directory = self.root / f"segment-{self._segment:05d}"
            self._segment_bytes = sum(
                (directory / f"{name}.bin").stat().st_size
                if (directory / f"{name}.bin").exists()
                else 0
                for name in STREAMS
            )
        self._open_segment(self._segment)
        return self

    def _open_segment(self, segment: int) -> None:
        for handle in self._files.values():
            handle.close()
        directory = self.root / f"segment-{segment:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        self._files = {
            name: (directory / f"{name}.bin").open("ab") for name in STREAMS
        }
        self._segment = segment
        if str(segment) not in self._manifest["segments"]:
            self._manifest["segments"].append(str(segment))
        _fsync_dir(directory)
        _fsync_dir(self.root)

    @staticmethod
    def _bf16_raw(value: torch.Tensor) -> bytes:
        return (
            value.to(torch.bfloat16)
            .contiguous()
            .view(torch.uint16)
            .numpy()
            .astype("<u2", copy=False)
            .tobytes(order="C")
        )

    def append(
        self,
        *,
        sample_id: str,
        source_index: int,
        input_ids: torch.Tensor | Sequence[int],
        loss_mask: torch.Tensor | Sequence[int | bool],
        aux_hidden_states: torch.Tensor,
        target_final_hidden: torch.Tensor,
        attestation: CaptureAttestation | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._index is None:
            raise RuntimeError("writer is not open")
        if not sample_id or sample_id in self._sample_ids:
            raise ValueError(f"duplicate or empty sample_id: {sample_id!r}")
        validated_ids = validate_token_ids(input_ids, vocab_size=self.spec.vocab_size)
        validated_mask = validate_loss_mask(
            loss_mask, expected_length=len(validated_ids)
        )
        ids = torch.tensor(validated_ids, dtype=torch.int64)
        mask = torch.tensor(validated_mask, dtype=torch.bool)
        aux = torch.as_tensor(aux_hidden_states).detach().cpu()
        final = torch.as_tensor(target_final_hidden).detach().cpu()
        expected_aux = (ids.numel(), len(self.spec.layer_ids), self.spec.hidden_size)
        expected_final = (ids.numel(), int(self.spec.final_hidden_size))
        if tuple(aux.shape) != expected_aux:
            raise ValueError(f"aux_hidden_states shape {tuple(aux.shape)} != {expected_aux}")
        if tuple(final.shape) != expected_final:
            raise ValueError(
                f"target_final_hidden shape {tuple(final.shape)} != {expected_final}"
            )
        if not bool(torch.isfinite(aux).all()) or not bool(torch.isfinite(final).all()):
            raise ValueError("hidden states contain NaN or Inf")
        evidence = _attestation_record(attestation)
        _validate_attestation_for_row(
            evidence, tokens=int(ids.numel()), spec=self.spec
        )
        raw = {
            "input_ids": ids.numpy().astype("<i8", copy=False).tobytes(order="C"),
            "loss_mask": mask.numpy().astype("u1", copy=False).tobytes(order="C"),
            "aux_hidden_states": self._bf16_raw(aux),
            "target_final_hidden": self._bf16_raw(final),
        }
        sample_bytes = sum(len(value) for value in raw.values())
        if self._segment_bytes and self._segment_bytes + sample_bytes > self.max_segment_bytes:
            self._open_segment(self._segment + 1)
            self._segment_bytes = 0
        offsets = {name: self._files[name].tell() for name in STREAMS}
        for name in STREAMS:
            handle = self._files[name]
            handle.write(raw[name])
            handle.flush()
            os.fsync(handle.fileno())
        record = {
            "sample_id": sample_id,
            "source_index": int(source_index),
            "segment": self._segment,
            "tokens": int(ids.numel()),
            "offsets": offsets,
            "nbytes": {name: len(raw[name]) for name in STREAMS},
            "sha256": {name: _sha256(raw[name]) for name in STREAMS},
            "metadata": dict(metadata or {}),
            "attestation": evidence,
        }
        self._index.write(_compact_json(record) + "\n")
        self._index.flush()
        os.fsync(self._index.fileno())
        self._sample_ids.add(sample_id)
        self._segment_bytes += sample_bytes
        self._manifest["samples"] += 1
        self._manifest["total_tokens"] += int(ids.numel())
        self._track_attestation(sample_id, evidence)
        self._manifest["attestation"] = self._current_attestation_summary()
        _atomic_json(self.root / "manifest.json", self._manifest)
        return record

    def freeze(self) -> None:
        if self._index is None:
            raise RuntimeError("writer is not open")
        self._index.flush()
        os.fsync(self._index.fileno())
        for handle in self._files.values():
            handle.flush()
            os.fsync(handle.fileno())
        files: dict[str, dict[str, Any]] = {}
        for segment in self._manifest["segments"]:
            directory = self.root / f"segment-{int(segment):05d}"
            for name in STREAMS:
                path = directory / f"{name}.bin"
                relative = path.relative_to(self.root).as_posix()
                files[relative] = {
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
        index_path = self.root / "index.jsonl"
        self._manifest["seal"] = {
            "index_bytes": index_path.stat().st_size,
            "index_sha256": _file_sha256(index_path),
            "rows": int(self._manifest["samples"]),
            "files": files,
            "attestation_rows_sha256": self._manifest["attestation"]["rows_sha256"],
        }
        all_attested = self._manifest["attestation"]["all_rows_passed"] is True
        hardware_attested = False
        if self._ascend_a2_attestation is not None:
            try:
                validate_live_ascend_a2_evidence(self._ascend_a2_attestation)
                hardware_attested = True
                self._manifest["hardware_attestation"] = {"passed": True}
            except Exception as exc:
                self._manifest["hardware_attestation"] = {
                    "passed": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        if self._production_requested and all_attested and hardware_attested:
            self._manifest["status"] = "frozen"
            self._manifest["production_eligible"] = True
        elif self._production_requested:
            self._manifest["status"] = "incomplete"
            self._manifest["production_eligible"] = False
        else:
            self._manifest["status"] = "smoke_unverified"
            self._manifest["production_eligible"] = False
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


class PackedHiddenDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        require_frozen: bool = True,
        allow_smoke_unverified: bool = False,
        verify_checksums: bool = False,
    ) -> None:
        self.root = Path(root)
        self.manifest = rebuild_manifest_from_index(self.root, write=False)
        self.verify_checksums = bool(verify_checksums)
        status = self.manifest.get("status")
        allowed = {"frozen"}
        if allow_smoke_unverified:
            allowed.add("smoke_unverified")
        if require_frozen and status not in allowed:
            if status == "smoke_unverified":
                raise ValueError("smoke_unverified cache requires explicit opt-in")
            raise ValueError("hidden cache is not frozen")
        raw_spec = self.manifest.get("spec")
        if not raw_spec:
            raise ValueError("hidden cache manifest has no spec")
        self.spec = _spec_from_json(raw_spec)
        self.rows = _read_index(self.root)
        self.cache_identity = str(self.manifest.get("cache_identity") or "")
        if not self.cache_identity:
            raise ValueError("hidden cache manifest has no immutable cache identity")
        if require_frozen:
            self._validate_seal()

    def _validate_seal(self) -> None:
        seal = self.manifest.get("seal")
        if not isinstance(seal, Mapping):
            raise ValueError("sealed schema-v2 cache is missing integrity seal")
        index = self.root / "index.jsonl"
        if int(seal.get("index_bytes", -1)) != index.stat().st_size:
            raise ValueError("sealed index byte length differs")
        if str(seal.get("index_sha256")) != _file_sha256(index):
            raise ValueError("sealed index checksum differs")
        if int(seal.get("rows", -1)) != len(self.rows):
            raise ValueError("sealed row count differs")
        attestation = self.manifest.get("attestation")
        if not isinstance(attestation, Mapping):
            raise ValueError("sealed cache has no capture attestation summary")
        if seal.get("attestation_rows_sha256") != attestation.get("rows_sha256"):
            raise ValueError("sealed capture attestation digest differs")
        if self.manifest.get("status") == "frozen" and (
            self.manifest.get("production_eligible") is not True
            or attestation.get("all_rows_passed") is not True
        ):
            raise ValueError("production cache lacks complete numeric attestation")
        if self.manifest.get("status") == "frozen":
            provenance = self.manifest.get("provenance")
            if not isinstance(provenance, Mapping) or provenance.get(
                "production_eligible"
            ) is not True:
                raise ValueError("production cache provenance is not eligible")
            validate_ascend_a2_evidence(provenance.get("ascend_a2_runtime"))
        files = seal.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("sealed cache has no stream inventory")
        for relative, expected in files.items():
            path = self.root / str(relative)
            if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
                raise ValueError(f"sealed stream size differs for {relative}")

    def __len__(self) -> int:
        return len(self.rows)

    def _slice(self, row: Mapping[str, Any], name: str, dtype: str) -> np.ndarray:
        path = self.root / f"segment-{int(row['segment']):05d}" / f"{name}.bin"
        nbytes = int(row["nbytes"][name])
        itemsize = np.dtype(dtype).itemsize
        if nbytes % itemsize:
            raise ValueError(f"stream {name} byte length is not dtype aligned")
        value = np.memmap(
            path,
            mode="r",
            dtype=dtype,
            offset=int(row["offsets"][name]),
            shape=(nbytes // itemsize,),
        )
        if self.verify_checksums and _sha256(value.tobytes(order="C")) != str(
            row["sha256"][name]
        ):
            raise ValueError(
                f"checksum mismatch for sample {row['sample_id']} stream {name}"
            )
        return value

    def _bf16(
        self, row: Mapping[str, Any], name: str, shape: tuple[int, ...]
    ) -> torch.Tensor:
        value = torch.from_numpy(np.array(self._slice(row, name, "<u2"), copy=True))
        return value.view(torch.bfloat16).reshape(shape)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        tokens = int(row["tokens"])
        ids = torch.from_numpy(np.array(self._slice(row, "input_ids", "<i8"), copy=True))
        mask = torch.from_numpy(
            np.array(self._slice(row, "loss_mask", "u1"), copy=True)
        ).bool()
        aux = self._bf16(
            row,
            "aux_hidden_states",
            (tokens, len(self.spec.layer_ids), self.spec.hidden_size),
        )
        final = self._bf16(
            row,
            "target_final_hidden",
            (tokens, int(self.spec.final_hidden_size)),
        )
        return {
            "sample_id": str(row["sample_id"]),
            "source_index": int(row["source_index"]),
            "input_ids": ids,
            "loss_mask": mask,
            "aux_hidden_states": aux,
            "layer_hidden_states": aux,
            "hidden_states": aux.flatten(1),
            "target_final_hidden": final,
            "metadata": row.get("metadata", {}),
            "cache_identity": self.cache_identity,
        }


def validate_frozen_hidden_cache(
    root: str | Path, *, allow_smoke_unverified: bool = False
) -> dict[str, Any]:
    dataset = PackedHiddenDataset(
        root,
        allow_smoke_unverified=allow_smoke_unverified,
        verify_checksums=True,
    )
    for index in range(len(dataset)):
        row = dataset[index]
        for name in ("layer_hidden_states", "target_final_hidden"):
            if not bool(torch.isfinite(row[name].float()).all()):
                raise ValueError(f"sample {row['sample_id']} has non-finite {name}")
    return dict(dataset.manifest)


class DFlashHiddenCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("cannot collate an empty batch")
        max_tokens = max(int(row["input_ids"].numel()) for row in rows)
        hidden_width = int(rows[0]["hidden_states"].shape[-1])
        final_width = int(rows[0]["target_final_hidden"].shape[-1])
        batch = len(rows)
        ids = torch.full((batch, max_tokens), self.pad_token_id, dtype=torch.int64)
        attention = torch.zeros((batch, max_tokens), dtype=torch.bool)
        loss = torch.zeros((batch, max_tokens), dtype=torch.bool)
        hidden = torch.zeros(
            (batch, max_tokens, hidden_width), dtype=torch.bfloat16
        )
        final = torch.zeros((batch, max_tokens, final_width), dtype=torch.bfloat16)
        for index, row in enumerate(rows):
            length = int(row["input_ids"].numel())
            if int(row["hidden_states"].shape[-1]) != hidden_width:
                raise ValueError("hidden widths differ within batch")
            if int(row["target_final_hidden"].shape[-1]) != final_width:
                raise ValueError("final hidden widths differ within batch")
            ids[index, :length] = row["input_ids"]
            attention[index, :length] = True
            loss[index, :length] = row["loss_mask"]
            hidden[index, :length] = row["hidden_states"]
            final[index, :length] = row["target_final_hidden"]
        return {
            "sample_ids": [str(row["sample_id"]) for row in rows],
            "input_ids": ids,
            "attention_mask": attention,
            "loss_mask": loss,
            "hidden_states": hidden,
            "target_final_hidden": final,
        }
