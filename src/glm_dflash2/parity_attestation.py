from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ATTESTATION_FILENAME = "parity_attestation.json"
ATTESTATION_SCHEMA = "glm-hidden-parity-attestation-v1"


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _target_weight_path(target_io_dir: Path) -> Path:
    manifest_path = target_io_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing target I/O manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = manifest.get("weights_file")
    if not isinstance(name, str) or not name:
        raise ValueError("target I/O manifest is missing weights_file")
    return target_io_dir / name


def _bound_files(cache_dir: Path, target_io_dir: Path) -> dict[str, str]:
    target_weight = _target_weight_path(target_io_dir)
    return {
        "cache_manifest_sha256": _file_sha256(cache_dir / "manifest.json"),
        "cache_index_sha256": _file_sha256(cache_dir / "index.jsonl"),
        "target_io_manifest_sha256": _file_sha256(target_io_dir / "manifest.json"),
        "target_io_weights_sha256": _file_sha256(target_weight),
    }


def write_parity_attestation(
    *,
    cache_dir: str | Path,
    target_io_dir: str | Path,
    parity_gate: str | Path,
    runtime_identity: str | Path,
    reference: str | Path,
    fixture: Mapping[str, Any],
    parity_result: Mapping[str, Any],
) -> Path:
    cache_dir = Path(cache_dir).resolve()
    target_io_dir = Path(target_io_dir).resolve()
    if parity_result.get("passed") is not True:
        raise ValueError("cannot attest a failed parity result")
    value = {
        "schema": ATTESTATION_SCHEMA,
        "passed": True,
        "bindings": _bound_files(cache_dir, target_io_dir),
        "evidence": {
            "parity_gate_sha256": _file_sha256(Path(parity_gate)),
            "runtime_identity_sha256": _file_sha256(Path(runtime_identity)),
            "reference_sha256": _file_sha256(Path(reference)),
        },
        "fixture": dict(fixture),
        "parity": dict(parity_result),
    }
    path = cache_dir / ATTESTATION_FILENAME
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def validate_training_parity_attestation(
    cache_dir: str | Path, target_io_dir: str | Path
) -> dict[str, Any]:
    cache_dir = Path(cache_dir).resolve()
    target_io_dir = Path(target_io_dir).resolve()
    path = cache_dir / ATTESTATION_FILENAME
    if not path.is_file():
        raise ValueError(f"missing parity attestation: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != ATTESTATION_SCHEMA or value.get("passed") is not True:
        raise ValueError("parity attestation is unsupported or not passed")
    expected = value.get("bindings")
    if not isinstance(expected, Mapping):
        raise ValueError("parity attestation is missing file bindings")
    actual = _bound_files(cache_dir, target_io_dir)
    for key, digest in actual.items():
        if expected.get(key) != digest:
            label = "target I/O" if key.startswith("target_io") else "hidden cache"
            raise ValueError(f"{label} differs from parity attestation ({key})")
    fixture = value.get("fixture")
    if not isinstance(fixture, Mapping) or not fixture.get("input_ids_sha256"):
        raise ValueError("parity attestation is missing fixture identity")
    return value
