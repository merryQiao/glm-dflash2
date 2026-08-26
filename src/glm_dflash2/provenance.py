from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_CONFIG_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "*.index.json",
)
MODEL_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt")
ENDPOINT_MANIFEST_SCHEMA = "glm-sglang-endpoint-v1"
ENDPOINT_RUNTIME_KEYS = (
    "sglang_version",
    "cann_version",
    "image_digest",
    "tp_size",
)


@dataclass(frozen=True)
class DatasetFingerprint:
    digest: str
    repo: str | None
    revision: str | None
    parquet_files: int
    bytes: int


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def load_verified_endpoint_manifest(
    path: str | Path,
    *,
    expected_model_fingerprint: str,
    expected_tokenizer_fingerprint: str,
    expected_served_model_name: str,
) -> dict[str, Any]:
    """Validate the immutable identity advertised by an external SGLang server."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"external endpoint manifest does not exist: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != ENDPOINT_MANIFEST_SCHEMA:
        raise ValueError(f"endpoint manifest schema must be {ENDPOINT_MANIFEST_SCHEMA}")
    if value.get("model_fingerprint") != expected_model_fingerprint:
        raise ValueError("external endpoint model fingerprint differs from local MODEL_PATH")
    if value.get("tokenizer_fingerprint") != expected_tokenizer_fingerprint:
        raise ValueError("external endpoint tokenizer fingerprint differs from local MODEL_PATH")
    if value.get("served_model_name") != expected_served_model_name:
        raise ValueError("external endpoint served model name differs from the request model")
    if value.get("dtype") != "bfloat16":
        raise ValueError("external endpoint must advertise dtype=bfloat16")
    runtime = value.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("external endpoint manifest is missing runtime identity")
    missing = [key for key in ENDPOINT_RUNTIME_KEYS if runtime.get(key) in (None, "")]
    if missing or not isinstance(runtime.get("tp_size"), int) or runtime["tp_size"] < 1:
        raise ValueError(
            "external endpoint manifest is missing runtime identity: "
            + ", ".join(missing or ["tp_size"])
        )
    canonical = _canonical_json_bytes(value)
    return {
        "manifest": value,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "path": str(manifest_path.resolve()),
    }


def _update_file_content(digest: "hashlib._Hash", path: Path, *, chunk_size: int = 1024 * 1024) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)


def _model_files(model_path: Path) -> tuple[list[Path], list[Path]]:
    configs = sorted({path for pattern in MODEL_CONFIG_PATTERNS for path in model_path.glob(pattern) if path.is_file()})
    weights = sorted({path for pattern in MODEL_WEIGHT_PATTERNS for path in model_path.glob(pattern) if path.is_file()})
    return configs, weights


def validate_local_model_artifacts(model_path: Path) -> None:
    if not model_path.is_absolute():
        raise ValueError("model_path must be an absolute local path")
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"config.json not found in model directory: {model_path}")
    tokenizer_files = [
        model_path / "tokenizer_config.json",
        model_path / "tokenizer.json",
        model_path / "tokenizer.model",
    ]
    if not any(path.is_file() for path in tokenizer_files):
        raise FileNotFoundError(f"tokenizer artifacts not found in model directory: {model_path}")
    _, weights = _model_files(model_path)
    if not weights:
        raise FileNotFoundError(f"model weight artifacts not found in model directory: {model_path}")


def local_model_fingerprint(model_path: Path) -> str:
    """Fingerprint configs, indexes and local weight artifacts without hashing a 700B model in full."""

    validate_local_model_artifacts(model_path)
    digest = hashlib.sha256()
    configs, weights = _model_files(model_path)
    for path in configs:
        relative = path.relative_to(model_path).as_posix()
        digest.update(f"config\0{relative}\0{path.stat().st_size}\0".encode())
        _update_file_content(digest, path)
    sample_bytes = 1024 * 1024
    for path in weights:
        stat = path.stat()
        relative = path.relative_to(model_path).as_posix()
        digest.update(f"weight\0{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
        with path.open("rb") as handle:
            head = handle.read(sample_bytes)
            digest.update(head)
            if stat.st_size > sample_bytes:
                handle.seek(max(0, stat.st_size - sample_bytes))
                digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def _download_identity(input_dir: Path) -> tuple[str | None, str | None, bytes]:
    candidates = [input_dir / "DOWNLOAD_REVISION", input_dir.parent / "DOWNLOAD_REVISION"]
    for path in candidates:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        lines = [line.strip() for line in raw.decode("utf-8", errors="replace").splitlines() if line.strip()]
        if len(lines) >= 2:
            return lines[-2], lines[-1], raw
        if lines:
            return None, lines[-1], raw
        return None, None, raw
    return None, None, b""


def dataset_fingerprint(input_dir: Path) -> DatasetFingerprint:
    import pyarrow.parquet as pq

    files = sorted(input_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files found under {input_dir}")
    repo, revision, revision_bytes = _download_identity(input_dir)
    digest = hashlib.sha256()
    digest.update(b"download-revision\0" + revision_bytes)
    total_bytes = 0
    for path in files:
        stat = path.stat()
        total_bytes += stat.st_size
        relative = path.relative_to(input_dir).as_posix()
        metadata = pq.ParquetFile(path).metadata
        digest.update(
            f"parquet\0{relative}\0{stat.st_size}\0{metadata.num_rows}\0{metadata.num_row_groups}\0".encode()
        )
        # Source data is only ~1 GiB. A full content hash is cheap compared with
        # a multi-day 753B rollout and prevents silent same-size replacements.
        _update_file_content(digest, path)
    return DatasetFingerprint(digest.hexdigest(), repo, revision, len(files), total_bytes)
