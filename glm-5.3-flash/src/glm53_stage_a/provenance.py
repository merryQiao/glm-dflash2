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
    "chat_template*.jinja",
    "*processor_config.json",
    "*.index.json",
)
MODEL_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt")
ENDPOINT_MANIFEST_SCHEMA = "glm-sglang-endpoint-v1"
ENDPOINT_RUNTIME_KEYS = (
    "sglang_version",
    "cann_version",
    "image_digest",
    "tp_size",
    "device",
    "attention_backend",
    "reasoning_parser",
    "tool_call_parser",
    "context_length",
    "max_total_tokens",
)
ENDPOINT_OPTIONAL_RUNTIME_KEYS = (
    "quantization",
    "moe_a2a_backend",
    "deepep_mode",
)


@dataclass(frozen=True)
class DatasetFingerprint:
    digest: str
    repo: str | None
    revision: str | None
    parquet_files: int
    bytes: int


def tokenizer_fingerprint(model_dir: str | Path) -> str:
    """Hash the tokenizer artifacts required to reproduce rendered prompts."""

    model_dir = Path(model_dir)
    patterns = (
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "chat_template*.jinja",
        "*processor_config.json",
    )
    digest = hashlib.sha256()
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in model_dir.glob(pattern)
            if path.is_file()
        }
    )
    tokenizer_paths = {
        model_dir / "tokenizer_config.json",
        model_dir / "tokenizer.json",
        model_dir / "tokenizer.model",
    }
    if not any(path.is_file() for path in tokenizer_paths):
        raise FileNotFoundError(f"no tokenizer artifacts found under {model_dir}")
    for path in paths:
        relative = path.relative_to(model_dir).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def target_vocab_size(config: dict[str, Any]) -> int:
    """Read vocab size from either a text sub-config or a flat model config."""

    text_config = config.get("text_config")
    candidates = (
        text_config.get("vocab_size") if isinstance(text_config, dict) else None,
        config.get("vocab_size"),
    )
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise ValueError("model config does not contain a positive vocab_size")


def model_revision(config: dict[str, Any], model_fingerprint: str) -> str:
    """Return an explicit model revision or its immutable local fingerprint."""

    for key in ("_commit_hash", "revision", "model_revision"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return model_fingerprint


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def load_endpoint_manifest_attestation(
    path: str | Path,
    *,
    expected_model_fingerprint: str,
    expected_tokenizer_fingerprint: str,
    expected_served_model_name: str,
) -> dict[str, Any]:
    """Validate an operator claim without pretending the endpoint proved its weights."""

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
    missing.extend(key for key in ENDPOINT_OPTIONAL_RUNTIME_KEYS if key not in runtime)
    positive_integer_keys = ("tp_size", "context_length", "max_total_tokens")
    invalid_integer_keys = [
        key
        for key in positive_integer_keys
        if not isinstance(runtime.get(key), int) or runtime[key] < 1
    ]
    if missing or invalid_integer_keys:
        raise ValueError(
            "external endpoint manifest is missing runtime identity: "
            + ", ".join(missing + invalid_integer_keys)
        )
    canonical = _canonical_json_bytes(value)
    return {
        "manifest": value,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "path": str(manifest_path.resolve()),
        "weight_identity_status": "operator_attested",
        "weight_identity_verified": False,
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
    """Fingerprint the full content of every config, index, tokenizer and weight shard."""

    validate_local_model_artifacts(model_path)
    digest = hashlib.sha256()
    configs, weights = _model_files(model_path)
    for path in configs:
        relative = path.relative_to(model_path).as_posix()
        digest.update(f"config\0{relative}\0{path.stat().st_size}\0".encode())
        _update_file_content(digest, path)
    for path in weights:
        stat = path.stat()
        relative = path.relative_to(model_path).as_posix()
        digest.update(f"weight\0{relative}\0{stat.st_size}\0".encode())
        _update_file_content(digest, path)
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
