from __future__ import annotations

import hashlib
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


def tokenizer_fingerprint(model_dir: str | Path) -> str:
    """Hash tokenizer artifacts required by the frozen Stage C contract."""

    root = Path(model_dir)
    patterns = (
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "chat_template*.jinja",
        "*processor_config.json",
    )
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in root.glob(pattern)
            if path.is_file()
        }
    )
    tokenizer_paths = (
        root / "tokenizer_config.json",
        root / "tokenizer.json",
        root / "tokenizer.model",
    )
    if not any(path.is_file() for path in tokenizer_paths):
        raise FileNotFoundError(f"no tokenizer artifacts found under {root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_revision(config: dict[str, Any], model_fingerprint: str) -> str:
    """Return an explicit model revision or its immutable local fingerprint."""

    for key in ("_commit_hash", "revision", "model_revision"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return model_fingerprint


def _update_file_content(
    digest: "hashlib._Hash", path: Path, *, chunk_size: int = 1024 * 1024
) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)


def _model_files(model_path: Path) -> tuple[list[Path], list[Path]]:
    configs = sorted(
        {
            path
            for pattern in MODEL_CONFIG_PATTERNS
            for path in model_path.glob(pattern)
            if path.is_file()
        }
    )
    weights = sorted(
        {
            path
            for pattern in MODEL_WEIGHT_PATTERNS
            for path in model_path.glob(pattern)
            if path.is_file()
        }
    )
    return configs, weights


def validate_local_model_artifacts(model_path: Path) -> None:
    if not model_path.is_absolute():
        raise ValueError("model_path must be an absolute local path")
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"config.json not found in model directory: {model_path}")
    tokenizer_files = (
        model_path / "tokenizer_config.json",
        model_path / "tokenizer.json",
        model_path / "tokenizer.model",
    )
    if not any(path.is_file() for path in tokenizer_files):
        raise FileNotFoundError(f"tokenizer artifacts not found in model directory: {model_path}")
    _, weights = _model_files(model_path)
    if not weights:
        raise FileNotFoundError(f"model weight artifacts not found in model directory: {model_path}")


def local_model_fingerprint(model_path: Path) -> str:
    """Fingerprint every local config, tokenizer, index, and weight shard."""

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
