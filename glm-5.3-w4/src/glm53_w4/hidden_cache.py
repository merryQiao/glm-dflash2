from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import Dataset


SCHEMA = "formal-glm53-w4a8-hidden-cache-v2"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_shard_record(root: Path, shard: Mapping[str, Any]) -> tuple[Path, list[str], list[int]]:
    filename = str(shard.get("file", ""))
    path = (root / filename).resolve()
    if not filename or path.parent != root.resolve() or path.suffix != ".safetensors":
        raise ValueError("hidden cache shard path must be a direct safetensors child")
    sample_ids = shard.get("sample_ids")
    offsets = shard.get("offsets")
    if not isinstance(sample_ids, list) or not isinstance(offsets, list):
        raise ValueError("hidden cache shard metadata is malformed")
    if len(offsets) != len(sample_ids) + 1 or not offsets or int(offsets[0]) != 0:
        raise ValueError("hidden cache shard offsets are malformed")
    normalized_offsets = [int(value) for value in offsets]
    if any(a > b for a, b in zip(normalized_offsets, normalized_offsets[1:])):
        raise ValueError("hidden cache shard offsets must be monotonic")
    if int(shard.get("tokens", -1)) != normalized_offsets[-1]:
        raise ValueError("hidden cache shard token count differs from offsets")
    normalized_ids = [str(value) for value in sample_ids]
    if any(not value for value in normalized_ids):
        raise ValueError("hidden cache shard contains an empty sample_id")
    return path, normalized_ids, normalized_offsets


class HiddenShardWriter:
    def __init__(
        self,
        root: str | Path,
        *,
        hidden_size: int,
        layer_ids: tuple[int, ...],
        provenance: Mapping[str, Any],
        max_shard_bytes: int = 512 << 20,
    ) -> None:
        self.root = Path(root)
        self.hidden_size = int(hidden_size)
        self.layer_ids = tuple(int(value) for value in layer_ids)
        self.provenance = json.loads(json.dumps(dict(provenance), default=str))
        self.max_shard_bytes = int(max_shard_bytes)
        self._samples: list[dict[str, Any]] = []
        self._bytes = 0
        self._shards: list[dict[str, Any]] = []
        self._sample_ids: set[str] = set()
        self._frozen = False
        self._partial_committed = False

    def __enter__(self) -> "HiddenShardWriter":
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != SCHEMA or manifest.get("status") != "building":
                raise ValueError("existing hidden cache is not resumable")
            if (
                int(manifest.get("hidden_size", -1)) != self.hidden_size
                or tuple(manifest.get("layer_ids", ())) != self.layer_ids
                or manifest.get("provenance") != self.provenance
            ):
                raise ValueError("hidden cache resume contract differs")
            self._shards = list(manifest.get("shards", ()))
            seen: set[str] = set()
            for shard in self._shards:
                path, sample_ids, offsets = _validate_shard_record(self.root, shard)
                if not path.is_file() or _sha256(path) != shard.get("sha256"):
                    raise ValueError(f"hidden cache resume shard is corrupt: {path}")
                if seen.intersection(sample_ids):
                    raise ValueError("hidden cache resume contains duplicate sample_id")
                seen.update(sample_ids)
                shard["sample_ids"] = sample_ids
                shard["offsets"] = offsets
            self._sample_ids = seen
        elif any(self.root.iterdir()):
            raise ValueError(f"output cache contains untracked files: {self.root}")
        else:
            self._write_manifest("building")
        return self

    @property
    def completed_ids(self) -> frozenset[str]:
        return frozenset(self._sample_ids)

    def _write_manifest(self, status: str) -> None:
        manifest = {
            "schema": SCHEMA,
            "status": status,
            "hidden_size": self.hidden_size,
            "layer_ids": list(self.layer_ids),
            "dtype": "bfloat16",
            "final_hidden_semantics": "post_final_norm_lm_head_input",
            "provenance": self.provenance,
            "samples": len(self._sample_ids),
            "tokens": sum(int(value["tokens"]) for value in self._shards),
            "shards": self._shards,
        }
        _atomic_json(self.root / "manifest.json", manifest)

    def append(
        self,
        *,
        sample_id: str,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        aux_hidden_states: torch.Tensor,
        target_final_hidden: torch.Tensor,
    ) -> None:
        sample_id = str(sample_id)
        if not sample_id or sample_id in self._sample_ids:
            raise ValueError("sample_id must be non-empty and unique")
        ids = torch.as_tensor(input_ids, dtype=torch.int64).reshape(-1).cpu()
        mask = torch.as_tensor(loss_mask, dtype=torch.bool).reshape(-1).cpu()
        aux = torch.as_tensor(aux_hidden_states).detach().cpu().contiguous()
        final = torch.as_tensor(target_final_hidden).detach().cpu().contiguous()
        expected_aux = (ids.numel(), len(self.layer_ids), self.hidden_size)
        if tuple(aux.shape) != expected_aux or tuple(final.shape) != (
            ids.numel(),
            self.hidden_size,
        ):
            raise ValueError("hidden cache tensor shape mismatch")
        if (
            len(mask) != len(ids)
            or not bool(mask.any())
            or aux.dtype != torch.bfloat16
            or final.dtype != torch.bfloat16
        ):
            raise ValueError("hidden cache requires aligned IDs/mask and BF16 hidden")
        if bool(((ids < 0)).any()) or not bool(torch.isfinite(aux.float()).all()) or not bool(
            torch.isfinite(final.float()).all()
        ):
            raise ValueError("hidden cache contains invalid IDs or non-finite hidden")
        sample = {
            "sample_id": sample_id,
            "input_ids": ids,
            "loss_mask": mask,
            "aux_hidden_states": aux,
            "target_final_hidden": final,
        }
        size = sum(value.numel() * value.element_size() for value in sample.values() if torch.is_tensor(value))
        if self._samples and self._bytes + size > self.max_shard_bytes:
            self._flush()
        self._samples.append(sample)
        self._bytes += size
        self._sample_ids.add(sample_id)

    def _flush(self) -> None:
        if not self._samples:
            return
        part = len(self._shards)
        path = self.root / f"hidden-{part:05d}.safetensors"
        offsets = [0]
        sample_ids: list[str] = []
        for sample in self._samples:
            offsets.append(offsets[-1] + int(sample["input_ids"].numel()))
            sample_ids.append(str(sample["sample_id"]))
        save_file(
            {
                "offsets": torch.tensor(offsets, dtype=torch.int64),
                "input_ids": torch.cat([value["input_ids"] for value in self._samples]),
                "loss_mask": torch.cat([value["loss_mask"] for value in self._samples]),
                "aux_hidden_states": torch.cat(
                    [value["aux_hidden_states"] for value in self._samples]
                ),
                "target_final_hidden": torch.cat(
                    [value["target_final_hidden"] for value in self._samples]
                ),
            },
            path,
            metadata={"sample_ids": json.dumps(sample_ids)},
        )
        self._shards.append(
            {
                "file": path.name,
                "sha256": _sha256(path),
                "sample_ids": sample_ids,
                "offsets": offsets,
                "tokens": offsets[-1],
            }
        )
        self._samples.clear()
        self._bytes = 0
        self._write_manifest("building")

    def freeze(self) -> None:
        self._flush()
        self._write_manifest("frozen")
        self._frozen = True

    def commit_partial(self) -> None:
        """Durably commit a bounded/debug run while keeping it resumable."""

        self._flush()
        self._write_manifest("building")
        self._partial_committed = True

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None and not self._frozen and not self._partial_committed:
            raise RuntimeError("hidden cache writer exited without freeze()")


class HiddenCacheDataset(Dataset):
    def __init__(self, root: str | Path, *, verify_checksums: bool = False) -> None:
        self.root = Path(root)
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("schema") != SCHEMA or self.manifest.get("status") != "frozen":
            raise ValueError("hidden cache is not a frozen formal GLM-5.3 cache")
        if self.manifest.get("dtype") != "bfloat16":
            raise ValueError("hidden cache must store BF16 hidden streams")
        self.rows: list[tuple[Mapping[str, Any], int]] = []
        seen: set[str] = set()
        for shard in self.manifest["shards"]:
            path, sample_ids, offsets = _validate_shard_record(self.root, shard)
            if not path.is_file():
                raise FileNotFoundError(path)
            path = self.root / str(shard["file"])
            if verify_checksums and _sha256(path) != shard.get("sha256"):
                raise ValueError(f"hidden shard checksum mismatch: {path}")
            if seen.intersection(sample_ids):
                raise ValueError("hidden cache contains duplicate sample_id")
            seen.update(sample_ids)
            shard["sample_ids"] = sample_ids
            shard["offsets"] = offsets
            self.rows.extend((shard, index) for index in range(len(sample_ids)))

    def __len__(self) -> int:
        return len(self.rows)

    def _row(self, index: int) -> tuple[Mapping[str, Any], int, int, int, Path]:
        if not 0 <= int(index) < len(self.rows):
            raise IndexError(index)
        shard, local = self.rows[index]
        start, end = int(shard["offsets"][local]), int(shard["offsets"][local + 1])
        path = self.root / str(shard["file"])
        return shard, local, start, end, path

    def token_fields(self, index: int) -> dict[str, Any]:
        shard, local, start, end, path = self._row(index)
        with safe_open(path, framework="pt", device="cpu") as handle:
            ids = handle.get_slice("input_ids")[start:end]
            mask = handle.get_slice("loss_mask")[start:end]
        return {
            "sample_id": str(shard["sample_ids"][local]),
            "input_ids": ids,
            "loss_mask": mask,
        }

    def get_window(self, index: int, window_start: int, window_end: int) -> dict[str, Any]:
        shard, local, start, end, path = self._row(index)
        sample_tokens = end - start
        window_start = int(window_start)
        window_end = int(window_end)
        if not 0 <= window_start < window_end <= sample_tokens:
            raise ValueError("hidden-cache window is outside the sample")
        absolute_start, absolute_end = start + window_start, start + window_end
        with safe_open(path, framework="pt", device="cpu") as handle:
            ids = handle.get_slice("input_ids")[absolute_start:absolute_end]
            mask = handle.get_slice("loss_mask")[absolute_start:absolute_end]
            aux = handle.get_slice("aux_hidden_states")[absolute_start:absolute_end]
            final = handle.get_slice("target_final_hidden")[absolute_start:absolute_end]
        return {
            "sample_id": str(shard["sample_ids"][local]),
            "input_ids": ids,
            "loss_mask": mask,
            "attention_mask": torch.ones_like(mask, dtype=torch.bool),
            "aux_hidden_states": aux,
            "target_final_hidden": final,
            "position_offset": window_start,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        _, _, start, end, _ = self._row(index)
        return self.get_window(index, 0, end - start)
