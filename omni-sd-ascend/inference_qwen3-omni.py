#!/usr/bin/env python
"""Profile the production Qwen3-Omni Thinker path on vLLM-Ascend.

This entry point deliberately profiles only the Thinker used by Stage A. It
does not load or report Talker/Code2Wav timings. Generation uses the exact same
engine, processor, request builder, and sampling provider as trajectory
generation, so its throughput is representative of that production path.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from omni_sd.ascend_runtime import runtime_identity  # noqa: E402
from omni_sd.data_io import atomic_write_json  # noqa: E402
from omni_sd.inference_profile import (  # noqa: E402
    ProfileContractError,
    aggregate_evaluation,
    aggregate_profile_performance,
    component_availability,
    evaluation_result,
    normalize_input_record,
    profile_batch_kind,
    profile_batches,
    reduce_hbm_measurements,
    request_stage_latencies_seconds,
    validate_evaluation_contract,
    validate_hbm_worker_identities,
    worker_reset_npu_peak,
    worker_snapshot_npu_memory,
)
from omni_sd.thinker_data import stable_hex  # noqa: E402
from omni_sd.thinker_generation import (  # noqa: E402
    DEFAULT_CONFIG,
    read_config,
)
from omni_sd.vllm_ascend_generation import (  # noqa: E402
    completion_payload,
    engine_kwargs,
    load_engine,
    prepare_request,
    sampling_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--conditions-parquet", type=Path)
    parser.add_argument("--audio", type=Path, action="append", default=[])
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--video", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("outputs/qwen3_omni_thinker_profile.jsonl"),
    )
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--success-marker", type=Path)
    parser.add_argument("--allow-missing-hbm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ProfileContractError("--limit must be positive")
    if args.warmup < 0:
        raise ProfileContractError("--warmup must be non-negative")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ProfileContractError("--batch-size must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ProfileContractError("--max-new-tokens must be positive")
    if args.text is None and (args.audio or args.image or args.video):
        raise ProfileContractError("--audio/--image/--video require --text")


class ProfileOutputLock:
    """Lock every final artifact for the entire inference-and-publish run."""

    def __init__(self, targets: Path | Iterable[Path]):
        values = [targets] if isinstance(targets, Path) else list(targets)
        if not values:
            raise ValueError("profile output lock needs at least one target")
        lock_paths = {
            target.resolve().with_suffix(target.suffix + ".lock")
            for target in values
        }
        self.paths = sorted(lock_paths, key=str)
        self._handles: list[Any] = []

    def __enter__(self) -> "ProfileOutputLock":
        import fcntl

        try:
            for path in self.paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    raise
                handle.seek(0)
                handle.truncate()
                handle.write(f"pid={os.getpid()}\n")
                handle.flush()
                os.fsync(handle.fileno())
                self._handles.append(handle)
        except BlockingIOError as error:
            self._release()
            raise RuntimeError(
                "one or more profile outputs are already locked"
            ) from error
        return self

    def _release(self) -> None:
        import fcntl

        for handle in reversed(self._handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        self._handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._release()


def _jsonl_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProfileContractError(f"{path}:{line_number} is not an object")
            rows.append(
                normalize_input_record(
                    value,
                    index=line_number - 1,
                    base_dir=path.resolve().parent,
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _parquet_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "condition_id",
        "source",
        "source_subset",
        "modality",
        "language",
        "task",
        "messages_json",
        "tools_json",
        "media_json",
    }
    parquet = pq.ParquetFile(path)
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise ProfileContractError(f"condition Parquet missing {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    columns = sorted(required)
    if "evaluation_json" in parquet.schema_arrow.names:
        columns.append("evaluation_json")
    for batch in parquet.iter_batches(columns=columns, batch_size=256):
        rows.extend(batch.to_pylist())
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
    return rows


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.text is not None:
        row = normalize_input_record(
            {
                "id": "command-line",
                "text": args.text,
                "audio": [str(path) for path in args.audio],
                "image": [str(path) for path in args.image],
                "video": [str(path) for path in args.video],
            },
            index=0,
            base_dir=Path.cwd(),
        )
        return [row]
    if args.input_jsonl is not None:
        rows = _jsonl_rows(args.input_jsonl, args.limit)
    else:
        assert args.conditions_parquet is not None
        rows = _parquet_rows(args.conditions_parquet, args.limit)
    if not rows:
        raise ProfileContractError("input contains no inference records")
    return rows


def effective_config(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    if args.max_new_tokens is not None:
        config = deepcopy(config)
        config["generation"]["max_new_tokens"] = int(args.max_new_tokens)
    return config


def batch_sizes(config: dict[str, Any], override: int | None) -> dict[str, int]:
    if override is None:
        return {key: int(value) for key, value in config["vllm_batch_size"].items()}
    return {key: int(override) for key in config["vllm_batch_size"]}


def _generate(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    llm: Any,
    processor: Any,
    sampling_params_class: Any,
) -> tuple[list[Any], dict[str, Any]]:
    outer_started = time.perf_counter()
    prepared: list[Any] = []
    preprocess_seconds: list[float] = []
    for row in rows:
        preprocess_started = time.perf_counter()
        prepared.append(prepare_request(row, config, processor))
        preprocess_elapsed = time.perf_counter() - preprocess_started
        if preprocess_elapsed <= 0:
            raise RuntimeError("preprocessing clock did not advance")
        preprocess_seconds.append(preprocess_elapsed)
    params = [
        sampling_params_class(**sampling_kwargs(config, str(row["condition_id"])))
        for row in rows
    ]
    engine_started = time.perf_counter()
    outputs = llm.generate(
        [request for request, _ in prepared],
        sampling_params=params,
        use_tqdm=False,
    )
    engine_seconds = time.perf_counter() - engine_started
    end_to_end_seconds = time.perf_counter() - outer_started
    if engine_seconds <= 0 or end_to_end_seconds <= 0 or len(outputs) != len(rows):
        raise RuntimeError(f"vLLM returned {len(outputs)}/{len(rows)} requests")
    return list(outputs), {
        "preprocess_request_seconds": preprocess_seconds,
        "preprocess_seconds": sum(preprocess_seconds),
        "engine_seconds": engine_seconds,
        "end_to_end_seconds": end_to_end_seconds,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_profile_bundle(
    *,
    output_jsonl: Path,
    profile_json: Path,
    success_marker: Path,
    records: Iterable[dict[str, Any]],
    profile: dict[str, Any],
) -> None:
    """Publish results together; the checksum marker is the completion gate."""

    targets = (output_jsonl, profile_json, success_marker)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.tmp"
    output_tmp = output_jsonl.with_name(f".{output_jsonl.name}{suffix}")
    profile_tmp = profile_json.with_name(f".{profile_json.name}{suffix}")
    marker_tmp = success_marker.with_name(f".{success_marker.name}{suffix}")
    temporaries = (output_tmp, profile_tmp, marker_tmp)
    published: list[Path] = []
    backups: dict[Path, Path] = {}
    try:
        with output_tmp.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        with profile_tmp.open("w", encoding="utf-8") as handle:
            json.dump(profile, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        marker = {
            "status": "PASS",
            "output_jsonl": {
                "path": str(output_jsonl.resolve()),
                "sha256": _sha256_file(output_tmp),
            },
            "profile_json": {
                "path": str(profile_json.resolve()),
                "sha256": _sha256_file(profile_tmp),
            },
        }
        with marker_tmp.open("w", encoding="utf-8") as handle:
            json.dump(marker, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for target in targets:
            if target.exists():
                backup = target.with_name(f".{target.name}{suffix}.backup")
                os.replace(target, backup)
                backups[target] = backup
        for temporary, target in (
            (output_tmp, output_jsonl),
            (profile_tmp, profile_json),
            (marker_tmp, success_marker),
        ):
            os.replace(temporary, target)
            published.append(target)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    except BaseException:
        for target in published:
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                os.replace(backup, target)
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _media_inventory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for row in rows:
        media_values: list[tuple[str, str]] = []
        for item in json.loads(str(row.get("media_json", "[]"))):
            if isinstance(item, dict):
                media_values.append(
                    (str(item.get("type", "")), str(item.get("path", "")))
                )
        for message in json.loads(str(row.get("messages_json", "[]"))):
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                media_type = str(item.get("type", ""))
                if media_type in {"audio", "image", "video"}:
                    media_values.append(
                        (media_type, str(item.get(media_type, "")))
                    )
        for media_type, value in media_values:
            if value.startswith(("http://", "https://")):
                raise ProfileContractError(
                    "strict comparison identity requires materialized local media"
                )
            if value.startswith("data:"):
                payload = value.encode("utf-8")
                inventory.append(
                    {
                        "condition_id": str(row["condition_id"]),
                        "type": media_type,
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                continue
            path = Path(value.removeprefix("file://"))
            if not path.is_file():
                raise FileNotFoundError(path)
            inventory.append(
                {
                    "condition_id": str(row["condition_id"]),
                    "type": media_type,
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return inventory


def _comparison_identity(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    sizes: dict[str, int],
    warmup: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    batch_ids = [
        [str(row["condition_id"]) for row in batch]
        for batch in profile_batches(rows, sizes)
    ]
    payload = {
        "contract": "omni_vllm_profile_v1",
        "model": config["model"],
        "generation": config["generation"],
        "engine": engine_kwargs(config),
        "runtime": runtime,
        "rows": rows,
        "media_artifacts": _media_inventory(rows),
        "batch_condition_ids": batch_ids,
        "sampling": [
            sampling_kwargs(config, str(row["condition_id"])) for row in rows
        ],
        "warmup": warmup,
        "warmup_policy": {
            "batch_shape": "first_measured_batch_per_actual_modality",
            "post_warmup_mm_cache_reset": warmup > 0,
        },
        "measurement_rounds": 1,
    }
    return {
        "contract": payload["contract"],
        "fingerprint": stable_hex(payload),
        "measurement_rounds": 1,
        "artifact_binding": "declared_model_and_processor_revisions",
        "strict_artifact_manifest_available": False,
        "strict_artifact_manifest_reason": (
            "no immutable file-level model artifact manifest was supplied"
        ),
    }


def dry_run_profile(
    config: dict[str, Any], rows: list[dict[str, Any]], sizes: dict[str, int]
) -> dict[str, Any]:
    first_id = str(rows[0]["condition_id"])
    return {
        "status": "DRY_RUN",
        "component": "thinker",
        "backend": "vllm_ascend",
        "requests": len(rows),
        "batch_sizes": sizes,
        "engine": engine_kwargs(config),
        "sampling": sampling_kwargs(config, first_id),
        "model": config["model"],
        "generation": config["generation"],
    }


def _warmup_batches(
    rows: list[dict[str, Any]], sizes: dict[str, int]
) -> list[list[dict[str, Any]]]:
    representatives: dict[str, list[dict[str, Any]]] = {}
    for batch in profile_batches(rows, sizes):
        representatives.setdefault(profile_batch_kind(batch[0]), batch)
    return list(representatives.values())


def run_profile(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    sizes: dict[str, int],
    warmup: int,
    *,
    allow_missing_hbm: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_evaluation_contract(rows)
    identity = runtime_identity(hardware=str(config["runtime"]["hardware"]))
    expected_devices = int(config["runtime"]["tensor_parallel_size"])
    if len(identity["visible_devices"]) != expected_devices:
        raise RuntimeError(
            "visible Ascend device count must equal tensor_parallel_size"
        )
    comparison_identity = _comparison_identity(
        config=config,
        rows=rows,
        sizes=sizes,
        warmup=warmup,
        runtime=identity,
    )

    load_started = time.perf_counter()
    llm, processor, sampling_params_class = load_engine(config)
    model_load_seconds = time.perf_counter() - load_started

    warmup_measurements = {
        "preprocess_seconds": 0.0,
        "engine_seconds": 0.0,
        "end_to_end_seconds": 0.0,
    }
    for _ in range(warmup):
        for warmup_batch in _warmup_batches(rows, sizes):
            _, measurement = _generate(
                warmup_batch,
                config,
                llm,
                processor,
                sampling_params_class,
            )
            for key in warmup_measurements:
                warmup_measurements[key] += float(measurement[key])
    if warmup > 0:
        try:
            llm.reset_mm_cache()
        except Exception as error:
            raise RuntimeError(
                "vLLM multimodal cache reset failed after warmup"
            ) from error

    records: list[dict[str, Any]] = []
    request_measurements: list[dict[str, Any]] = []
    batch_measurements: list[dict[str, Any]] = []
    hbm_snapshots: list[list[dict[str, Any]]] = []
    hbm_error: str | None = None
    eos = int(config["generation"]["eos_token_id"])
    for batch_index, batch in enumerate(profile_batches(rows, sizes)):
        try:
            reset_identities = llm.collective_rpc(worker_reset_npu_peak)
            validate_hbm_worker_identities(reset_identities, expected_devices)
        except Exception as error:
            if not allow_missing_hbm:
                raise RuntimeError("required vLLM worker HBM reset failed") from error
            hbm_error = f"worker HBM reset unavailable: {type(error).__name__}: {error}"
        outputs, measurement = _generate(
            batch, config, llm, processor, sampling_params_class
        )
        modality = profile_batch_kind(batch[0])
        batch_measurements.append(
            {
                "batch_index": batch_index,
                "modality": modality,
                "requests": len(batch),
                "engine_seconds": measurement["engine_seconds"],
                "end_to_end_seconds": measurement["end_to_end_seconds"],
            }
        )
        if hbm_error is None:
            try:
                hbm_snapshots.append(
                    list(llm.collective_rpc(worker_snapshot_npu_memory))
                )
            except Exception as error:
                if not allow_missing_hbm:
                    raise RuntimeError(
                        "required vLLM worker HBM snapshot failed"
                    ) from error
                hbm_error = (
                    f"worker HBM snapshot unavailable: {type(error).__name__}: {error}"
                )
                hbm_snapshots.clear()
        for request_index, (row, output) in enumerate(
            zip(batch, outputs, strict=True)
        ):
            payload = completion_payload(output, eos)
            stage_latencies = request_stage_latencies_seconds(output)
            preprocess_elapsed = float(
                measurement["preprocess_request_seconds"][request_index]
            )
            request_measurements.append(
                {
                    "condition_id": str(row["condition_id"]),
                    "modality": modality,
                    "prompt_tokens": int(payload["prompt_tokens"]),
                    "completion_tokens": int(payload["response_tokens"]),
                    "preprocess_seconds": preprocess_elapsed,
                    **stage_latencies,
                    "batch_index": batch_index,
                }
            )
            row_evaluation = evaluation_result(
                row.get("evaluation_json"), str(payload["response_text"])
            )
            records.append(
                {
                    "condition_id": str(row["condition_id"]),
                    "modality": modality,
                    "response_text": payload["response_text"],
                    "prompt_token_ids": payload["prompt_token_ids"],
                    "response_token_ids": payload["response_token_ids"],
                    "prompt_tokens": payload["prompt_tokens"],
                    "response_tokens": payload["response_tokens"],
                    "finish_reason": payload["finish_reason"],
                    "evaluation_json": row.get("evaluation_json"),
                    "evaluation_result": row_evaluation,
                    "preprocess_latency_ms": preprocess_elapsed * 1000.0,
                    **{
                        f"request_{name.removesuffix('_seconds')}_latency_ms": (
                            value * 1000.0 if value is not None else None
                        )
                        for name, value in stage_latencies.items()
                    },
                    "batch_index": batch_index,
                    "batch_engine_latency_ms": measurement["engine_seconds"] * 1000.0,
                    "batch_end_to_end_latency_ms": measurement["end_to_end_seconds"]
                    * 1000.0,
                }
            )

    performance = aggregate_profile_performance(
        requests=request_measurements,
        batches=batch_measurements,
        model_load_seconds=model_load_seconds,
        warmup=warmup_measurements,
    )
    if hbm_error is None:
        try:
            memory = reduce_hbm_measurements(
                hbm_snapshots,
                tensor_parallel_size=expected_devices,
            )
        except Exception as error:
            if not allow_missing_hbm:
                raise
            memory = {
                "available": False,
                "source": "torch_npu_allocator",
                "reason": f"malformed worker HBM telemetry: {error}",
            }
    else:
        memory = {
            "available": False,
            "source": "torch_npu_allocator",
            "reason": hbm_error,
        }
    evaluation = aggregate_evaluation(records)
    modalities = [str(request["modality"]) for request in request_measurements]
    profile = {
        "status": "PASS",
        "component": "thinker",
        "backend": "vllm_ascend",
        "model": config["model"],
        "generation": config["generation"],
        "engine": engine_kwargs(config),
        "runtime": identity,
        "warmup_policy": {
            "rounds": warmup,
            "batch_shape": "first_measured_batch_per_actual_modality",
            "post_warmup_mm_cache_reset": warmup > 0,
        },
        "comparison_identity": comparison_identity,
        "variant_identity": {"kind": "target_only"},
        "performance": performance,
        "memory": memory,
        "evaluation": evaluation,
        "components": component_availability(modalities),
    }
    return records, profile


def print_summary(profile: dict[str, Any]) -> None:
    if profile["status"] == "DRY_RUN":
        print(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True))
        return
    perf = profile["performance"]
    overall = perf["overall"]
    print("Qwen3-Omni Thinker performance (vLLM-Ascend)")
    print(f"  requests:                {overall['requests']}")
    print(f"  model load:              {perf['model_load_seconds']:.3f} s")
    print(f"  warmup engine:           {perf['warmup']['engine_seconds']:.3f} s")
    print(f"  measured engine:         {overall['engine_seconds']:.3f} s")
    print(f"  measured end-to-end:     {overall['end_to_end_seconds']:.3f} s")
    print(
        "  engine completion TPS:   "
        f"{overall['engine']['completion_tokens_per_second']:.3f}"
    )
    print(
        "  end-to-end completion TPS: "
        f"{overall['end_to_end']['completion_tokens_per_second']:.3f}"
    )
    latency = overall["request_engine_inference_latency_ms"]
    if latency["available"]:
        print(
            "  engine inference p50/p95: "
            f"{latency['p50']:.3f} / {latency['p95']:.3f} ms"
        )
    else:
        print("  engine inference latency: unavailable")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = effective_config(args)
    rows = load_rows(args)
    sizes = batch_sizes(config, args.batch_size)
    profile_path = args.profile_json or args.output_jsonl.with_suffix(".profile.json")
    success_marker = args.success_marker or profile_path.with_name(
        f"{profile_path.name}.SUCCESS.json"
    )
    targets = (
        [profile_path]
        if args.dry_run
        else [args.output_jsonl, profile_path, success_marker]
    )
    with ProfileOutputLock(targets):
        if not args.overwrite:
            existing = [path for path in targets if path.exists()]
            if existing:
                raise FileExistsError(f"output already exists: {existing[0]}")

        if args.dry_run:
            profile = dry_run_profile(config, rows, sizes)
            atomic_write_json(profile_path, profile)
        else:
            records, profile = run_profile(
                config,
                rows,
                sizes,
                args.warmup,
                allow_missing_hbm=args.allow_missing_hbm,
            )
            write_profile_bundle(
                output_jsonl=args.output_jsonl,
                profile_json=profile_path,
                success_marker=success_marker,
                records=records,
                profile=profile,
            )
    print_summary(profile)
    print(f"Profile: {profile_path.resolve()}")
    if not args.dry_run:
        print(f"Results: {args.output_jsonl.resolve()}")


if __name__ == "__main__":
    main()
