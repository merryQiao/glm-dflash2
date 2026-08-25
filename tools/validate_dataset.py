#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm_dflash2.data import SOURCE_COLUMNS, normalize_row, parquet_files
from glm_dflash2.provenance import dataset_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and classify the downloaded vibe_coding_630k corpus.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=630000)
    parser.add_argument("--expected-files", type=int, default=141)
    parser.add_argument("--summary-path", type=Path, default=None)
    args = parser.parse_args()

    files = parquet_files(args.input_dir)
    rows = 0
    categories: Counter[str] = Counter()
    input_routes: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    empty_ids = 0
    empty_prompts = 0
    malformed_context_json = 0
    normalization_errors = 0
    global_index = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        missing = set(SOURCE_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise SystemExit(f"{path} missing columns: {sorted(missing)}")
        relative = path.relative_to(args.input_dir).as_posix()
        source_row = 0
        for batch in parquet.iter_batches(columns=list(SOURCE_COLUMNS), batch_size=8192):
            for record in batch.to_pylist():
                rows += 1
                sample_id = str(record.get("id") or "").strip()
                if not sample_id:
                    empty_ids += 1
                elif sample_id in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(sample_id)
                categories[str(record.get("category") or "<missing>")] += 1
                prompt = record.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    empty_prompts += 1
                context_value = record.get("context_json")
                if context_value not in (None, ""):
                    try:
                        parsed_context = json.loads(context_value)
                    except (TypeError, json.JSONDecodeError):
                        malformed_context_json += 1
                    else:
                        if not isinstance(parsed_context, dict):
                            malformed_context_json += 1
                try:
                    sample = normalize_row(record, relative, source_row, global_index)
                except (TypeError, ValueError):
                    normalization_errors += 1
                else:
                    input_routes[sample.conversation_source] += 1
                source_row += 1
                global_index += 1

    identity = dataset_fingerprint(args.input_dir)
    summary = {
        "parquet_files": len(files),
        "rows": rows,
        "unique_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "empty_ids": empty_ids,
        "empty_prompts": empty_prompts,
        "malformed_context_json": malformed_context_json,
        "normalization_errors": normalization_errors,
        "dataset_repo": identity.repo,
        "dataset_revision": identity.revision,
        "dataset_fingerprint": identity.digest,
        "generation_input_routes": dict(sorted(input_routes.items())),
        "category_rows": dict(sorted(categories.items())),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    summary_path = args.summary_path or args.input_dir.parent / "validation_summary.json"
    summary_path.write_text(rendered, encoding="utf-8")
    if (
        len(files) != args.expected_files
        or rows != args.expected_rows
        or duplicate_ids
        or empty_ids
        or empty_prompts
        or malformed_context_json
        or normalization_errors
    ):
        raise SystemExit("dataset validation failed")


if __name__ == "__main__":
    main()
