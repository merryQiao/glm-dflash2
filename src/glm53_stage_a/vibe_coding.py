"""Load and route the normalized ``vibe_coding_630k`` prompt corpus.

The Parquet rows are prompt seeds, not homogeneous chat transcripts.  This
module keeps provenance out of the model prompt and explicitly separates four
routes:

* ``single_turn``: a system/user pair ready for generation;
* ``workspace_task``: a system/user pair that should be paired with a checkout;
* ``conversation_seed``: ordered user turns whose missing assistant turns must
  be generated in sequence;
* ``trajectory_prefix``: an existing structured message/tool prefix.

The complete corpus is small enough for the intended build hosts to load as a
single in-memory Arrow table.  At the current revision it occupies about
10.9 GiB after Parquet decompression.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.dataset as pads


GENERAL_SYSTEM_PROMPT = (
    "You are a capable software engineering assistant. Give a correct, focused "
    "answer. Inspect supplied artifacts before drawing conclusions, do not invent "
    "repository contents or command results, and state important assumptions."
)

WORKSPACE_SYSTEM_PROMPT = (
    "You are a coding agent working in an isolated repository checkout. Use the "
    "provided tools to inspect the repository and implement the requested change. "
    "Do not invent file contents or tool results. Keep changes focused, run the "
    "predefined tests when available, inspect the final diff, and then summarize "
    "the changes and validation."
)


Route = Literal[
    "single_turn", "workspace_task", "conversation_seed", "trajectory_prefix"
]


WORKSPACE_INPUT_KINDS = frozenset(
    {
        "repo_checkout_reference",
        "executable_repo_reference",
        "file_before_change",
    }
)
TRAJECTORY_INPUT_KINDS = frozenset(
    {"agent_trajectory_prefix", "terminal_trajectory_prefix"}
)
CONVERSATION_INPUT_KINDS = frozenset(
    {
        "developer_conversation_user_followup",
        "developer_conversation_with_repo_artifact",
        "pull_request_discussion",
        "pull_request_inline_review",
    }
)


@dataclass
class ModelInput:
    """Canonical, JSON-serializable input produced from one dataset row."""

    id: str
    route: Route
    input_kind: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    remaining_user_turns: list[str] = field(default_factory=list)
    workspace_required: bool = False
    workspace_seed_files: dict[str, str] = field(default_factory=dict)
    workspace_image: str = ""
    workspace_test_command: str = ""
    repo: str = ""
    base_commit: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_parquet_files(root: str | Path) -> list[Path]:
    """Return only formal Parquet shards, excluding manifests and build cache."""

    root = Path(root).expanduser().resolve()
    processed = root / "processed" if (root / "processed").is_dir() else root
    files = sorted(processed.glob("*/*/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no part-*.parquet shards below {processed}")
    return files


def load_vibe_coding_table(root: str | Path) -> pa.Table:
    """Read every shard into RAM and return one Arrow table.

    This deliberately calls :meth:`Dataset.to_table`; it does not create a
    Hugging Face cache and does not leave the data memory-mapped on first use.
    """

    files = [str(path) for path in discover_parquet_files(root)]
    return pads.dataset(files, format="parquet").to_table()


def iter_table_rows(table: pa.Table, *, batch_size: int = 4096) -> Iterator[dict[str, Any]]:
    """Iterate Python dictionaries without converting the whole table at once."""

    for batch in table.to_batches(max_chunksize=batch_size):
        yield from batch.to_pylist()


def _json_value(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _json_object(value: Any) -> dict[str, Any]:
    parsed = _json_value(value, {})
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _decode_json_objects(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    result: list[dict[str, Any]] = []
    for value in values:
        parsed = _json_value(value, None)
        if isinstance(parsed, Mapping):
            result.append(dict(parsed))
    return result


def _tool_arguments(value: Any) -> dict[str, Any]:
    """Return chat-template-safe tool arguments while preserving malformed input."""

    parsed = _json_value(value, None)
    if isinstance(parsed, Mapping):
        return dict(parsed)
    if parsed is None:
        return {}
    # Qwen's chat template iterates argument key/value pairs.  A scalar or
    # malformed source payload cannot remain a string without crashing the
    # template, so retain it explicitly instead of silently discarding it.
    return {"raw_arguments": parsed}


def _clean_tool_calls(values: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for call in _decode_json_objects(values):
        clean = dict(call)
        function = clean.get("function")
        if isinstance(function, Mapping):
            normalized_function = dict(function)
        else:
            normalized_function = {
                "name": str(clean.pop("name", "")),
                "arguments": clean.pop("arguments", {}),
            }
        normalized_function["name"] = str(normalized_function.get("name") or "")
        normalized_function["arguments"] = _tool_arguments(
            normalized_function.get("arguments")
        )
        clean["type"] = str(clean.get("type") or "function")
        clean["function"] = normalized_function
        result.append(clean)
    return result


def _clean_tool_definitions(values: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in _decode_json_objects(values):
        clean = dict(tool)
        function = clean.get("function")
        if isinstance(function, Mapping):
            normalized_function = dict(function)
        elif clean.get("name"):
            normalized_function = {
                "name": str(clean.pop("name")),
                "description": str(clean.pop("description", "")),
                "parameters": clean.pop("input_schema", {"type": "object"}),
            }
        else:
            continue
        parameters = _json_value(normalized_function.get("parameters"), None)
        if not isinstance(parameters, Mapping):
            parameters = {"type": "object", "additionalProperties": True}
        normalized_function["name"] = str(normalized_function.get("name") or "")
        normalized_function["parameters"] = dict(parameters)
        clean["type"] = "function"
        clean["function"] = normalized_function
        result.append(clean)
    return result


def _decode_artifact_body(value: Any) -> str:
    """Decode DevGPT's base64 artifact body, falling back to the original text."""

    text = str(value or "")
    if not text:
        return ""
    try:
        compact = "".join(text.split())
        decoded = base64.b64decode(compact, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return text
    printable = sum(char.isprintable() or char in "\n\r\t" for char in decoded)
    return decoded if decoded and printable / len(decoded) >= 0.95 else text


def _clean_messages(values: Any) -> list[dict[str, Any]]:
    messages = _decode_json_objects(values)
    result: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            continue
        clean = {
            key: value
            for key, value in message.items()
            if value is not None and key not in {"think"}
        }
        clean["role"] = role
        if "content" in clean:
            clean["content"] = str(clean.get("content") or "")
        tool_calls = _clean_tool_calls(clean.get("tool_calls"))
        if tool_calls:
            clean["tool_calls"] = tool_calls
        else:
            clean.pop("tool_calls", None)
        result.append(clean)
    return result


def _structured_prefix(context: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decode direct or nested Open-SWE messages/tools when structurally valid."""

    candidate: Mapping[str, Any] = context
    if "messages" not in candidate and isinstance(candidate.get("unparsed_text"), str):
        nested = _json_value(candidate["unparsed_text"], {})
        if isinstance(nested, Mapping):
            candidate = nested
    return _clean_messages(candidate.get("messages")), _clean_tool_definitions(
        candidate.get("tools")
    )


def _format_file_context(prompt: str, context: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    path = str(context.get("path") or "uploaded_file.txt").lstrip("/")
    old_contents = str(context.get("old_contents") or "")
    if not old_contents:
        return prompt, {}
    user = f"{prompt}\n\nThe checkout initially contains `{path}`. Inspect it with tools before editing."
    return user, {path: old_contents}


def _format_artifact_context(prompt: str, context: Mapping[str, Any]) -> str:
    artifact = context.get("artifact")
    if not isinstance(artifact, Mapping):
        return prompt
    body = _decode_artifact_body(artifact.get("body"))
    if not body:
        return prompt
    artifact_type = str(artifact.get("type") or "repository artifact")
    url = str(artifact.get("url") or "")
    header = f"Supplied {artifact_type}" + (f" ({url})" if url else "")
    return f"{header}:\n\n```text\n{body}\n```\n\nCurrent request:\n{prompt}"


def _format_event_history(events: Any, *, max_chars: int = 48_000) -> str:
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return ""
    blocks: list[str] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        body = str(event.get("body") or "").strip()
        if not body:
            continue
        author = str(event.get("reviewer") or event.get("type") or "participant")
        path = str(event.get("path") or "")
        label = author + (f" on {path}" if path else "")
        blocks.append(f"[{label}]\n{body}")
    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        text = "[Earlier discussion omitted]\n" + text[-max_chars:]
    return text


def _source_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    # These fields are provenance for filtering/validation and never model input.
    return {
        key: row.get(key)
        for key in ("category", "subcategory", "source", "source_id", "language", "license")
        if row.get(key) not in (None, "")
    }


def row_to_model_input(row: Mapping[str, Any]) -> ModelInput:
    """Route one normalized row into an API-ready or rollout-ready structure."""

    row_id = str(row.get("id") or row.get("source_id") or "")
    if not row_id:
        raise ValueError("row has no id or source_id")
    prompt = str(row.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"row {row_id} has an empty prompt")
    kind = str(row.get("input_kind") or "instruction_only")
    context = _json_object(row.get("context_json"))
    repo = str(row.get("repo") or "")
    base_commit = str(row.get("base_commit") or "")
    install_config = context.get("install_config") or {}
    if not isinstance(install_config, Mapping):
        install_config = {}
    common = {
        "id": row_id,
        "input_kind": kind,
        "repo": repo,
        "base_commit": base_commit,
        "workspace_image": str(context.get("image_name") or ""),
        "workspace_test_command": str(install_config.get("test_cmd") or ""),
        "source_metadata": _source_metadata(row),
    }

    if kind in TRAJECTORY_INPUT_KINDS:
        messages, tools = _structured_prefix(context)
        if messages:
            return ModelInput(
                route="trajectory_prefix",
                messages=messages,
                tools=tools,
                workspace_required=True,
                **common,
            )
        return ModelInput(
            route="workspace_task",
            messages=[
                {"role": "system", "content": WORKSPACE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            workspace_required=True,
            warnings=["structured trajectory prefix could not be decoded; using prompt only"],
            **common,
        )

    if kind == "developer_conversation_user_followup":
        prior = context.get("prior_user_prompts") or []
        user_turns = [
            str(item.get("prompt") or "").strip()
            for item in prior
            if isinstance(item, Mapping) and str(item.get("prompt") or "").strip()
        ]
        user_turns.append(prompt)
        return ModelInput(
            route="conversation_seed",
            messages=[
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_turns[0]},
            ],
            remaining_user_turns=user_turns[1:],
            **common,
        )

    if kind == "developer_conversation_with_repo_artifact":
        user = _format_artifact_context(prompt, context)
        return ModelInput(
            route="single_turn",
            messages=[
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            **common,
        )

    if kind in {"pull_request_discussion", "pull_request_inline_review"}:
        events = context.get("conversation_prefix") or context.get("review_prefix") or []
        history = _format_event_history(events)
        user = f"Pull request discussion so far:\n\n{history}\n\nCurrent request:\n{prompt}" if history else prompt
        return ModelInput(
            route="single_turn",
            messages=[
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            **common,
        )

    if kind in WORKSPACE_INPUT_KINDS:
        user, seed_files = _format_file_context(prompt, context) if kind == "file_before_change" else (prompt, {})
        return ModelInput(
            route="workspace_task",
            messages=[
                {"role": "system", "content": WORKSPACE_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            workspace_required=kind != "file_before_change" or bool(seed_files),
            workspace_seed_files=seed_files,
            **common,
        )

    if kind == "terminal_instruction":
        shell = str(context.get("shell") or "")
        platform = str(context.get("platform") or "")
        prefix = "Target environment: " + ", ".join(item for item in (platform, shell) if item)
        user = f"{prefix}.\n\n{prompt}" if prefix != "Target environment: " else prompt
    else:
        # Context for instruction/code/question rows is provenance or source seed;
        # feeding it may leak reference code or irrelevant source metadata.
        user = prompt
    return ModelInput(
        route="single_turn",
        messages=[
            {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        **common,
    )


def iter_model_inputs(table: pa.Table, *, batch_size: int = 4096) -> Iterator[ModelInput]:
    for row in iter_table_rows(table, batch_size=batch_size):
        yield row_to_model_input(row)
