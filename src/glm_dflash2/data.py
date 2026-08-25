from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


SOURCE_COLUMNS = (
    "id",
    "category",
    "category_zh",
    "subcategory",
    "source",
    "source_id",
    "prompt",
    "context_json",
    "repo",
    "base_commit",
    "language",
    "license",
    "input_kind",
    "dedup_key",
    "metadata_json",
)

CHAT_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})


@dataclass(frozen=True)
class SourceSample:
    sample_id: str
    global_index: int
    source_path: str
    source_row: int
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    conversation_source: str
    original: dict[str, Any]


def _json_object(value: Any, *, strict: bool = True) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value, strict=strict)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _valid_messages(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or not value:
        return None
    messages: list[dict[str, Any]] = []
    for item in value:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("role"), str)
            or item["role"] not in CHAT_ROLES
        ):
            return None
        content = item.get("content")
        if content is not None and not isinstance(content, (str, list)):
            return None
        messages.append({key: val for key, val in item.items() if val is not None})
    return messages


def _valid_tools(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or not value:
        return None
    tools: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                return None
        if not isinstance(item, dict):
            return None
        tools.append(item)
    return tools


def _serialized_messages(context: dict[str, Any] | None) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None, str | None]:
    if not context:
        return None, None, None
    raw_messages = context.get("messages")
    messages = _valid_messages(raw_messages)
    if messages:
        return messages, _valid_tools(context.get("tools")), "context_json.messages"
    if raw_messages not in (None, [], ""):
        return None, None, "context_json.invalid_messages_fallback_to_prompt"

    unparsed_text = context.get("unparsed_text")
    if isinstance(unparsed_text, str) and "...[TRUNCATED]..." in unparsed_text:
        return None, None, "context_json.truncated_messages_fallback_to_prompt"
    # The Open-SWE source stores raw control characters inside this nested JSON
    # string. Python's permissive mode recovers the intended messages without
    # altering the outer, strictly parsed context_json object.
    unparsed = _json_object(unparsed_text, strict=False)
    messages = _valid_messages(unparsed.get("messages")) if unparsed else None
    if messages:
        return messages, _valid_tools(unparsed.get("tools")), "context_json.unparsed_text.messages"
    if unparsed and unparsed.get("messages") not in (None, [], ""):
        return None, None, "context_json.invalid_messages_fallback_to_prompt"
    return None, None, None


def _format_artifact(artifact: Any) -> str | None:
    if not isinstance(artifact, dict):
        return None
    selected = []
    for key in ("type", "url", "title", "body", "sha", "commit_sha"):
        value = artifact.get(key)
        if value not in (None, ""):
            selected.append(f"{key}: {value}")
    return "\n".join(selected) if selected else None


def _format_event_prefix(events: Any) -> str | None:
    if not isinstance(events, list) or not events:
        return None
    rendered = []
    for index, event in enumerate(events, 1):
        if not isinstance(event, dict):
            continue
        fields = []
        for key in (
            "type",
            "timestamp",
            "reviewer",
            "reviewer_type",
            "state",
            "author",
            "title",
            "path",
            "line",
            "side",
            "body",
            "diff_hunk",
        ):
            value = event.get(key)
            if value not in (None, ""):
                fields.append(f"{key}: {value}")
        replies = event.get("thread_replies")
        if isinstance(replies, list) and replies:
            rendered_replies = []
            for reply_index, reply in enumerate(replies, 1):
                if not isinstance(reply, dict):
                    continue
                reply_fields = [
                    f"{key}: {value}"
                    for key in ("author", "reviewer", "reviewer_type", "state", "timestamp", "body")
                    if (value := reply.get(key)) not in (None, "")
                ]
                if reply_fields:
                    rendered_replies.append(
                        f"[thread reply {reply_index}]\n" + "\n".join(reply_fields)
                    )
            if rendered_replies:
                fields.append("thread_replies:\n" + "\n".join(rendered_replies))
        if fields:
            rendered.append(f"[event {index}]\n" + "\n".join(fields))
    return "\n\n".join(rendered) if rendered else None


def _static_context_prompt(prompt: str, context: dict[str, Any] | None) -> tuple[str, str]:
    if not context:
        return prompt, "prompt"

    unparsed_text = context.get("unparsed_text")
    if isinstance(unparsed_text, str) and "...[TRUNCATED]..." in unparsed_text:
        return prompt, "context_json.truncated_messages_fallback_to_prompt"

    prior_prompts = context.get("prior_user_prompts")
    if isinstance(prior_prompts, list) and prior_prompts:
        prior = []
        for item in prior_prompts:
            if isinstance(item, dict) and isinstance(item.get("prompt"), str) and item["prompt"].strip():
                prior.append(item["prompt"])
        if prior:
            sections = [
                "<incomplete_conversation_context>",
                "Only earlier user requests are available; assistant replies are unavailable.",
            ]
            artifact = _format_artifact(context.get("artifact"))
            if artifact:
                sections.extend(["<artifact>", artifact, "</artifact>"])
            for index, value in enumerate(prior, 1):
                sections.extend([f"<prior_user_request index=\"{index}\">", value, "</prior_user_request>"])
            sections.extend(["</incomplete_conversation_context>", "<current_user_request>", prompt, "</current_user_request>"])
            return "\n".join(sections), "context_json.incomplete_user_history+prompt"

    old_contents = context.get("old_contents")
    if isinstance(old_contents, str):
        path = context.get("path") or "<unknown>"
        content = (
            f"<file_before_change path=\"{path}\">\n{old_contents}\n</file_before_change>\n"
            f"<task>\n{prompt}\n</task>"
        )
        return content, "context_json.static_context+prompt"

    for prefix_key in ("conversation_prefix", "review_prefix"):
        prefix = _format_event_prefix(context.get(prefix_key))
        if prefix:
            metadata = []
            for key in ("task", "repo", "pr_url", "pr_title", "base_commit"):
                value = context.get(key)
                if value not in (None, ""):
                    metadata.append(f"{key}: {value}")
            extra = "\n<context_metadata>\n" + "\n".join(metadata) + "\n</context_metadata>" if metadata else ""
            diff = context.get("diff_context")
            if isinstance(diff, dict):
                extra += "\n<diff_context>\n" + json.dumps(diff, ensure_ascii=False, indent=2) + "\n</diff_context>"
            content = f"<{prefix_key}>\n{prefix}\n</{prefix_key}>{extra}\n<current_request>\n{prompt}\n</current_request>"
            return content, "context_json.static_context+prompt"

    metadata = []
    for key in ("task", "repo", "pr_url", "pr_title", "base_commit"):
        value = context.get(key)
        if value not in (None, ""):
            metadata.append(f"{key}: {value}")
    if metadata:
        content = (
            "<repository_context>\n"
            + "\n".join(metadata)
            + "\n</repository_context>\n<current_request>\n"
            + prompt
            + "\n</current_request>"
        )
        return content, "context_json.static_context+prompt"

    return prompt, "prompt"


def normalize_row(
    row: dict[str, Any], source_path: str, source_row: int, global_index: int
) -> SourceSample:
    sample_id = str(row.get("id") or "").strip()
    if not sample_id:
        raise ValueError(f"row {global_index} has an empty id")

    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"row {global_index} ({sample_id}) has an empty prompt")
    context = _json_object(row.get("context_json"))
    messages, tools, conversation_source = _serialized_messages(context)
    if not messages:
        if conversation_source in {
            "context_json.invalid_messages_fallback_to_prompt",
            "context_json.truncated_messages_fallback_to_prompt",
        }:
            content = prompt
        else:
            content, conversation_source = _static_context_prompt(prompt, context)
        messages = [{"role": "user", "content": content}]
        tools = None

    original = {name: row.get(name) for name in SOURCE_COLUMNS}
    return SourceSample(
        sample_id=sample_id,
        global_index=global_index,
        source_path=source_path,
        source_row=source_row,
        messages=messages,
        tools=tools,
        conversation_source=conversation_source,
        original=original,
    )


def source_belongs_to_shard(global_index: int, shard_index: int, shard_count: int) -> bool:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return global_index % shard_count == shard_index


def parquet_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files found under {input_dir}")
    return files


def iter_source_records(
    input_dir: Path,
    *,
    start_index: int = 0,
    end_index: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    read_batch_size: int = 1024,
) -> Iterator[SourceSample]:
    import pyarrow.parquet as pq

    if start_index < 0:
        raise ValueError("start_index cannot be negative")
    global_index = 0
    for path in parquet_files(input_dir):
        relative_path = path.relative_to(input_dir).as_posix()
        parquet = pq.ParquetFile(path)
        source_row = 0
        available = set(parquet.schema_arrow.names)
        missing = set(SOURCE_COLUMNS) - available
        if missing:
            raise ValueError(f"{relative_path} is missing columns: {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=read_batch_size, columns=list(SOURCE_COLUMNS)):
            for row in batch.to_pylist():
                current = global_index
                global_index += 1
                local_row = source_row
                source_row += 1
                if current < start_index:
                    continue
                if end_index is not None and current >= end_index:
                    return
                if not source_belongs_to_shard(current, shard_index, shard_count):
                    continue
                yield normalize_row(row, relative_path, local_row, current)
