"""Canonical raw conditions for Qwen3-Omni Thinker speculative training."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator

import pyarrow as pa


WHITESPACE = re.compile(r"\s+")
MEDIA_PLACEHOLDER = re.compile(
    r"<(?:image|video|audio)>|\b(?:attached|uploaded)\s+(?:image|photo|video|audio|file)\b",
    re.IGNORECASE,
)
PROMPT_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b",
    re.IGNORECASE,
)
PII_OR_SECRET = re.compile(
    r"(?:\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b)"
    r"|(?:\b(?:\d[ -]*?){13,19}\b)"
    r"|(?:\b(?:sk|rk|ghp|github_pat)_[A-Za-z0-9_-]{16,}\b)",
    re.IGNORECASE,
)

ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}
LANGUAGE_NAMES = {
    "arabic": "ar",
    "chinese": "zh",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "norwegian": "no",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "spanish": "es",
    "swedish": "sv",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "vietnamese": "vi",
}


TRAIN_CONDITION_SCHEMA = pa.schema(
    [
        ("condition_id", pa.string()),
        ("split", pa.string()),
        ("source", pa.string()),
        ("source_subset", pa.string()),
        ("source_split", pa.string()),
        ("source_row_id", pa.string()),
        ("source_revision", pa.string()),
        ("source_url", pa.string()),
        ("license", pa.string()),
        ("modality", pa.string()),
        ("language", pa.string()),
        ("task", pa.string()),
        ("conversation_id", pa.string()),
        ("turn_index", pa.int32()),
        ("messages_json", pa.large_string()),
        ("tools_json", pa.large_string()),
        ("media_json", pa.large_string()),
        ("reference_json", pa.large_string()),
        ("selection_priority", pa.string()),
        ("total_characters", pa.int32()),
    ]
)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = "".join(
        character
        for character in text
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
    )
    return WHITESPACE.sub(" ", text).strip()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hex(*values: Any, digest_size: int = 32) -> str:
    payload = "\x1f".join(canonical_json(value) for value in values).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()


def stable_int(*values: Any, modulus: int = 2**63 - 1) -> int:
    """Map structured values to a deterministic non-negative integer."""

    return int(stable_hex(*values, digest_size=8), 16) % modulus


def normalize_language(value: Any) -> str:
    language = normalize_text(value).casefold().replace("_", "-")
    if language in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[language]
    prefix = language.split("-", 1)[0]
    return prefix if 2 <= len(prefix) <= 3 else "und"


def normalize_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for message in messages:
        role = ROLE_ALIASES.get(str(message.get("role", message.get("from", ""))).lower())
        if role is None:
            continue
        content = normalize_text(message.get("content", message.get("value", "")))
        if not content:
            continue
        if result and result[-1]["role"] == role and role != "tool":
            result[-1]["content"] += "\n\n" + content
        else:
            result.append({"role": role, "content": content})
    return result


def task_from_text(text: str, fallback: str = "general_instruction") -> str:
    lowered = text.casefold()
    if any(token in lowered for token in ("function", "api", "tool call", "json schema")):
        return "tool_use"
    if any(token in lowered for token in ("python", "javascript", "c++", " sql ", "debug", "code")):
        return "code"
    if any(token in lowered for token in ("equation", "calculate", "theorem", "proof", "geometry")):
        return "math"
    if any(token in lowered for token in ("summarize", "summary")):
        return "summarization"
    if any(token in lowered for token in ("rewrite", "rephrase")):
        return "rewriting"
    return fallback


def basic_text_quality(
    messages: list[dict[str, str]],
    *,
    min_user_characters: int,
    max_user_characters: int,
    max_total_characters: int,
    reject_missing_media: bool = False,
    reject_prompt_injection: bool = False,
    reject_pii: bool = False,
) -> str | None:
    if not messages or messages[-1]["role"] not in {"user", "tool"}:
        return "invalid_terminal_role"
    user_messages = [message["content"] for message in messages if message["role"] == "user"]
    if not user_messages:
        return "missing_user"
    last_user = user_messages[-1]
    if not min_user_characters <= len(last_user) <= max_user_characters:
        return "user_length"
    if sum(len(message["content"]) for message in messages) > max_total_characters:
        return "context_length"
    joined = "\n".join(message["content"] for message in messages)
    if reject_missing_media and MEDIA_PLACEHOLDER.search(joined):
        return "missing_media"
    if reject_prompt_injection and PROMPT_INJECTION.search(last_user):
        return "prompt_injection"
    if reject_pii and PII_OR_SECRET.search(joined):
        return "pii_or_secret"
    return None


@dataclass(frozen=True)
class RawTrainingCondition:
    condition_id: str
    split: str
    source: str
    source_subset: str
    source_split: str
    source_row_id: str
    source_revision: str
    source_url: str
    license: str
    modality: str
    language: str
    task: str
    conversation_id: str
    turn_index: int
    messages_json: str
    tools_json: str
    media_json: str
    reference_json: str
    selection_priority: str
    total_characters: int

    @classmethod
    def create(
        cls,
        *,
        source: dict[str, Any],
        source_subset: str,
        source_row_id: str,
        conversation_id: str,
        turn_index: int,
        messages: Iterable[dict[str, Any]],
        language: str,
        reference: Any = "",
        tools: Any = None,
        media: Any = None,
        modality: str | None = None,
        task: str | None = None,
        split: str = "train",
    ) -> "RawTrainingCondition":
        canonical_messages = normalize_messages(messages)
        canonical_tools = tools or []
        canonical_media = media or []
        if not canonical_messages or canonical_messages[-1]["role"] not in {"user", "tool"}:
            raise ValueError("a training condition must end with user or tool input")
        media_identity = [
            {
                "type": str(item.get("type", "")).lower(),
                "sha256": str(item.get("sha256", "")),
                "has_audio": bool(item.get("has_audio", False)),
            }
            for item in canonical_media
        ]
        condition_id = stable_hex(
            "thinker-speculative-condition-v1",
            canonical_messages,
            canonical_tools,
            media_identity,
        )
        priority = stable_hex(
            "thinker-speculative-selection-v1",
            int(source["selection_seed"]),
            condition_id,
            digest_size=16,
        )
        last_user = next(
            message["content"]
            for message in reversed(canonical_messages)
            if message["role"] == "user"
        )
        media_types = {str(item.get("type", "")).lower() for item in canonical_media}
        media_types.discard("")
        inferred_modality = "+".join(sorted(media_types)) if media_types else "text"
        return cls(
            condition_id=condition_id,
            split=split,
            source=str(source["name"]),
            source_subset=source_subset,
            source_split=str(source["split"]),
            source_row_id=str(source_row_id),
            source_revision=str(source["resolved_revision"]),
            source_url=str(source["source_url"]),
            license=str(source["license"]),
            modality=modality or inferred_modality,
            language=normalize_language(language),
            task=task or task_from_text(last_user, str(source.get("task", "general_instruction"))),
            conversation_id=str(conversation_id),
            turn_index=int(turn_index),
            messages_json=canonical_json(canonical_messages),
            tools_json=canonical_json(canonical_tools),
            media_json=canonical_json(canonical_media),
            reference_json=canonical_json({"assistant": reference} if reference else {}),
            selection_priority=priority,
            total_characters=sum(len(message["content"]) for message in canonical_messages),
        )

    @property
    def messages(self) -> list[dict[str, str]]:
        return json.loads(self.messages_json)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chat_conditions(
    row: dict[str, Any],
    source: dict[str, Any],
    *,
    row_index: int,
    messages_field: str,
    source_subset: str,
    conversation_id: str,
    language: str,
    system_prompt: str = "",
) -> Iterator[RawTrainingCondition]:
    messages = normalize_messages(row.get(messages_field) or [])
    if system_prompt:
        messages.insert(0, {"role": "system", "content": normalize_text(system_prompt)})
    user_indices = [index for index, message in enumerate(messages) if message["role"] == "user"]
    limit = int(source.get("max_conditions_per_conversation", 2))
    for index in user_indices[-limit:]:
        prefix = messages[: index + 1]
        reference = ""
        if index + 1 < len(messages) and messages[index + 1]["role"] == "assistant":
            reference = messages[index + 1]["content"]
        if source.get("require_reference") and not reference:
            continue
        reason = basic_text_quality(
            prefix,
            min_user_characters=int(source["min_user_characters"]),
            max_user_characters=int(source["max_user_characters"]),
            max_total_characters=int(source["max_total_characters"]),
            reject_missing_media=bool(source.get("reject_missing_media", False)),
            reject_prompt_injection=bool(source.get("reject_prompt_injection", False)),
            reject_pii=bool(source.get("reject_pii", False)),
        )
        if reason is not None:
            continue
        yield RawTrainingCondition.create(
            source=source,
            source_subset=source_subset,
            source_row_id=f"{conversation_id}:{index}",
            conversation_id=conversation_id,
            turn_index=index,
            messages=prefix,
            language=language,
            reference=reference,
        )


def extract_json_array(text: str) -> list[Any]:
    marker = text.find("[")
    if marker < 0:
        return []
    try:
        value, _ = json.JSONDecoder().raw_decode(text[marker:])
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def extract_tools_and_clean_system(text: str) -> tuple[list[Any], str]:
    """Separate ToolACE's embedded JSON schemas from the system instruction."""

    marker = text.find("[")
    if marker < 0:
        return [], normalize_text(text)
    try:
        value, end = json.JSONDecoder().raw_decode(text[marker:])
    except json.JSONDecodeError:
        return [], normalize_text(text)
    if not isinstance(value, list):
        return [], normalize_text(text)
    # Qwen's chat template serializes ``tools`` into the system turn. Keeping
    # the original embedded copy would expose every schema twice.
    cleaned = normalize_text(text[:marker] + " " + text[marker + end :])
    return value, cleaned


def toolace_conditions(
    row: dict[str, Any], source: dict[str, Any], row_index: int
) -> Iterator[RawTrainingCondition]:
    tools, system = extract_tools_and_clean_system(str(row.get("system", "")))
    messages = normalize_messages(row.get("conversations") or [])
    if system:
        messages.insert(0, {"role": "system", "content": system})
    candidates: list[tuple[int, str]] = []
    for index, message in enumerate(messages[:-1]):
        if message["role"] not in {"user", "tool"} or messages[index + 1]["role"] != "assistant":
            continue
        reference = messages[index + 1]["content"]
        candidates.append((index, reference))
    tool_calls = [item for item in candidates if item[1].lstrip().startswith("[")]
    final_answers = [item for item in candidates if not item[1].lstrip().startswith("[")]
    chosen = (tool_calls[:1] + final_answers[:1])[: int(source.get("max_conditions_per_conversation", 2))]
    for index, reference in chosen:
        prefix = messages[: index + 1]
        if basic_text_quality(
            prefix,
            min_user_characters=int(source["min_user_characters"]),
            max_user_characters=int(source["max_user_characters"]),
            max_total_characters=int(source["max_total_characters"]),
        ) is not None:
            continue
        yield RawTrainingCondition.create(
            source=source,
            source_subset="ToolACE",
            source_row_id=f"{row_index}:{index}",
            conversation_id=str(row_index),
            turn_index=index,
            messages=prefix,
            language="en",
            reference=reference,
            tools=tools,
            task="tool_use",
        )


def turnstile_conditions(
    row: dict[str, Any],
    source: dict[str, Any],
    row_index: int,
    api_definitions: dict[str, Any],
) -> Iterator[RawTrainingCondition]:
    messages: list[dict[str, str]] = []
    candidates: list[tuple[int, str]] = []
    for event in row.get("interaction") or []:
        if not isinstance(event, dict) or len(event) != 1:
            continue
        kind, value = next(iter(event.items()))
        if kind == "THINKING":
            continue
        if kind == "SYSTEM":
            messages.append({"role": "system", "content": normalize_text(value)})
        elif kind == "USER":
            messages.append({"role": "user", "content": normalize_text(value)})
        elif kind in {"API_CALL", "ASST"}:
            if messages and messages[-1]["role"] in {"user", "tool"}:
                candidates.append((len(messages) - 1, normalize_text(value)))
            messages.append({"role": "assistant", "content": normalize_text(value)})
        elif kind == "API_OBS":
            messages.append({"role": "tool", "content": canonical_json(value)})
    tool_names = [*(row.get("api_names") or []), *(row.get("distractors") or [])]
    tools = [api_definitions[name] for name in tool_names if name in api_definitions]
    call_targets = [item for item in candidates if "(" in item[1]]
    answer_targets = [item for item in candidates if "(" not in item[1]]
    chosen = (call_targets[:1] + answer_targets[:1])[: int(source.get("max_conditions_per_conversation", 2))]
    for index, reference in chosen:
        prefix = messages[: index + 1]
        if basic_text_quality(
            prefix,
            min_user_characters=int(source["min_user_characters"]),
            max_user_characters=int(source["max_user_characters"]),
            max_total_characters=int(source["max_total_characters"]),
        ) is not None:
            continue
        yield RawTrainingCondition.create(
            source=source,
            source_subset=str(row.get("interaction_template_name", "unknown")),
            source_row_id=f"{row_index}:{index}",
            conversation_id=str(row_index),
            turn_index=index,
            messages=prefix,
            language="en",
            reference=reference,
            tools=tools,
            task="tool_use",
        )
