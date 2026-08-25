from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .agent_trajectory import TrajectoryError, render_with_assistant_mask


class TrajectoryTokenError(ValueError):
    """Raised when a completed trajectory cannot be frozen losslessly."""


def _fingerprint(
    tokenizer: Any,
    *,
    tools: Sequence[Mapping[str, Any]],
    chat_template_kwargs: Mapping[str, Any],
) -> str:
    payload = {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "chat_template": str(getattr(tokenizer, "chat_template", "")),
        "tools": list(tools),
        "chat_template_kwargs": dict(chat_template_kwargs),
    }
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def freeze_trajectory_tokens(
    tokenizer: Any,
    trajectory: Mapping[str, Any],
    *,
    chat_template_kwargs: Mapping[str, Any],
    max_sequence_tokens: int | None = None,
) -> dict[str, Any]:
    """Render one immutable trajectory and create a DFlash token-position mask.

    The mask marks target assistant-token positions. It is not shifted to the
    preceding AR predictor position, because SpecForge DFlash samples anchors
    and labels at the target token indices themselves.
    """

    messages = trajectory.get("messages")
    tools = trajectory.get("tools") or []
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise TrajectoryTokenError("trajectory messages must be a sequence")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        raise TrajectoryTokenError("trajectory tools must be an ordered sequence")
    start = trajectory.get("generation_start_message_index", 0)
    if not isinstance(start, int):
        raise TrajectoryTokenError("generation_start_message_index must be an integer")
    try:
        input_ids, loss_mask = render_with_assistant_mask(
            tokenizer,
            messages,
            tools,
            assistant_start_index=start,
            chat_template_kwargs=chat_template_kwargs,
        )
    except (TrajectoryError, TypeError, ValueError) as exc:
        raise TrajectoryTokenError(str(exc)) from exc

    if not input_ids:
        raise TrajectoryTokenError("rendered trajectory is empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in input_ids):
        raise TrajectoryTokenError("rendered input_ids must contain integers")
    if len(loss_mask) != len(input_ids) or any(value not in (0, 1) for value in loss_mask):
        raise TrajectoryTokenError("loss_mask must be binary and match input_ids")
    if max_sequence_tokens is not None:
        if max_sequence_tokens < 1:
            raise TrajectoryTokenError("max_sequence_tokens must be positive")
        if len(input_ids) > max_sequence_tokens:
            raise TrajectoryTokenError(
                f"trajectory has {len(input_ids)} tokens and exceeds "
                f"max_sequence_tokens={max_sequence_tokens}"
            )

    # The final target token has no complete successor/block continuation in
    # the stored record. Match SpecForge's preprocessing contract.
    loss_mask[-1] = 0
    supervised = int(sum(loss_mask))
    if supervised == 0:
        raise TrajectoryTokenError("trajectory has no supervised assistant tokens")

    kwargs = dict(chat_template_kwargs)
    ordered_tools = [dict(tool) for tool in tools]
    generated_indices = [
        index
        for index, message in enumerate(messages)
        if index >= start and isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    response_metadata = trajectory.get("response_metadata") or []
    if response_metadata and len(response_metadata) != len(generated_indices):
        raise TrajectoryTokenError(
            "response metadata count differs from generated assistant turns"
        )

    def render(prefix: Sequence[Mapping[str, Any]], *, generation: bool) -> list[int]:
        value = tokenizer.apply_chat_template(
            list(prefix),
            tools=ordered_tools,
            tokenize=True,
            add_generation_prompt=generation,
            **kwargs,
        )
        if isinstance(value, Mapping):
            value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            value = value[0]
        return [int(token_id) for token_id in value]

    round_checks: list[str] = []
    for ordinal, message_index in enumerate(generated_indices):
        metadata = response_metadata[ordinal] if response_metadata else {}
        server_prompt = metadata.get("prompt_token_ids")
        server_response = metadata.get("response_token_ids")
        if server_prompt is None and server_response is None:
            round_checks.append("unavailable")
            continue
        prompt_ids = render(messages[:message_index], generation=True)
        through_ids = render(messages[: message_index + 1], generation=False)
        response_ids = through_ids[len(prompt_ids) :]
        if server_prompt is not None and [int(value) for value in server_prompt] != prompt_ids:
            raise TrajectoryTokenError(
                f"round {ordinal} server prompt token IDs differ from frozen replay"
            )
        if server_response is not None and [int(value) for value in server_response] != response_ids:
            raise TrajectoryTokenError(
                f"round {ordinal} server response token IDs differ from frozen replay"
            )
        if server_prompt is not None and server_response is not None:
            round_checks.append("matched")
        elif server_prompt is not None:
            round_checks.append("prompt_matched")
        else:
            round_checks.append("response_matched")
    return {
        "input_ids": [int(token_id) for token_id in input_ids],
        "loss_mask": [int(value) for value in loss_mask],
        "token_contract": {
            "schema_version": 1,
            "mask_semantics": "dflash_target_token",
            "token_count": len(input_ids),
            "supervised_tokens": supervised,
            "generation_start_message_index": start,
            "chat_template_kwargs": kwargs,
            "round_token_checks": round_checks,
            "render_fingerprint": _fingerprint(
                tokenizer, tools=ordered_tools, chat_template_kwargs=kwargs
            ),
        },
    }
