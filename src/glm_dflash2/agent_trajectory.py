"""Model-agnostic coding-agent rollout primitives.

The model proposes assistant messages and tool calls. Tool observations are
always produced by :class:`WorkspaceToolExecutor`; they are never synthesized
by the model. The module speaks the OpenAI chat-completions wire format used by
the SGLang endpoint, allowing a small debug endpoint and the final target
endpoint to share the same rollout logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import requests


DEFAULT_SYSTEM_PROMPT = """You are a coding agent working in a repository checkout.
Use the supplied tools to inspect the repository and make the requested change.
Never invent file contents or command results. Inspect before editing, keep the
change focused, run the supplied test command when available, and finish with a
concise summary of the changes and validation performed."""


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files below a path in the repository checkout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "max_depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 6,
                        "default": 2,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a bounded line range from a UTF-8 repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1, "default": 1},
                    "end_line": {"type": "integer", "minimum": 1, "default": 400},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a literal string in repository text files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "glob": {"type": "string", "default": ""},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff to files inside the repository.",
            "parameters": {
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show the current repository diff.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the task's predefined test command in the checkout.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]

WORKSPACE_TOOL_NAMES = frozenset(
    definition["function"]["name"] for definition in TOOL_DEFINITIONS
)


def tool_definitions_for(names: Sequence[str]) -> list[dict[str, Any]]:
    """Return only tool schemas actually backed by the active executor."""

    selected = frozenset(str(name) for name in names)
    return [
        definition
        for definition in TOOL_DEFINITIONS
        if definition["function"]["name"] in selected
    ]


class TrajectoryError(RuntimeError):
    """Raised when a rollout violates the trajectory contract."""


class ToolExecutor(Protocol):
    tool_names: frozenset[str]

    def execute(self, call: Mapping[str, Any]) -> dict[str, Any]: ...


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _stable_int(*parts: Any) -> int:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def normalize_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return strict OpenAI-style tool calls with decoded argument objects."""

    result: list[dict[str, Any]] = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function") or {}
        name = str(function.get("name") or "").strip()
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise TrajectoryError(f"invalid JSON arguments for tool {name!r}: {exc}") from exc
        if not isinstance(arguments, Mapping):
            raise TrajectoryError(f"tool {name!r} arguments must decode to an object")
        if not name:
            raise TrajectoryError("tool call has an empty function name")
        result.append(
            {
                "id": str(raw.get("id") or f"call_{index}_{uuid.uuid4().hex[:12]}"),
                "type": "function",
                "function": {"name": name, "arguments": dict(arguments)},
            }
        )
    return result


def openai_wire_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert canonical stored messages back to the OpenAI request schema."""

    wire: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if item.get("role") == "assistant" and item.get("tool_calls"):
            calls = []
            for call in normalize_tool_calls(item):
                call["function"]["arguments"] = _compact_json(call["function"]["arguments"])
                calls.append(call)
            item["tool_calls"] = calls
        wire.append(item)
    return wire


@dataclass(frozen=True)
class ChatCompletionConfig:
    endpoint: str
    model: str
    api_key: str = "EMPTY"
    timeout_seconds: float = 3600.0
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 32768
    reasoning_effort: str | None = "medium"
    enable_thinking: bool = True
    preserve_thinking: bool = True
    top_k: int = -1
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    chat_template_kwargs: Mapping[str, Any] | None = None
    return_token_ids: bool = False


class OpenAIChatClient:
    def __init__(self, config: ChatCompletionConfig) -> None:
        self.config = config
        self.base_url = config.endpoint.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }
        )

    def server_models(self) -> list[str]:
        response = self.session.get(
            f"{self.base_url}/v1/models", timeout=self.config.timeout_seconds
        )
        response.raise_for_status()
        return [str(item["id"]) for item in response.json().get("data", [])]

    def assert_model_available(self) -> None:
        models = self.server_models()
        if self.config.model not in models:
            raise TrajectoryError(
                f"requested model {self.config.model!r} is not exposed by server: {models}"
            )

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        template_kwargs = (
            dict(self.config.chat_template_kwargs)
            if self.config.chat_template_kwargs is not None
            else {
                "enable_thinking": self.config.enable_thinking,
                "preserve_thinking": self.config.preserve_thinking,
            }
        )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": openai_wire_messages(messages),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "presence_penalty": self.config.presence_penalty,
            "repetition_penalty": self.config.repetition_penalty,
            "max_tokens": self.config.max_tokens,
            "chat_template_kwargs": template_kwargs,
        }
        if self.config.return_token_ids:
            payload["return_prompt_token_ids"] = True
            payload["return_token_ids"] = True
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        if self.config.reasoning_effort:
            # Qwen3.8 names its highest template profile ``xhigh`` while the
            # OpenAI-compatible SGLang schema names the same API level ``high``.
            # Preserve xhigh inside the chat template and use the wire spelling
            # accepted by SGLang/OpenAI request validation.
            payload["chat_template_kwargs"]["reasoning_effort"] = self.config.reasoning_effort
            payload["reasoning_effort"] = (
                "high" if self.config.reasoning_effort == "xhigh" else self.config.reasoning_effort
            )
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        if not response.ok:
            raise TrajectoryError(
                f"chat completion failed ({response.status_code}): {response.text[:2000]}"
            )
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise TrajectoryError(f"chat completion returned no choices: {body}")
        message = choices[0].get("message") or {}
        canonical: dict[str, Any] = {
            "role": "assistant",
            "content": str(message.get("content") or ""),
        }
        reasoning = message.get("reasoning_content")
        if reasoning is not None:
            canonical["reasoning_content"] = str(reasoning)
        calls = normalize_tool_calls(message)
        if calls:
            canonical["tool_calls"] = calls
        canonical["_response_metadata"] = {
            "finish_reason": choices[0].get("finish_reason"),
            "usage": body.get("usage"),
            "response_id": body.get("id"),
            "prompt_token_ids": choices[0].get("prompt_token_ids"),
            "response_token_ids": choices[0].get("response_token_ids"),
        }
        return canonical


class WorkspaceToolExecutor:
    """Bounded file tools for one isolated repository checkout.

    ``run_tests`` never accepts a model-generated command.  It executes only
    ``test_argv`` supplied by the dataset materializer.  Unknown repository
    code should still be run inside a container/VM in production; host test
    execution is opt-in and exists for controlled local fixtures only.
    """

    tool_names = WORKSPACE_TOOL_NAMES

    def __init__(
        self,
        root: str | Path,
        *,
        test_argv: Sequence[str] | None = None,
        allow_host_tests: bool = False,
        timeout_seconds: float = 300.0,
        max_output_chars: int = 32_768,
    ) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise TrajectoryError(f"workspace is not a directory: {self.root}")
        self.test_argv = tuple(str(item) for item in (test_argv or ()))
        self.allow_host_tests = allow_host_tests
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.tool_names = (
            WORKSPACE_TOOL_NAMES
            if self.test_argv and self.allow_host_tests
            else WORKSPACE_TOOL_NAMES - {"run_tests"}
        )

    def _path(self, value: Any, *, must_exist: bool = True) -> Path:
        raw = str(value or ".")
        candidate = Path(raw)
        if candidate.is_absolute():
            raise TrajectoryError("absolute paths are not allowed")
        resolved = (self.root / candidate).resolve(strict=must_exist)
        if not resolved.is_relative_to(self.root):
            raise TrajectoryError("path escapes the repository checkout")
        return resolved

    def _relative(self, path: Path) -> str:
        value = path.relative_to(self.root).as_posix()
        return value or "."

    def _cap(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.max_output_chars:
            return text, False
        half = self.max_output_chars // 2
        return text[:half] + "\n...[OUTPUT TRUNCATED]...\n" + text[-half:], True

    def execute(self, call: Mapping[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments") or {}
        try:
            if name == "list_files":
                result = self._list_files(arguments)
            elif name == "read_file":
                result = self._read_file(arguments)
            elif name == "search_code":
                result = self._search_code(arguments)
            elif name == "apply_patch":
                result = self._apply_patch(arguments)
            elif name == "git_diff":
                result = self._git_diff()
            elif name == "run_tests":
                result = self._run_tests()
            else:
                raise TrajectoryError(f"unknown tool: {name!r}")
            result.setdefault("ok", True)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return result

    def _list_files(self, args: Mapping[str, Any]) -> dict[str, Any]:
        base = self._path(args.get("path", "."))
        depth = max(1, min(6, int(args.get("max_depth", 2))))
        if not base.is_dir():
            raise TrajectoryError("list_files path is not a directory")
        files: list[str] = []
        base_parts = len(base.parts)
        for path in sorted(base.rglob("*")):
            if len(path.parts) - base_parts > depth:
                continue
            if ".git" in path.relative_to(self.root).parts:
                continue
            suffix = "/" if path.is_dir() else ""
            files.append(self._relative(path) + suffix)
            if len(files) >= 2000:
                break
        text, truncated = self._cap("\n".join(files))
        return {"path": self._relative(base), "entries": text, "truncated": truncated}

    def _read_file(self, args: Mapping[str, Any]) -> dict[str, Any]:
        path = self._path(args.get("path"))
        if not path.is_file():
            raise TrajectoryError("read_file path is not a regular file")
        start = max(1, int(args.get("start_line", 1)))
        end = max(start, min(start + 1999, int(args.get("end_line", 400))))
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            raise TrajectoryError("binary files are not supported")
        lines = raw.decode("utf-8", errors="replace").splitlines()
        rendered = "\n".join(
            f"{index}: {lines[index - 1]}"
            for index in range(start, min(end, len(lines)) + 1)
        )
        text, truncated = self._cap(rendered)
        return {
            "path": self._relative(path),
            "start_line": start,
            "end_line": min(end, len(lines)),
            "content": text,
            "truncated": truncated,
        }

    def _search_code(self, args: Mapping[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        if not query or len(query) > 1000:
            raise TrajectoryError("query must contain 1-1000 characters")
        base = self._path(args.get("path", "."))
        command = [
            "rg",
            "--fixed-strings",
            "--line-number",
            "--color=never",
            "--max-count=200",
        ]
        glob = str(args.get("glob") or "")
        if glob:
            command.extend(["--glob", glob])
        command.extend(["--", query, str(base)])
        proc = subprocess.run(
            command,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=min(60.0, self.timeout_seconds),
            check=False,
        )
        output = proc.stdout if proc.returncode in (0, 1) else proc.stderr
        text, truncated = self._cap(output)
        return {
            "query": query,
            "path": self._relative(base),
            "matches": text,
            "returncode": proc.returncode,
            "truncated": truncated,
        }

    def _patch_paths(self, patch: str) -> list[Path]:
        paths: list[Path] = []
        for match in re.finditer(r"^(?:---|\+\+\+)\s+([^\t\n]+)", patch, re.MULTILINE):
            raw = match.group(1).strip()
            if raw == "/dev/null":
                continue
            if raw.startswith(("a/", "b/")):
                raw = raw[2:]
            paths.append(self._path(raw, must_exist=False))
        if not paths:
            raise TrajectoryError("patch contains no file headers")
        return paths

    def _apply_patch(self, args: Mapping[str, Any]) -> dict[str, Any]:
        patch = str(args.get("patch") or "")
        if not patch or len(patch) > 2_000_000:
            raise TrajectoryError("patch must contain 1-2,000,000 characters")
        paths = self._patch_paths(patch)
        check = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", "-"],
            cwd=self.root,
            input=patch,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if check.returncode:
            text, truncated = self._cap(check.stderr or check.stdout)
            return {"ok": False, "error": text, "truncated": truncated}
        apply = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=self.root,
            input=patch,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if apply.returncode:
            text, truncated = self._cap(apply.stderr or apply.stdout)
            return {"ok": False, "error": text, "truncated": truncated}
        return {"files": sorted({self._relative(path) for path in paths})}

    def _git_diff(self) -> dict[str, Any]:
        proc = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--"],
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        text, truncated = self._cap(proc.stdout if proc.returncode == 0 else proc.stderr)
        return {"diff": text, "returncode": proc.returncode, "truncated": truncated}

    def _run_tests(self) -> dict[str, Any]:
        if not self.test_argv:
            raise TrajectoryError("this task has no predefined test command")
        if not self.allow_host_tests:
            raise TrajectoryError(
                "host test execution is disabled; use a container/VM executor for unknown repositories"
            )
        env = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONNOUSERSITE": "1",
        }
        proc = subprocess.run(
            list(self.test_argv),
            cwd=self.root,
            env=env,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        text, truncated = self._cap(proc.stdout + proc.stderr)
        return {
            "argv": list(self.test_argv),
            "returncode": proc.returncode,
            "output": text,
            "truncated": truncated,
        }


class RoutedToolExecutor:
    """Dispatch one model tool call to its owning executor."""

    def __init__(self, executors: Sequence[ToolExecutor]) -> None:
        self.routes: dict[str, ToolExecutor] = {}
        for executor in executors:
            for name in executor.tool_names:
                if name in self.routes:
                    raise ValueError(f"duplicate tool executor route: {name}")
                self.routes[name] = executor
        self.tool_names = frozenset(self.routes)

    def execute(self, call: Mapping[str, Any]) -> dict[str, Any]:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        executor = self.routes.get(name)
        if executor is None:
            raise TrajectoryError(f"no executor is registered for tool {name!r}")
        return executor.execute(call)


def _assistant_message(message: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value for key, value in message.items() if not str(key).startswith("_")
    }
    result["role"] = "assistant"
    result["content"] = str(result.get("content") or "")
    if result.get("tool_calls"):
        result["tool_calls"] = normalize_tool_calls(result)
    return result


def validate_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    require_tool: bool,
    required_tool_names: Sequence[str] = (),
) -> dict[str, Any]:
    open_calls: set[str] = set()
    assistant_turns = 0
    tool_calls = 0
    tool_results = 0
    called_tool_names: set[str] = set()
    errors: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            assistant_turns += 1
            for call in normalize_tool_calls(message):
                call_id = call["id"]
                if call_id in open_calls:
                    errors.append(f"duplicate_tool_call_id:{call_id}")
                open_calls.add(call_id)
                tool_calls += 1
                called_tool_names.add(call["function"]["name"])
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in open_calls:
                errors.append(f"orphan_tool_result:{call_id}")
            else:
                open_calls.remove(call_id)
            tool_results += 1
    if open_calls:
        errors.append(f"missing_tool_results:{sorted(open_calls)}")
    if require_tool and tool_calls == 0:
        errors.append("required_tool_call_missing")
    for name in required_tool_names:
        if name not in called_tool_names:
            errors.append(f"required_tool_call_missing:{name}")
    if not messages or messages[-1].get("role") != "assistant":
        errors.append("trajectory_does_not_end_with_assistant")
    elif messages[-1].get("tool_calls"):
        errors.append("trajectory_ends_with_unresolved_tool_call")
    return {
        "valid": not errors,
        "errors": errors,
        "assistant_turns": assistant_turns,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
    }


def rollout_episode(
    *,
    episode_id: str,
    prompt: str,
    client: Any,
    executor: ToolExecutor,
    tools: Sequence[Mapping[str, Any]] = TOOL_DEFINITIONS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_rounds: int = 32,
    require_tool: bool = True,
    required_tool_names: Sequence[str] = (),
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(prompt)},
    ]
    tool_events: list[dict[str, Any]] = []
    response_metadata: list[dict[str, Any]] = []
    terminal_reason = "max_rounds"
    terminal_error: str | None = None
    for round_index in range(max_rounds):
        raw_assistant = client.complete(messages, tools)
        metadata = raw_assistant.get("_response_metadata") or {}
        response_metadata.append(dict(metadata))
        assistant = _assistant_message(raw_assistant)
        messages.append(assistant)
        calls = normalize_tool_calls(assistant)
        if not calls:
            finish_reason = metadata.get("finish_reason")
            if finish_reason in {"length", "max_tokens", "content_filter"}:
                terminal_reason = str(finish_reason)
                terminal_error = f"non_terminal_finish_reason:{finish_reason}"
                break
            terminal_reason = "final_answer"
            break
        for call in calls:
            result = executor.execute(call)
            event = {
                "round": round_index,
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
                "result": result,
            }
            tool_events.append(event)
            messages.append(
                {
                    "role": "tool",
                    "name": call["function"]["name"],
                    "tool_call_id": call["id"],
                    "content": _compact_json(result),
                }
            )
    validation = validate_messages(
        messages,
        require_tool=require_tool,
        required_tool_names=required_tool_names,
    )
    if terminal_error:
        validation["valid"] = False
        validation["errors"].append(terminal_error)
    return {
        "id": episode_id,
        "messages": messages,
        "tools": list(tools),
        "tool_events": tool_events,
        "terminal_reason": terminal_reason,
        "validation": validation,
        "response_metadata": response_metadata,
        "source_metadata": dict(source_metadata or {}),
    }


def rollout_from_messages(
    *,
    episode_id: str,
    initial_messages: Sequence[Mapping[str, Any]],
    client: Any,
    executor: ToolExecutor | None = None,
    tools: Sequence[Mapping[str, Any]] = (),
    remaining_user_turns: Sequence[str] = (),
    max_rounds: int = 32,
    require_tool: bool = False,
    required_tool_names: Sequence[str] = (),
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Continue a prompt, conversation seed, or structured trajectory prefix.

    ``remaining_user_turns`` contains only source user turns.  The function
    generates one assistant response before appending each following user turn,
    so user-only source histories are not mislabeled as completed conversations.
    Tool observations always come from ``executor``.
    """

    messages = canonical_messages_for_template(initial_messages)
    if not messages:
        raise TrajectoryError("initial_messages is empty")
    generation_start_message_index = len(messages)
    pending_users = [str(turn).strip() for turn in remaining_user_turns if str(turn).strip()]
    tool_events: list[dict[str, Any]] = []
    response_metadata: list[dict[str, Any]] = []
    terminal_reason = "max_rounds"
    terminal_error: str | None = None
    for round_index in range(max_rounds):
        raw_assistant = client.complete(messages, tools)
        metadata = dict(raw_assistant.get("_response_metadata") or {})
        response_metadata.append(metadata)
        assistant = _assistant_message(raw_assistant)
        messages.append(assistant)
        calls = normalize_tool_calls(assistant)
        if calls:
            if executor is None:
                raise TrajectoryError("model requested a tool but no executor was supplied")
            for call in calls:
                result = executor.execute(call)
                tool_events.append(
                    {
                        "round": round_index,
                        "tool_call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": call["function"]["name"],
                        "tool_call_id": call["id"],
                        "content": _compact_json(result),
                    }
                )
            continue
        finish_reason = metadata.get("finish_reason")
        if finish_reason in {"length", "max_tokens", "content_filter"}:
            terminal_reason = str(finish_reason)
            terminal_error = f"non_terminal_finish_reason:{finish_reason}"
            break
        if pending_users:
            messages.append({"role": "user", "content": pending_users.pop(0)})
            continue
        terminal_reason = "final_answer"
        break
    if pending_users and terminal_error is None:
        terminal_reason = "max_rounds_with_pending_user_turns"
    validation = validate_messages(
        messages,
        require_tool=require_tool,
        required_tool_names=required_tool_names,
    )
    if terminal_error:
        validation["valid"] = False
        validation["errors"].append(terminal_error)
    if pending_users:
        validation["valid"] = False
        validation["errors"].append(f"pending_user_turns:{len(pending_users)}")
    return {
        "id": episode_id,
        "messages": messages,
        "generation_start_message_index": generation_start_message_index,
        "tools": list(tools),
        "tool_events": tool_events,
        "terminal_reason": terminal_reason,
        "validation": validation,
        "response_metadata": response_metadata,
        "source_metadata": dict(source_metadata or {}),
    }


def canonical_messages_for_template(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove wire-only fields while keeping structured tool arguments."""

    canonical: list[dict[str, Any]] = []
    for message in messages:
        item = {
            key: value for key, value in message.items() if not str(key).startswith("_")
        }
        if item.get("role") == "assistant" and item.get("tool_calls"):
            item["tool_calls"] = normalize_tool_calls(item)
        canonical.append(item)
    return canonical


def render_with_assistant_mask(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    assistant_start_index: int = 0,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Render a trajectory and mark only tokens generated by assistant turns.

    Each assistant span starts after that turn's exact generation prompt and
    ends after its rendered end-of-turn token.  Prefix equality is checked, so
    a model template whose earlier rendering changes with future turns fails
    loudly instead of silently producing a wrong training mask.
    """

    canonical = canonical_messages_for_template(messages)
    if not 0 <= assistant_start_index <= len(canonical):
        raise ValueError("assistant_start_index is outside the message sequence")
    kwargs = dict(chat_template_kwargs or {})

    def render(prefix: Sequence[Mapping[str, Any]], *, generation: bool) -> list[int]:
        value = tokenizer.apply_chat_template(
            list(prefix),
            tools=list(tools),
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
        return [int(item) for item in value]

    full_ids = render(canonical, generation=False)
    mask = [0] * len(full_ids)
    for index, message in enumerate(canonical):
        if index < assistant_start_index or message.get("role") != "assistant":
            continue
        generation_prefix = render(canonical[:index], generation=True)
        through_turn = render(canonical[: index + 1], generation=False)
        if full_ids[: len(generation_prefix)] != generation_prefix:
            raise TrajectoryError(
                f"chat template is not prefix-stable before assistant message {index}"
            )
        if full_ids[: len(through_turn)] != through_turn:
            raise TrajectoryError(
                f"chat template is not prefix-stable after assistant message {index}"
            )
        if len(through_turn) <= len(generation_prefix):
            raise TrajectoryError(f"assistant message {index} rendered no target tokens")
        mask[len(generation_prefix) : len(through_turn)] = [1] * (
            len(through_turn) - len(generation_prefix)
        )
    return full_ids, mask


def eligible_anchor_positions(assistant_mask: Sequence[int], *, block_size: int = 16) -> list[int]:
    """Positions whose complete speculative block stays in an assistant span."""

    if block_size < 1:
        raise ValueError("block_size must be positive")
    mask = [bool(item) for item in assistant_mask]
    if len(mask) < block_size:
        return []
    return [
        start
        for start in range(len(mask) - block_size + 1)
        if all(mask[start : start + block_size])
    ]


def select_anchor_positions(
    assistant_mask: Sequence[int],
    *,
    episode_id: str,
    count: int = 512,
    block_size: int = 16,
    seed: int = 0,
) -> list[int]:
    """Deterministically sample anchor positions without crossing tool turns."""

    import random

    eligible = eligible_anchor_positions(assistant_mask, block_size=block_size)
    if len(eligible) < count:
        raise TrajectoryError(
            f"episode has {len(eligible)} eligible anchors, fewer than required {count}"
        )
    rng = random.Random(_stable_int(seed, episode_id, count, block_size))
    return sorted(rng.sample(eligible, count))
