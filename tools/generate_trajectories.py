#!/usr/bin/env python3
"""Stage A: generate and freeze GLM-5.2 coding-agent trajectories with SGLang."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm_dflash2.agent_trajectory import (  # noqa: E402
    ChatCompletionConfig,
    OpenAIChatClient,
    RoutedToolExecutor,
    TrajectoryError,
    rollout_from_messages,
    tool_definitions_for,
)
from glm_dflash2.sglang_stage_a import (  # noqa: E402
    AttemptErrorLedger,
    CommittedJsonlWriter,
    SGLangServerConfig,
    build_server_command,
    owns_source_index,
)
from glm_dflash2.jsonl import OutputShardLock, repair_truncated_jsonl  # noqa: E402
from glm_dflash2.trajectory_tokens import freeze_trajectory_tokens  # noqa: E402
from glm_dflash2.vibe_coding import (  # noqa: E402
    ModelInput,
    iter_table_rows,
    load_vibe_coding_table,
    row_to_model_input,
)
from glm_dflash2.provenance import (  # noqa: E402
    dataset_fingerprint,
    load_endpoint_manifest_attestation,
    local_model_fingerprint,
)
from glm_dflash2.target_io import model_revision, tokenizer_fingerprint  # noqa: E402
from glm_dflash2.open_swe_trajectories import OpenSWETrajectoryStore  # noqa: E402
from glm_dflash2.workspaces import (  # noqa: E402
    ORIGINAL_TRAJECTORY_KINDS,
    AutomaticWorkspaceProvider,
)
from glm_dflash2.web_tools import (  # noqa: E402
    BROWSER_TOOL_NAME,
    DEFAULT_SEARXNG_ENDPOINT,
    DEFAULT_SERPER_ENDPOINT,
    DEFAULT_SERPER_SCRAPE_ENDPOINT,
    WEB_RESEARCH_SYSTEM_APPENDIX,
    WEB_SEARCH_TOOL_NAME,
    WEB_TOOL_DEFINITIONS,
    WebToolClient,
    WebToolConfig,
    WebToolExecutor,
)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/vibe_coding_630k")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/GLM-5.2")
    parser.add_argument("--served-model-name", default="GLM-5.2")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument(
        "--endpoint-manifest",
        type=Path,
        help="Immutable model/runtime identity for an external SGLang endpoint.",
    )
    parser.add_argument(
        "--allow-unverified-endpoint",
        action="store_true",
        help="Smoke-only escape hatch; production external endpoints require a manifest.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=16)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--attention-backend", default="ascend")
    parser.add_argument("--quantization")
    parser.add_argument("--moe-a2a-backend")
    parser.add_argument("--deepep-mode")
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--max-total-tokens", type=int, default=131072)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent trajectory episodes; SGLang concurrency is limited separately.",
    )
    parser.add_argument("--episode-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--server-extra-arg", action="append", default=[])
    parser.add_argument("--workspace-map", type=Path)
    parser.add_argument(
        "--workspace-cache", type=Path, default=ROOT / "outputs/workspace_cache"
    )
    parser.add_argument(
        "--open-swe-store",
        type=Path,
        default=ROOT / "outputs/open_swe_original.sqlite",
    )
    parser.add_argument("--repo-url-template", default="https://github.com/{repo}.git")
    parser.add_argument("--container-runtime", default="docker")
    parser.add_argument("--no-container-pull", action="store_true")
    parser.add_argument("--container-network", default="none")
    parser.add_argument("--container-cpus", type=float)
    parser.add_argument("--container-memory", default="")
    parser.add_argument("--input-kind", action="append", dest="input_kinds")
    parser.add_argument("--row-id")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-rounds", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-sequence-tokens", type=int, default=131072)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--chat-template-kwargs-json", default='{"enable_thinking":true}')
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--allow-host-tests", action="store_true")
    parser.add_argument("--shard-index", type=int, default=int(os.getenv("DATA_SHARD_INDEX", "0")))
    parser.add_argument("--shard-count", type=int, default=int(os.getenv("DATA_SHARD_COUNT", "1")))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--web-input-kind", action="append", default=["web_research_coding"])
    parser.add_argument("--web-tools-for-all", action="store_true")
    parser.add_argument("--web-search-provider", choices=["searxng", "serper"], default="searxng")
    parser.add_argument("--web-search-endpoint")
    parser.add_argument("--browser-provider", choices=["direct", "serper"], default="direct")
    parser.add_argument("--browser-endpoint", default=DEFAULT_SERPER_SCRAPE_ENDPOINT)
    parser.add_argument("--web-api-key-file", type=Path)
    parser.add_argument("--web-search-max-calls", type=int, default=2)
    parser.add_argument("--browser-max-calls", type=int, default=8)
    parser.add_argument("--show-result", action="store_true")
    return parser.parse_args()


class ThreadLocalClientPool:
    """Create one non-shared closeable client per trajectory worker."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._local = threading.local()
        self._clients: list[Any] = []
        self._lock = threading.Lock()

    def get(self) -> Any:
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._factory()
            self._local.client = client
            with self._lock:
                self._clients.append(client)
        return client

    def close(self) -> None:
        with self._lock:
            clients, self._clients = self._clients, []
        for client in clients:
            close = getattr(client, "close", None)
            if close is not None:
                close()
            elif getattr(client, "session", None) is not None:
                client.session.close()


class ConcurrencyLimitedChatClient:
    """Limit model HTTP calls while leaving tool execution fully concurrent."""

    def __init__(self, client: Any, semaphore: threading.BoundedSemaphore) -> None:
        self._client = client
        self._semaphore = semaphore

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        with self._semaphore:
            return self._client.complete(messages, tools)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()
        elif getattr(self._client, "session", None) is not None:
            self._client.session.close()


def bounded_completed_futures(
    function: Callable[[Any], Any],
    values: Iterable[Any],
    *,
    max_workers: int,
    max_pending: int,
) -> Iterator[tuple[Any, Future[Any]]]:
    """Submit bounded work and yield completed futures without head-of-line blocking."""

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if max_pending < 1:
        raise ValueError("max_pending must be positive")
    pending: dict[Future[Any], Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for value in values:
            future = executor.submit(function, value)
            pending[future] = value
            if len(pending) < max_pending:
                continue
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for completed in finished:
                yield pending.pop(completed), completed
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for completed in finished:
                yield pending.pop(completed), completed


def retry_call(
    function: Callable[[], Any],
    *,
    retries: int,
    backoff_seconds: float,
) -> Any:
    """Retry one isolated episode with bounded exponential backoff."""

    if retries < 0:
        raise ValueError("retries cannot be negative")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds cannot be negative")
    for attempt in range(retries + 1):
        try:
            return function()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            if attempt == retries:
                raise
            if backoff_seconds:
                time.sleep(backoff_seconds * (2**attempt))
    raise AssertionError("unreachable")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            yield value


def _existing_ids(path: Path) -> set[str]:
    result: set[str] = set()
    for row in _iter_jsonl(path):
        sample_id = str(row.get("id") or "")
        if not sample_id:
            continue
        if sample_id in result:
            raise ValueError(f"duplicate trajectory id in {path}: {sample_id}")
        result.add(sample_id)
    return result


def _workspace_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = {}
    for row in _iter_jsonl(path):
        sample_id = str(row.get("id") or row.get("source_id") or "")
        if not sample_id or sample_id in rows:
            raise ValueError(f"invalid/duplicate workspace map id: {sample_id!r}")
        rows[sample_id] = row
    return rows


def _select_table(table: pa.Table, args: argparse.Namespace) -> pa.Table:
    mask = None
    if args.row_id:
        mask = pc.equal(table["id"], args.row_id)
    if args.input_kinds:
        selected = pc.is_in(table["input_kind"], value_set=pa.array(args.input_kinds))
        mask = selected if mask is None else pc.and_(mask, selected)
    result = table if mask is None else table.filter(mask)
    if args.row_id and result.num_rows != 1:
        raise ValueError(f"--row-id matched {result.num_rows} rows")
    return result


def _wait(endpoint: str, process: subprocess.Popen[str], log_path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(errors="replace")[-12000:] if log_path.exists() else ""
            raise RuntimeError(f"SGLang exited with {process.returncode}:\n{tail}")
        try:
            if requests.get(endpoint.rstrip("/") + "/health", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("SGLang server did not become healthy")


@contextmanager
def _endpoint(args: argparse.Namespace, temp_root: Path) -> Iterator[str]:
    if args.endpoint:
        yield args.endpoint
        return
    config = SGLangServerConfig(
        python=args.python,
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        host=args.host,
        port=args.port,
        tp_size=args.tp_size,
        device=args.device,
        attention_backend=args.attention_backend,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        max_total_tokens=args.max_total_tokens,
        quantization=args.quantization,
        moe_a2a_backend=args.moe_a2a_backend,
        deepep_mode=args.deepep_mode,
        extra_args=tuple(args.server_extra_arg),
    )
    log_path = temp_root / "sglang-stage-a.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        build_server_command(config),
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    endpoint = f"http://{args.host}:{args.port}"
    try:
        _wait(endpoint, process, log_path, args.timeout)
        yield endpoint
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log_handle.close()


def _append_web_policy(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(message) for message in messages]
    for message in result:
        if message.get("role") == "system":
            content = str(message.get("content") or "").rstrip()
            if WEB_RESEARCH_SYSTEM_APPENDIX not in content:
                message["content"] = content + "\n\n" + WEB_RESEARCH_SYSTEM_APPENDIX
            return result
    result.insert(0, {"role": "system", "content": WEB_RESEARCH_SYSTEM_APPENDIX})
    return result


def _tool_name(tool: Mapping[str, Any]) -> str:
    return str((tool.get("function") or {}).get("name") or "")


def _ordered_tools(item_tools: Sequence[Mapping[str, Any]], additions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in [*item_tools, *additions]:
        name = _tool_name(raw)
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(dict(raw))
    return result


def _web_client(args: argparse.Namespace) -> WebToolClient:
    endpoint = args.web_search_endpoint or (
        DEFAULT_SEARXNG_ENDPOINT if args.web_search_provider == "searxng" else DEFAULT_SERPER_ENDPOINT
    )
    key = args.web_api_key_file.read_text().strip() if args.web_api_key_file else ""
    return WebToolClient(
        WebToolConfig(
            search_provider=args.web_search_provider,
            search_endpoint=endpoint,
            browser_provider=args.browser_provider,
            browser_endpoint=args.browser_endpoint,
            api_key=key,
            search_max_calls=args.web_search_max_calls,
            browser_max_calls=args.browser_max_calls,
        )
    )


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.with_suffix(path.suffix + ".manifest.json")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(compact_json(payload) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _ids_digest(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(ids):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_one(
    *,
    item: ModelInput,
    source_index: int,
    args: argparse.Namespace,
    client: OpenAIChatClient,
    tokenizer: Any,
    chat_kwargs: Mapping[str, Any],
    open_swe_store: OpenSWETrajectoryStore | None,
    web_client: WebToolClient | None,
    workspace_provider: AutomaticWorkspaceProvider,
) -> tuple[dict[str, Any], str]:
    """Generate or restore one item without mutating shard-level state."""

    if item.input_kind in ORIGINAL_TRAJECTORY_KINDS:
        if open_swe_store is None:
            raise TrajectoryError("Open-SWE trajectory store is unavailable")
        trajectory = open_swe_store.get(item)
        if not trajectory["validation"]["valid"]:
            raise TrajectoryError(compact_json(trajectory["validation"]))
        trajectory["source_metadata"].update(
            {
                "selected_source_index": source_index,
                "model": args.served_model_name,
                "target_replay_mode": "teacher_forced_original_trajectory",
            }
        )
        route = "original_trajectory"
    else:
        web_enabled = args.web_tools_for_all or item.input_kind in set(
            args.web_input_kind
        )
        web_executor = (
            WebToolExecutor(
                web_client,
                search_max_calls=args.web_search_max_calls,
                browser_max_calls=args.browser_max_calls,
            )
            if web_enabled and web_client is not None
            else None
        )
        messages = _append_web_policy(item.messages) if web_enabled else item.messages
        with workspace_provider.acquire(item) as workspace_lease:
            workspace_executor = (
                workspace_lease.executor if workspace_lease is not None else None
            )
            executors = [
                value for value in (workspace_executor, web_executor) if value is not None
            ]
            executor = RoutedToolExecutor(executors) if executors else None
            additions: list[Mapping[str, Any]] = []
            if workspace_executor:
                additions.extend(tool_definitions_for(workspace_executor.tool_names))
            if web_executor:
                additions.extend(WEB_TOOL_DEFINITIONS)
            # Expose only tools backed by the active real executor. Prefix rows
            # with source schemas take the restoration route above.
            tools = _ordered_tools((), additions)
            trajectory = rollout_from_messages(
                episode_id=item.id,
                initial_messages=messages,
                remaining_user_turns=item.remaining_user_turns,
                client=client,
                executor=executor,
                tools=tools,
                max_rounds=args.max_rounds,
                require_tool=executor is not None,
                required_tool_names=(WEB_SEARCH_TOOL_NAME, BROWSER_TOOL_NAME)
                if item.input_kind == "web_research_coding"
                else (),
                source_metadata={
                    **item.source_metadata,
                    "selected_source_index": source_index,
                    "input_kind": item.input_kind,
                    "route": item.route,
                    "repo": item.repo,
                    "base_commit": item.base_commit,
                    "model": args.served_model_name,
                    "workspace_mode": workspace_lease.mode if workspace_lease else "none",
                },
            )
        route = item.route

    if not trajectory["validation"]["valid"]:
        raise TrajectoryError(compact_json(trajectory["validation"]))
    trajectory.update(
        freeze_trajectory_tokens(
            tokenizer,
            trajectory,
            chat_template_kwargs=chat_kwargs,
            max_sequence_tokens=args.max_sequence_tokens,
            require_server_token_ids=route != "original_trajectory",
        )
    )
    trajectory["stage_a_complete"] = True
    return trajectory, route


def main() -> int:
    args = parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_running_requests < 1:
        raise ValueError("--max-running-requests must be positive")
    if args.max_total_tokens < 1:
        raise ValueError("--max-total-tokens must be positive")
    if args.episode_retries < 0:
        raise ValueError("--episode-retries cannot be negative")
    if args.retry_backoff_seconds < 0:
        raise ValueError("--retry-backoff-seconds cannot be negative")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k == 0 or args.top_k < -1:
        raise ValueError("--top-k must be -1 (disabled) or a positive integer")
    chat_kwargs = json.loads(args.chat_template_kwargs_json)
    if not isinstance(chat_kwargs, dict):
        raise ValueError("--chat-template-kwargs-json must decode to an object")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    table = load_vibe_coding_table(args.dataset)
    selected = _select_table(table, args)
    owned_ids = [
        str(sample_id)
        for source_index, sample_id in enumerate(selected["id"].to_pylist())
        if owns_source_index(
            source_index, shard_index=args.shard_index, shard_count=args.shard_count
        )
    ]
    data_fingerprint = dataset_fingerprint(args.dataset)
    model_fingerprint = local_model_fingerprint(args.model_path)
    tokenizer_identity = tokenizer_fingerprint(args.model_path)
    target_config = json.loads(
        (args.model_path / "config.json").read_text(encoding="utf-8")
    )
    target_revision = model_revision(target_config, model_fingerprint)
    target_vocab_size = int(target_config["vocab_size"])
    if args.endpoint_manifest is not None and args.endpoint is None:
        raise ValueError("--endpoint-manifest is only valid with --endpoint")
    if args.allow_unverified_endpoint and args.endpoint is None:
        raise ValueError("--allow-unverified-endpoint is only valid with --endpoint")
    endpoint_identity = None
    if args.endpoint is not None:
        if args.endpoint_manifest is None and not args.allow_unverified_endpoint:
            raise ValueError(
                "external --endpoint requires --endpoint-manifest; use "
                "--allow-unverified-endpoint only for a small smoke run"
            )
        if args.endpoint_manifest is not None:
            endpoint_identity = load_endpoint_manifest_attestation(
                args.endpoint_manifest,
                expected_model_fingerprint=model_fingerprint,
                expected_tokenizer_fingerprint=tokenizer_identity,
                expected_served_model_name=args.served_model_name,
            )
    workspaces = _workspace_map(args.workspace_map)
    workspace_provider = AutomaticWorkspaceProvider(
        args.workspace_cache,
        workspace_map=workspaces,
        repo_url_template=args.repo_url_template,
        container_runtime=args.container_runtime,
        container_auto_pull=not args.no_container_pull,
        container_network=args.container_network,
        container_cpus=args.container_cpus,
        container_memory=args.container_memory,
        allow_host_tests=args.allow_host_tests,
        timeout_seconds=args.timeout,
    )
    selected_kinds = frozenset(selected["input_kind"].unique().to_pylist())
    needs_open_swe_store = bool(selected_kinds & ORIGINAL_TRAJECTORY_KINDS)
    if needs_open_swe_store and not args.open_swe_store.is_file():
        raise FileNotFoundError(
            f"{args.open_swe_store} is required for Open-SWE trajectory rows; "
            "build it with tools/prepare_open_swe_trajectories.py"
        )
    output_lock = OutputShardLock(args.output_jsonl)
    output_lock.__enter__()
    if not args.no_resume:
        repair_truncated_jsonl(args.output_jsonl)
    done = set() if args.no_resume else _existing_ids(args.output_jsonl)
    truncate = args.no_resume
    run_contract = {
        "schema_version": 2,
        "stage": "trajectory",
        "status": "running",
        "model_path": str(args.model_path.resolve()),
        "model_fingerprint": model_fingerprint,
        "model_revision": target_revision,
        "tokenizer_fingerprint": tokenizer_identity,
        "vocab_size": target_vocab_size,
        "served_model_name": args.served_model_name,
        "service": {
            "mode": "external_endpoint" if args.endpoint else "local_sglang",
            "endpoint": args.endpoint,
            "host": args.host,
            "port": args.port,
            "tp_size": args.tp_size,
            "mem_fraction_static": args.mem_fraction_static,
            "context_length": args.context_length,
            "max_running_requests": args.max_running_requests,
            "max_total_tokens": args.max_total_tokens,
            "server_extra_args": list(args.server_extra_arg),
            "weight_identity_verified": args.endpoint is None,
            "weight_identity_status": (
                "local_full_content_hash"
                if args.endpoint is None
                else "operator_attested"
                if endpoint_identity is not None
                else "unverified_smoke_only"
            ),
            "endpoint_identity": endpoint_identity,
        },
        "dataset": str(args.dataset.resolve()),
        "dataset_fingerprint": data_fingerprint.digest,
        "dataset_revision": data_fingerprint.revision,
        "matching_rows": selected.num_rows,
        "owned_rows": len(owned_ids),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
        "chat_template_kwargs": chat_kwargs,
        "max_rounds": args.max_rounds,
        "max_new_tokens": args.max_new_tokens,
        "max_sequence_tokens": args.max_sequence_tokens,
        "execution": {
            "workers": args.workers,
            "max_pending": args.workers * 2,
            "episode_retries": args.episode_retries,
            "retry_backoff_seconds": args.retry_backoff_seconds,
        },
        "input_kinds": args.input_kinds,
        "row_id": args.row_id,
        "workspace": {
            "map_path": str(args.workspace_map.resolve()) if args.workspace_map else None,
            "map_sha256": _file_digest(args.workspace_map) if args.workspace_map else None,
            "repo_url_template": args.repo_url_template,
            "container_runtime": args.container_runtime,
            "container_network": args.container_network,
            "container_auto_pull": not args.no_container_pull,
            "container_cpus": args.container_cpus,
            "container_memory": args.container_memory,
            "allow_host_tests": args.allow_host_tests,
        },
        "open_swe_store": (
            {
                "path": str(args.open_swe_store.resolve()),
                "sha256": _file_digest(args.open_swe_store),
            }
            if needs_open_swe_store
            else None
        ),
        "web_tools": {
            "input_kinds": args.web_input_kind,
            "for_all": args.web_tools_for_all,
            "search_provider": args.web_search_provider,
            "browser_provider": args.browser_provider,
            "search_max_calls": args.web_search_max_calls,
            "browser_max_calls": args.browser_max_calls,
        },
    }
    manifest_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".manifest.json")
    if manifest_path.exists() and not args.no_resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in run_contract.items():
            if key == "status":
                continue
            if key not in previous:
                raise ValueError(f"resume contract is missing required key {key}")
            if previous[key] != value:
                raise ValueError(
                    f"resume contract mismatch for {key}: {previous[key]!r} != {value!r}"
                )
    _write_manifest(args.output_jsonl, run_contract)

    with tempfile.TemporaryDirectory(prefix="glm52-stage-a-") as temp_name:
        temp_root = Path(temp_name)
        error_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".errors.jsonl")
        with _endpoint(args, temp_root) as endpoint, ExitStack() as stack, CommittedJsonlWriter(
            args.output_jsonl, truncate=truncate, lock=False
        ) as writer, AttemptErrorLedger(error_path, truncate=truncate) as error_ledger:
            # A crash can happen after the trajectory commit and before the
            # corresponding ledger resolution. The committed JSONL wins.
            for committed_id in done & set(error_ledger.unresolved_ids):
                error_ledger.resolve(sample_id=committed_id, source_index=-1)
            chat_config = ChatCompletionConfig(
                endpoint=endpoint,
                model=args.served_model_name,
                timeout_seconds=args.timeout,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_new_tokens,
                reasoning_effort=None,
                chat_template_kwargs=chat_kwargs,
                return_token_ids=True,
            )
            probe_client = OpenAIChatClient(chat_config)
            probe_client.assert_model_available()
            probe_client.assert_token_id_capability()
            stack.callback(probe_client.session.close)
            completion_slots = threading.BoundedSemaphore(args.max_running_requests)
            chat_clients = ThreadLocalClientPool(
                lambda: ConcurrencyLimitedChatClient(
                    OpenAIChatClient(chat_config), completion_slots
                )
            )
            stack.callback(chat_clients.close)
            uses_any_web = args.web_tools_for_all or any(
                kind in set(args.web_input_kind) for kind in selected["input_kind"].unique().to_pylist()
            )
            web_clients = (
                ThreadLocalClientPool(lambda: _web_client(args))
                if uses_any_web
                else None
            )
            if web_clients is not None:
                stack.callback(web_clients.close)
            open_swe_store = (
                OpenSWETrajectoryStore(args.open_swe_store)
                if needs_open_swe_store
                else None
            )
            if open_swe_store is not None:
                stack.callback(open_swe_store.close)

            attempted = accepted = 0

            def record_error(source_index: int, sample_id: str, exc: BaseException) -> None:
                error_ledger.record_error(
                    sample_id=sample_id, source_index=source_index, error=exc
                )
                print(
                    compact_json(
                        {
                            "id": sample_id,
                            "source_index": source_index,
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )

            def commit(
                source_index: int,
                sample_id: str,
                trajectory: Mapping[str, Any],
                route: str,
            ) -> None:
                nonlocal accepted
                writer.append(trajectory)
                error_ledger.resolve(sample_id=sample_id, source_index=source_index)
                done.add(sample_id)
                accepted += 1
                summary = {
                    "id": sample_id,
                    "source_index": source_index,
                    "route": route,
                    "tokens": len(trajectory["input_ids"]),
                    "supervised_tokens": sum(trajectory["loss_mask"]),
                }
                if args.show_result:
                    summary["messages"] = trajectory["messages"]
                print(compact_json(summary), flush=True)

            def jobs() -> Iterator[tuple[int, str, ModelInput]]:
                nonlocal attempted
                for source_index, source_row in enumerate(iter_table_rows(selected)):
                    if not owns_source_index(
                        source_index,
                        shard_index=args.shard_index,
                        shard_count=args.shard_count,
                    ):
                        continue
                    sample_id = str(
                        source_row.get("id")
                        or source_row.get("source_id")
                        or f"source-index-{source_index}"
                    )
                    if sample_id in done:
                        continue
                    if args.max_samples is not None and attempted >= args.max_samples:
                        break
                    attempted += 1
                    try:
                        item = row_to_model_input(source_row)
                        if item.input_kind in ORIGINAL_TRAJECTORY_KINDS:
                            trajectory, route = _generate_one(
                                item=item,
                                source_index=source_index,
                                args=args,
                                client=probe_client,
                                tokenizer=tokenizer,
                                chat_kwargs=chat_kwargs,
                                open_swe_store=open_swe_store,
                                web_client=None,
                                workspace_provider=workspace_provider,
                            )
                            commit(source_index, sample_id, trajectory, route)
                            continue
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except BaseException as exc:
                        record_error(source_index, sample_id, exc)
                        continue
                    yield source_index, sample_id, item

            def generate(job: tuple[int, str, ModelInput]) -> tuple[dict[str, Any], str]:
                source_index, _sample_id, item = job

                def attempt() -> tuple[dict[str, Any], str]:
                    web_enabled = args.web_tools_for_all or item.input_kind in set(
                        args.web_input_kind
                    )
                    return _generate_one(
                        item=item,
                        source_index=source_index,
                        args=args,
                        client=chat_clients.get(),
                        tokenizer=tokenizer,
                        chat_kwargs=chat_kwargs,
                        open_swe_store=None,
                        web_client=(
                            web_clients.get()
                            if web_enabled and web_clients is not None
                            else None
                        ),
                        workspace_provider=workspace_provider,
                    )

                return retry_call(
                    attempt,
                    retries=args.episode_retries,
                    backoff_seconds=args.retry_backoff_seconds,
                )

            for job, future in bounded_completed_futures(
                generate,
                jobs(),
                max_workers=args.workers,
                max_pending=args.workers * 2,
            ):
                source_index, sample_id, _item = job
                try:
                    trajectory, route = future.result()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:
                    record_error(source_index, sample_id, exc)
                    continue
                commit(source_index, sample_id, trajectory, route)

            unresolved_ids = set(error_ledger.unresolved_ids)

    committed_ids = _existing_ids(args.output_jsonl)
    missing = set(owned_ids) - committed_ids
    extra = committed_ids - set(owned_ids)
    complete = not missing and not extra and not unresolved_ids
    status = "frozen" if complete else ("partial" if args.max_samples is not None else "incomplete")
    run_contract.update(
        {
            "status": status,
            "attempted_this_run": attempted,
            "accepted_this_run": accepted,
            "committed_ids": len(committed_ids),
            "committed_ids_sha256": _ids_digest(list(committed_ids)),
            "jsonl_bytes": args.output_jsonl.stat().st_size,
            "jsonl_sha256": _file_digest(args.output_jsonl),
            "unresolved_errors": len(unresolved_ids),
            "unresolved_ids_sha256": _ids_digest(list(unresolved_ids)),
            "missing_ids": len(missing),
            "extra_ids": len(extra),
        }
    )
    _write_manifest(args.output_jsonl, run_contract)
    output_lock.__exit__(None, None, None)
    print(compact_json(run_contract), flush=True)
    if status == "incomplete":
        raise RuntimeError(
            "trajectory shard is incomplete: "
            f"missing={len(missing)} extra={len(extra)} errors={len(unresolved_ids)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
