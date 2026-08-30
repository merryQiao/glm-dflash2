"""On-demand isolated workspaces for vibe-coding trajectory generation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .agent_trajectory import (
    WORKSPACE_TOOL_NAMES,
    ToolExecutor,
    TrajectoryError,
    WorkspaceToolExecutor,
)
from .vibe_coding import ModelInput


GIT_WORKSPACE_KINDS = frozenset({"repo_checkout_reference"})
CONTAINER_WORKSPACE_KINDS = frozenset({"executable_repo_reference"})
ORIGINAL_TRAJECTORY_KINDS = frozenset(
    {"agent_trajectory_prefix", "terminal_trajectory_prefix"}
)
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout: float = 3600.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        list(argv),
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and process.returncode:
        output = (process.stderr or process.stdout)[-12_000:]
        raise TrajectoryError(
            f"command failed ({process.returncode}): {list(argv)!r}\n{output}"
        )
    return process


def _safe_name(value: str, *, limit: int = 80) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:limit]
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{prefix or 'workspace'}-{digest}"


def _seed_workspace(root: Path, item: ModelInput) -> Path:
    workspace = root / "checkout"
    workspace.mkdir(parents=True)
    resolved_workspace = workspace.resolve()
    for relative, content in item.workspace_seed_files.items():
        destination = (workspace / relative).resolve()
        if not destination.is_relative_to(resolved_workspace):
            raise TrajectoryError(f"seed file escapes workspace: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    _run(["git", "init", "-q", str(workspace)])
    _run(["git", "-C", str(workspace), "add", "."])
    return workspace


@dataclass(frozen=True)
class WorkspaceLease:
    """One executor and the provenance of its isolated workspace."""

    executor: ToolExecutor
    mode: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class GitWorkspaceProvider:
    """Reuse one bare mirror per repository and create worktrees per episode."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        repo_url_template: str = "https://github.com/{repo}.git",
        timeout_seconds: float = 3600.0,
        allow_host_tests: bool = False,
    ) -> None:
        self.root = Path(cache_root).expanduser().resolve()
        self.mirrors = self.root / "git-mirrors"
        self.runs = self.root / "git-runs"
        self.locks = self.root / "locks"
        for path in (self.mirrors, self.runs, self.locks):
            path.mkdir(parents=True, exist_ok=True)
        self.repo_url_template = repo_url_template
        self.timeout_seconds = timeout_seconds
        self.allow_host_tests = allow_host_tests

    def _validate(self, item: ModelInput) -> None:
        if not _REPO_RE.fullmatch(item.repo):
            raise TrajectoryError(f"invalid GitHub repository name: {item.repo!r}")
        if not _COMMIT_RE.fullmatch(item.base_commit):
            raise TrajectoryError(
                f"workspace requires a hexadecimal base_commit: {item.base_commit!r}"
            )

    def _mirror_path(self, repo: str) -> Path:
        return self.mirrors / (_safe_name(repo) + ".git")

    @contextmanager
    def _repo_lock(self, repo: str) -> Iterator[None]:
        lock_path = self.locks / (_safe_name(repo) + ".lock")
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield

    def _has_commit(self, mirror: Path, commit: str) -> bool:
        process = _run(
            ["git", "--git-dir", str(mirror), "cat-file", "-e", f"{commit}^{{commit}}"],
            timeout=self.timeout_seconds,
            check=False,
        )
        return process.returncode == 0

    def _ensure_mirror(self, item: ModelInput) -> Path:
        mirror = self._mirror_path(item.repo)
        with self._repo_lock(item.repo):
            if not mirror.exists():
                url = self.repo_url_template.format(repo=item.repo)
                with tempfile.TemporaryDirectory(
                    prefix="clone-", dir=self.mirrors
                ) as temp_name:
                    candidate = Path(temp_name) / "mirror.git"
                    _run(
                        [
                            "git",
                            "clone",
                            "--mirror",
                            "--filter=blob:none",
                            url,
                            str(candidate),
                        ],
                        timeout=self.timeout_seconds,
                    )
                    os.replace(candidate, mirror)
            if not self._has_commit(mirror, item.base_commit):
                _run(
                    [
                        "git",
                        "--git-dir",
                        str(mirror),
                        "fetch",
                        "--no-tags",
                        "origin",
                        item.base_commit,
                    ],
                    timeout=self.timeout_seconds,
                )
            if not self._has_commit(mirror, item.base_commit):
                raise TrajectoryError(
                    f"base commit {item.base_commit} is unavailable in {item.repo}"
                )
        return mirror

    @contextmanager
    def acquire(self, item: ModelInput) -> Iterator[WorkspaceLease]:
        self._validate(item)
        mirror = self._ensure_mirror(item)
        _run(["git", "--git-dir", str(mirror), "worktree", "prune"])
        with tempfile.TemporaryDirectory(
            prefix=_safe_name(item.id) + "-", dir=self.runs
        ) as temp_name:
            workspace = Path(temp_name) / "checkout"
            _run(
                [
                    "git",
                    "--git-dir",
                    str(mirror),
                    "worktree",
                    "add",
                    "--detach",
                    str(workspace),
                    item.base_commit,
                ],
                timeout=self.timeout_seconds,
            )
            executor = WorkspaceToolExecutor(
                workspace,
                allow_host_tests=self.allow_host_tests,
                timeout_seconds=self.timeout_seconds,
            )
            try:
                yield WorkspaceLease(
                    executor=executor,
                    mode="git-worktree",
                    metadata={
                        "repo": item.repo,
                        "base_commit": item.base_commit,
                        "mirror": str(mirror),
                    },
                )
            finally:
                _run(
                    [
                        "git",
                        "--git-dir",
                        str(mirror),
                        "worktree",
                        "remove",
                        "--force",
                        str(workspace),
                    ],
                    timeout=min(self.timeout_seconds, 300.0),
                    check=False,
                )


class ContainerWorkspaceToolExecutor:
    """Workspace tools executed inside one isolated OCI container."""

    def __init__(
        self,
        container: str,
        *,
        runtime: str = "docker",
        root: str = "/testbed",
        test_command: str = "",
        timeout_seconds: float = 3600.0,
        max_output_chars: int = 32_768,
    ) -> None:
        self.container = container
        self.runtime = runtime
        self.root = root.rstrip("/") or "/"
        self.test_command = test_command.strip()
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.tool_names = (
            WORKSPACE_TOOL_NAMES
            if self.test_command
            else WORKSPACE_TOOL_NAMES - {"run_tests"}
        )

    def _cap(self, value: str) -> tuple[str, bool]:
        if len(value) <= self.max_output_chars:
            return value, False
        half = self.max_output_chars // 2
        return (
            value[:half] + "\n...[OUTPUT TRUNCATED]...\n" + value[-half:],
            True,
        )

    def _path(self, value: Any) -> str:
        raw = str(value or ".")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise TrajectoryError("path escapes the container workspace")
        normalized = relative.as_posix()
        return self.root if normalized == "." else f"{self.root}/{normalized}"

    def _exec(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return _run(
            [self.runtime, "exec", "-i", self.container, *argv],
            input_text=input_text,
            timeout=timeout or self.timeout_seconds,
            check=check,
        )

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
        path = self._path(args.get("path", "."))
        depth = max(1, min(6, int(args.get("max_depth", 2))))
        script = (
            'find "$1" -mindepth 1 -maxdepth "$2" '
            "\\( -path '*/.git' -o -path '*/.git/*' \\) -prune -o -print | head -2000"
        )
        process = self._exec(["sh", "-c", script, "specforge", path, str(depth)])
        output = process.stdout if process.returncode == 0 else process.stderr
        if process.returncode == 0:
            output = "\n".join(
                line.removeprefix(self.root + "/")
                for line in output.splitlines()
            )
        text, truncated = self._cap(output)
        return {
            "path": path.removeprefix(self.root).lstrip("/") or ".",
            "entries": text,
            "returncode": process.returncode,
            "truncated": truncated,
        }

    def _read_file(self, args: Mapping[str, Any]) -> dict[str, Any]:
        path = self._path(args.get("path"))
        start = max(1, int(args.get("start_line", 1)))
        end = max(start, min(start + 1999, int(args.get("end_line", 400))))
        script = (
            'test -f "$1" && '
            "awk -v start=\"$2\" -v end=\"$3\" "
            "'NR >= start && NR <= end {print NR \": \" $0}' \"$1\""
        )
        process = self._exec(
            ["sh", "-c", script, "specforge", path, str(start), str(end)]
        )
        if process.returncode:
            raise TrajectoryError(process.stderr or f"not a regular file: {path}")
        text, truncated = self._cap(process.stdout)
        return {
            "path": path.removeprefix(self.root).lstrip("/"),
            "start_line": start,
            "end_line": end,
            "content": text,
            "truncated": truncated,
        }

    def _search_code(self, args: Mapping[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        if not query or len(query) > 1000:
            raise TrajectoryError("query must contain 1-1000 characters")
        path = self._path(args.get("path", "."))
        glob = str(args.get("glob") or "")
        script = (
            'if test -n "$3"; then '
            'grep -R -F -n -m 200 --exclude-dir=.git --include="$3" -- "$1" "$2"; '
            "else grep -R -F -n -m 200 --exclude-dir=.git -- \"$1\" \"$2\"; fi"
        )
        process = self._exec(
            ["sh", "-c", script, "specforge", query, path, glob],
            timeout=min(60.0, self.timeout_seconds),
        )
        output = process.stdout if process.returncode in (0, 1) else process.stderr
        text, truncated = self._cap(output)
        return {
            "query": query,
            "path": path.removeprefix(self.root).lstrip("/") or ".",
            "matches": text,
            "returncode": process.returncode,
            "truncated": truncated,
        }

    def _patch_paths(self, patch: str) -> list[str]:
        paths: list[str] = []
        for match in re.finditer(r"^(?:---|\+\+\+)\s+([^\t\n]+)", patch, re.MULTILINE):
            raw = match.group(1).strip()
            if raw == "/dev/null":
                continue
            if raw.startswith(("a/", "b/")):
                raw = raw[2:]
            self._path(raw)
            paths.append(raw)
        if not paths:
            raise TrajectoryError("patch contains no file headers")
        return paths

    def _apply_patch(self, args: Mapping[str, Any]) -> dict[str, Any]:
        patch = str(args.get("patch") or "")
        if not patch or len(patch) > 2_000_000:
            raise TrajectoryError("patch must contain 1-2,000,000 characters")
        paths = self._patch_paths(patch)
        base = [self.runtime, "exec", "-i", self.container, "git", "-C", self.root]
        check = _run(
            [*base, "apply", "--check", "--whitespace=nowarn", "-"],
            input_text=patch,
            timeout=min(60.0, self.timeout_seconds),
            check=False,
        )
        if check.returncode:
            text, truncated = self._cap(check.stderr or check.stdout)
            return {"ok": False, "error": text, "truncated": truncated}
        apply = _run(
            [*base, "apply", "--whitespace=nowarn", "-"],
            input_text=patch,
            timeout=min(60.0, self.timeout_seconds),
            check=False,
        )
        if apply.returncode:
            text, truncated = self._cap(apply.stderr or apply.stdout)
            return {"ok": False, "error": text, "truncated": truncated}
        return {"files": sorted(set(paths))}

    def _git_diff(self) -> dict[str, Any]:
        process = self._exec(
            ["git", "-C", self.root, "diff", "--no-ext-diff", "--"],
            timeout=min(60.0, self.timeout_seconds),
        )
        text, truncated = self._cap(
            process.stdout if process.returncode == 0 else process.stderr
        )
        return {
            "diff": text,
            "returncode": process.returncode,
            "truncated": truncated,
        }

    def _run_tests(self) -> dict[str, Any]:
        if not self.test_command:
            raise TrajectoryError("this task has no predefined test command")
        process = self._exec(
            [
                "sh",
                "-c",
                'cd "$1" && exec sh -lc "$2"',
                "specforge",
                self.root,
                self.test_command,
            ],
            timeout=self.timeout_seconds,
        )
        text, truncated = self._cap(process.stdout + process.stderr)
        return {
            "command": self.test_command,
            "returncode": process.returncode,
            "output": text,
            "truncated": truncated,
        }


class ContainerWorkspaceProvider:
    """Create one disposable container for an executable repository task."""

    def __init__(
        self,
        *,
        runtime: str = "docker",
        auto_pull: bool = True,
        network: str = "none",
        cpus: float | None = None,
        memory: str = "",
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.runtime = runtime
        self.auto_pull = auto_pull
        self.network = network
        self.cpus = cpus
        self.memory = memory
        self.timeout_seconds = timeout_seconds

    def _ensure_image(self, image: str) -> None:
        inspect = _run(
            [self.runtime, "image", "inspect", image],
            timeout=min(self.timeout_seconds, 300.0),
            check=False,
        )
        if inspect.returncode == 0:
            return
        if not self.auto_pull:
            raise TrajectoryError(f"container image is unavailable locally: {image}")
        _run(
            [self.runtime, "pull", image],
            timeout=self.timeout_seconds,
        )

    @contextmanager
    def acquire(self, item: ModelInput) -> Iterator[WorkspaceLease]:
        image = item.workspace_image.strip()
        if not image:
            raise TrajectoryError(
                f"executable workspace has no image_name: {item.id}"
            )
        self._ensure_image(image)
        name = "specforge-" + _safe_name(
            f"{item.id}-{uuid.uuid4().hex[:12]}", limit=45
        )
        command = [
            self.runtime,
            "create",
            "--name",
            name,
            "--network",
            self.network,
        ]
        if self.cpus is not None:
            command.extend(["--cpus", str(self.cpus)])
        if self.memory:
            command.extend(["--memory", self.memory])
        command.extend(
            [
                "--entrypoint",
                "/bin/sh",
                image,
                "-c",
                "while :; do sleep 3600; done",
            ]
        )
        _run(command, timeout=min(self.timeout_seconds, 300.0))
        try:
            _run([self.runtime, "start", name], timeout=300.0)
            _run(
                [self.runtime, "exec", name, "test", "-d", "/testbed"],
                timeout=60.0,
            )
            executor = ContainerWorkspaceToolExecutor(
                name,
                runtime=self.runtime,
                root="/testbed",
                test_command=item.workspace_test_command,
                timeout_seconds=self.timeout_seconds,
            )
            yield WorkspaceLease(
                executor=executor,
                mode="container",
                metadata={"image": image, "container": name, "root": "/testbed"},
            )
        finally:
            _run(
                [self.runtime, "rm", "-f", name],
                timeout=300.0,
                check=False,
            )


class AutomaticWorkspaceProvider:
    """Resolve manual overrides, seed files, Git worktrees, and containers."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        workspace_map: Mapping[str, Mapping[str, Any]] | None = None,
        repo_url_template: str = "https://github.com/{repo}.git",
        container_runtime: str = "docker",
        container_auto_pull: bool = True,
        container_network: str = "none",
        container_cpus: float | None = None,
        container_memory: str = "",
        allow_host_tests: bool = False,
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.root = Path(cache_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace_map = dict(workspace_map or {})
        self.allow_host_tests = allow_host_tests
        self.timeout_seconds = timeout_seconds
        self.git = GitWorkspaceProvider(
            self.root,
            repo_url_template=repo_url_template,
            timeout_seconds=timeout_seconds,
            allow_host_tests=allow_host_tests,
        )
        self.container = ContainerWorkspaceProvider(
            runtime=container_runtime,
            auto_pull=container_auto_pull,
            network=container_network,
            cpus=container_cpus,
            memory=container_memory,
            timeout_seconds=timeout_seconds,
        )

    @contextmanager
    def acquire(self, item: ModelInput) -> Iterator[WorkspaceLease | None]:
        mapped = self.workspace_map.get(item.id)
        if mapped:
            path = Path(str(mapped.get("workspace") or "")).expanduser().resolve()
            tests = tuple(str(value) for value in (mapped.get("test_argv") or ()))
            yield WorkspaceLease(
                executor=WorkspaceToolExecutor(
                    path,
                    test_argv=tests,
                    allow_host_tests=self.allow_host_tests,
                    timeout_seconds=self.timeout_seconds,
                ),
                mode="mapped-host",
                metadata={"workspace": str(path)},
            )
            return
        if item.workspace_seed_files:
            seed_root = self.root / "seed-runs"
            seed_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=_safe_name(item.id) + "-", dir=seed_root
            ) as temp_name:
                workspace = _seed_workspace(Path(temp_name), item)
                yield WorkspaceLease(
                    executor=WorkspaceToolExecutor(
                        workspace,
                        allow_host_tests=self.allow_host_tests,
                        timeout_seconds=self.timeout_seconds,
                    ),
                    mode="temporary-file",
                    metadata={"workspace": str(workspace)},
                )
            return
        if not item.workspace_required:
            yield None
            return
        if item.input_kind in GIT_WORKSPACE_KINDS:
            with self.git.acquire(item) as lease:
                yield lease
            return
        if item.input_kind in CONTAINER_WORKSPACE_KINDS:
            with self.container.acquire(item) as lease:
                yield lease
            return
        if item.input_kind in ORIGINAL_TRAJECTORY_KINDS:
            raise TrajectoryError(
                "trajectory_prefix_requires_original_trajectory_store"
            )
        raise TrajectoryError(
            f"no automatic workspace route for input_kind={item.input_kind!r}"
        )
