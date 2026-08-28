"""Structured web-search and browser tools for coding-agent trajectories.

The public function schemas match the online deployment.  Search and page-fetch
backends are private runtime choices, so trajectories generated with a local
SearXNG instance can later be replayed against a hosted provider unchanged.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Mapping, Sequence
from urllib.parse import urljoin, urlsplit

import requests


WEB_SEARCH_TOOL_NAME = "web_search"
BROWSER_TOOL_NAME = "browser"
WEB_TOOL_NAMES = frozenset({WEB_SEARCH_TOOL_NAME, BROWSER_TOOL_NAME})
TOOL_RESPONSE_KEYS = (
    "command",
    "stdout",
    "stderr",
    "exit_code",
    "timed_out",
    "truncated",
    "information_lines",
)
DEFAULT_SEARXNG_ENDPOINT = "http://127.0.0.1:8080/search"
DEFAULT_SERPER_ENDPOINT = "https://google.serper.dev/search"
DEFAULT_SERPER_SCRAPE_ENDPOINT = "https://scrape.serper.dev"
_LOCALE = re.compile(r"\A[A-Za-z]{2}\Z")


WEB_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": WEB_SEARCH_TOOL_NAME,
            "description": (
                "Search the live public web for current or external evidence. "
                "Use browser on a returned URL before relying on a search snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 500},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                    },
                    "timeout": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 60,
                    },
                    "gl": {
                        "type": "string",
                        "description": "Two-letter country code.",
                        "pattern": "^[A-Za-z]{2}$",
                    },
                    "hl": {
                        "type": "string",
                        "description": "Two-letter language code.",
                        "pattern": "^[A-Za-z]{2}$",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": BROWSER_TOOL_NAME,
            "description": (
                "Open a public HTTP(S) source page and return bounded cleaned text. "
                "Use it to verify URLs discovered by web_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "timeout": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 60,
                    },
                    "max_length": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 32000,
                        "default": 8000,
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
]


WEB_RESEARCH_SYSTEM_APPENDIX = """Live web tools are available for this task.
Use web_search when the answer depends on external documentation, releases,
issues, compatibility, or other information not contained in the prompt. Treat
search snippets only as discovery: after a successful search, open at least one
relevant returned URL with browser before reaching a conclusion. Prefer primary
sources such as official documentation, release notes, and project repositories.
Never invent tool results, and include the source URLs used in the final answer."""


class WebToolError(ValueError):
    """A model-controlled web-tool payload violates the public contract."""


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._title_depth = 0
        self.parts: list[str] = []
        self.title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript", "svg", "canvas"}:
            self._ignored_depth += 1
        elif tag == "title":
            self._title_depth += 1
        elif tag in {"p", "br", "li", "tr", "h1", "h2", "h3", "h4", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "canvas"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        elif tag == "title":
            self._title_depth = max(0, self._title_depth - 1)
        elif tag in {"p", "li", "tr", "h1", "h2", "h3", "h4", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        self.parts.append(data)
        if self._title_depth:
            self.title_parts.append(data)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _canonical_search_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"query", "top_k", "timeout", "gl", "hl"}
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise WebToolError("unsupported web_search arguments: " + ", ".join(unknown))
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise WebToolError("web_search query must be a nonempty string")
    query = query.strip()
    if len(query) > 500:
        raise WebToolError("web_search query exceeds 500 characters")
    result: dict[str, Any] = {"query": query}
    top_k = arguments.get("top_k")
    if top_k is not None:
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 10
        ):
            raise WebToolError("web_search top_k must be an integer from 1 to 10")
        result["top_k"] = top_k
    timeout = arguments.get("timeout")
    if timeout is not None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or not 1 <= float(timeout) <= 60
        ):
            raise WebToolError("web_search timeout must be between 1 and 60 seconds")
        result["timeout"] = float(timeout)
    for name in ("gl", "hl"):
        value = arguments.get(name)
        if value is not None:
            if not isinstance(value, str) or _LOCALE.fullmatch(value) is None:
                raise WebToolError(f"web_search {name} must be a two-letter code")
            result[name] = value.lower()
    return result


def _canonical_browser_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"url", "timeout", "max_length"}
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise WebToolError("unsupported browser arguments: " + ", ".join(unknown))
    raw_url = arguments.get("url")
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise WebToolError("browser url must be a nonempty string")
    url = raw_url.strip()
    if len(url) > 2000 or any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise WebToolError("browser url is too long or contains control characters")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.fragment)
    ):
        raise WebToolError("browser requires a public HTTP(S) URL without credentials")
    result: dict[str, Any] = {"url": url}
    timeout = arguments.get("timeout")
    if timeout is not None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or not 1 <= float(timeout) <= 60
        ):
            raise WebToolError("browser timeout must be between 1 and 60 seconds")
        result["timeout"] = float(timeout)
    max_length = arguments.get("max_length")
    if (
        max_length is not None
        and (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or not 1 <= max_length <= 32000
        )
    ):
        raise WebToolError("browser max_length must be an integer from 1 to 32000")
    if max_length is not None:
        result["max_length"] = max_length
    return result


def _command(name: str, arguments: Mapping[str, Any]) -> str:
    return name + " " + json.dumps(
        dict(arguments), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _tool_result(
    command: str,
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    timed_out: bool = False,
    truncated: bool = False,
) -> dict[str, Any]:
    return {
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "truncated": truncated,
        "information_lines": [line for line in stdout.splitlines() if line.strip()],
    }


def _clean_space(text: str) -> str:
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _public_url(url: str) -> None:
    hostname = (urlsplit(url).hostname or "").lower().rstrip(".")
    if not hostname or hostname == "localhost" or hostname.endswith(".local"):
        raise WebToolError("browser must not target a local host")
    try:
        literal_address = ipaddress.ip_address(hostname)
    except ValueError:
        literal_address = None
    if literal_address is not None:
        if not literal_address.is_global:
            raise WebToolError("browser must not target private or reserved addresses")
        return
    try:
        addresses = {
            item[4][0] for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as exc:
        raise WebToolError(f"browser hostname cannot be resolved: {hostname}") from exc
    if not addresses:
        raise WebToolError(f"browser hostname cannot be resolved: {hostname}")
    for raw in addresses:
        address = ipaddress.ip_address(raw)
        if not address.is_global:
            raise WebToolError("browser must not target private or reserved addresses")


@dataclass(frozen=True)
class WebToolConfig:
    search_provider: str = "searxng"
    search_endpoint: str = DEFAULT_SEARXNG_ENDPOINT
    browser_provider: str = "direct"
    browser_endpoint: str = DEFAULT_SERPER_SCRAPE_ENDPOINT
    api_key: str = ""
    search_timeout: float = 15.0
    browser_timeout: float = 15.0
    search_max_calls: int = 2
    browser_max_calls: int = 8
    browser_max_length: int = 8000
    max_output_chars: int = 32768
    max_download_bytes: int = 2 << 20


class WebToolClient:
    """Reusable HTTP client; episode call budgets live in WebToolExecutor."""

    def __init__(self, config: WebToolConfig) -> None:
        if config.search_provider not in {"searxng", "serper"}:
            raise ValueError("search_provider must be searxng or serper")
        if config.browser_provider not in {"direct", "serper"}:
            raise ValueError("browser_provider must be direct or serper")
        if (config.search_provider == "serper" or config.browser_provider == "serper") and not config.api_key:
            raise ValueError("Serper providers require an API key")
        for name, value in (
            ("search_timeout", config.search_timeout),
            ("browser_timeout", config.browser_timeout),
        ):
            if not 1 <= value <= 60:
                raise ValueError(f"{name} must be between 1 and 60 seconds")
        if not 1 <= config.browser_max_length <= 32000:
            raise ValueError("browser_max_length must be between 1 and 32000")
        if config.max_output_chars < 1024 or config.max_download_bytes < 1024:
            raise ValueError("web output and download limits must be at least 1024")
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "SpecForge-WebTrajectory/1.0", "Accept": "application/json,text/html,text/plain"}
        )

    def close(self) -> None:
        self.session.close()

    def search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            args = _canonical_search_arguments(arguments)
        except WebToolError as exc:
            return _tool_result(
                _command(WEB_SEARCH_TOOL_NAME, arguments),
                stderr=f"validation error: {exc}",
                exit_code=-2,
            )
        command = _command(WEB_SEARCH_TOOL_NAME, args)
        timeout = min(float(args.get("timeout", self.config.search_timeout)), self.config.search_timeout)
        try:
            raw = self._searxng(args, timeout) if self.config.search_provider == "searxng" else self._serper_search(args, timeout)
            results = self._normalize_search(raw, int(args.get("top_k", 5)))
            stdout, truncated = self._search_stdout(args["query"], results)
            return _tool_result(command, stdout=stdout, truncated=truncated)
        except requests.Timeout:
            message = f"[web_search timed out after {timeout}s]"
            return _tool_result(command, stdout=message, exit_code=-1, timed_out=True)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            return _tool_result(command, stderr=self._error("web_search", exc), exit_code=1)

    def browse(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            args = _canonical_browser_arguments(arguments)
            _public_url(args["url"])
        except WebToolError as exc:
            return _tool_result(
                _command(BROWSER_TOOL_NAME, arguments),
                stderr=f"validation error: {exc}",
                exit_code=-2,
            )
        command = _command(BROWSER_TOOL_NAME, args)
        timeout = min(float(args.get("timeout", self.config.browser_timeout)), self.config.browser_timeout)
        max_length = min(
            int(args.get("max_length", 8000)), self.config.browser_max_length
        )
        try:
            page = self._direct_browser(args["url"], timeout, max_length) if self.config.browser_provider == "direct" else self._serper_browser(args["url"], timeout, max_length)
            stdout = _compact_json(page)
            truncated = bool(page.get("character_truncated"))
            if len(stdout) > self.config.max_output_chars:
                page["text"] = str(page.get("text", ""))[: max(0, self.config.max_output_chars // 2)]
                page["character_truncated"] = True
                stdout = _compact_json(page)
                truncated = True
            return _tool_result(command, stdout=stdout, truncated=truncated)
        except requests.Timeout:
            message = f"[browser timed out after {timeout}s]"
            return _tool_result(command, stdout=message, exit_code=-1, timed_out=True)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            return _tool_result(command, stderr=self._error("browser", exc), exit_code=1)

    def _searxng(self, args: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
        params: dict[str, Any] = {
            "q": args["query"],
            "format": "json",
            "safesearch": 0,
        }
        if args.get("hl"):
            params["language"] = args["hl"]
        response = self.session.get(
            self.config.search_endpoint,
            params=params,
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, Mapping):
            raise RuntimeError("SearXNG returned non-object JSON")
        return raw

    def _serper_search(self, args: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
        payload: dict[str, Any] = {"q": args["query"], "num": args["top_k"]}
        for name in ("gl", "hl"):
            if args.get(name):
                payload[name] = args[name]
        response = self.session.post(
            self.config.search_endpoint,
            headers={"X-API-KEY": self.config.api_key},
            json=payload,
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, Mapping):
            raise RuntimeError("Serper returned non-object JSON")
        return raw

    def _normalize_search(self, raw: Mapping[str, Any], top_k: int) -> list[dict[str, Any]]:
        if self.config.search_provider == "searxng":
            source_items = raw.get("results") or []
            normalized = []
            for index, item in enumerate(source_items, 1):
                if not isinstance(item, Mapping):
                    continue
                engines = item.get("engines")
                engine_list = engines if isinstance(engines, Sequence) else ()
                normalized.append(
                    {
                        "rank": index,
                        "type": "organic",
                        "title": str(item.get("title") or "")[:500],
                        "link": str(item.get("url") or "")[:2000],
                        "snippet": str(item.get("content") or "")[:900],
                        "source": str(
                            item.get("engine")
                            or ",".join(str(engine) for engine in engine_list)
                        )[:300],
                    }
                )
        else:
            source_items = raw.get("organic") or raw.get("news") or []
            normalized = [
                {
                    "rank": int(item.get("position") or index),
                    "type": "organic",
                    "title": str(item.get("title") or "")[:500],
                    "link": str(item.get("link") or "")[:2000],
                    "snippet": str(item.get("snippet") or "")[:900],
                    "source": str(item.get("source") or item.get("displayedLink") or "")[:300],
                }
                for index, item in enumerate(source_items, 1)
                if isinstance(item, Mapping)
            ]
        return [item for item in normalized if item["link"].startswith(("http://", "https://"))][:top_k]

    def _search_stdout(self, query: str, results: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
        kept = [dict(item) for item in results]
        truncated = False
        while True:
            payload: dict[str, Any] = {"query": query, "results": kept}
            if kept:
                payload["next_action_required"] = {
                    "name": BROWSER_TOOL_NAME,
                    "arguments": {"url": kept[0]["link"]},
                    "reason": "verify the search evidence before answering",
                }
            if truncated:
                payload["truncated"] = True
            stdout = _compact_json(payload)
            if len(stdout) <= self.config.max_output_chars or not kept:
                return stdout, truncated
            kept.pop()
            truncated = True

    def _direct_browser(self, url: str, timeout: float, max_length: int) -> dict[str, Any]:
        current = url
        for _ in range(4):
            _public_url(current)
            with self.session.get(
                current,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            ) as response:
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise RuntimeError("browser redirect has no location")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                content_type = (
                    response.headers.get("content-type", "")
                    .split(";", 1)[0]
                    .lower()
                )
                if not (
                    content_type.startswith("text/")
                    or content_type
                    in {
                        "application/json",
                        "application/xml",
                        "application/xhtml+xml",
                    }
                ):
                    raise RuntimeError(
                        f"browser does not support content type {content_type!r}"
                    )
                chunks: list[bytes] = []
                size = 0
                download_truncated = False
                for chunk in response.iter_content(65536):
                    if not chunk:
                        continue
                    remaining = self.config.max_download_bytes - size
                    if remaining <= 0:
                        download_truncated = True
                        break
                    chunks.append(chunk[:remaining])
                    size += min(len(chunk), remaining)
                    if len(chunk) > remaining:
                        download_truncated = True
                        break
                encoding = response.encoding or "utf-8"
                raw_text = b"".join(chunks).decode(encoding, errors="replace")
                title = ""
                if "html" in content_type:
                    parser = _HTMLTextExtractor()
                    parser.feed(raw_text)
                    text = _clean_space("".join(parser.parts))
                    title = _clean_space("".join(parser.title_parts))[:500]
                else:
                    text = raw_text
                character_truncated = download_truncated or len(text) > max_length
                return {
                    "title": title,
                    "url": current,
                    "text": text[:max_length],
                    "source": "direct_http",
                    "content_type": content_type,
                    "status_code": response.status_code,
                    "character_truncated": character_truncated,
                }
        raise RuntimeError("browser exceeded three redirects")

    def _serper_browser(self, url: str, timeout: float, max_length: int) -> dict[str, Any]:
        response = self.session.post(
            self.config.browser_endpoint,
            headers={"X-API-KEY": self.config.api_key},
            json={"url": url, "includeMarkdown": True},
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, Mapping):
            raise RuntimeError("Serper Scrape returned non-object JSON")
        text = str(raw.get("markdown") or raw.get("text") or raw.get("content") or "")
        return {
            "title": str(raw.get("title") or "")[:500],
            "url": str(raw.get("url") or url),
            "text": text[:max_length],
            "source": "serper_scrape",
            "content_type": "application/json",
            "status_code": response.status_code,
            "character_truncated": len(text) > max_length,
        }

    def _error(self, tool: str, exc: Exception) -> str:
        diagnostic = f"{type(exc).__name__}: {exc}"
        if self.config.api_key:
            diagnostic = diagnostic.replace(self.config.api_key, "[REDACTED]")
        return f"{tool} error: {diagnostic}"


class WebToolExecutor:
    tool_names = WEB_TOOL_NAMES

    def __init__(self, client: WebToolClient, *, search_max_calls: int, browser_max_calls: int) -> None:
        if search_max_calls < 1 or browser_max_calls < 1:
            raise ValueError("web tool call budgets must be positive")
        self.client = client
        self.search_max_calls = search_max_calls
        self.browser_max_calls = browser_max_calls
        self.search_calls = 0
        self.browser_calls = 0

    def execute(self, call: Mapping[str, Any]) -> dict[str, Any]:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments") or {}
        if name == WEB_SEARCH_TOOL_NAME:
            if self.search_calls >= self.search_max_calls:
                return _tool_result(
                    _command(name, arguments),
                    stderr=f"web_search call budget exceeded: {self.search_calls}/{self.search_max_calls}",
                    exit_code=-2,
                )
            self.search_calls += 1
            return self.client.search(arguments)
        if name == BROWSER_TOOL_NAME:
            if self.browser_calls >= self.browser_max_calls:
                return _tool_result(
                    _command(name, arguments),
                    stderr=f"browser call budget exceeded: {self.browser_calls}/{self.browser_max_calls}",
                    exit_code=-2,
                )
            self.browser_calls += 1
            return self.client.browse(arguments)
        raise WebToolError(f"unknown web tool: {name!r}")
