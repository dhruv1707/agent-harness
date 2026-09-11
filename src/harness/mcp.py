"""The MCP bridge: remote servers become ordinary local tools.

The Gemini Interactions API can declare an MCP server natively — `{"type": "mcp_server",
"url": ...}` — and call it itself. We deliberately do not. Those calls execute on Google's
side and arrive in our stream as `mcp_server_tool_call` steps *after the fact*, so the
permission gate never gets a vote. That contradicts chapter 4's whole premise: the model
proposes, the runtime authorizes.

So the harness is the MCP client. Every remote tool is registered as an ordinary function
tool named `mcp__<server>__<tool>`, which means it flows through the same permission gate,
the same concurrency partitioning, and the same ledger as anything local.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tomllib
import webbrowser
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from mcp import Client
from mcp.client.auth import AuthorizationCodeResult, OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.shared.auth import OAuthClientMetadata, OAuthClientInformationFull, OAuthToken

from .config import AGENT_DIR, ROOT
from .permissions import matches_pattern
from .tools import Tool

#: Where OAuth tokens land. Gitignored — these are credentials.
AUTH_DIR = ROOT / ".mcp-auth"

#: Loopback port for the OAuth redirect. Must match the registered redirect URI.
CALLBACK_PORT = 8765
CALLBACK_URI = f"http://localhost:{CALLBACK_PORT}/callback"

#: JSON Schema keywords Gemini accepts. Everything else is dropped rather than guessed at:
#: MCP servers emit full JSON Schema, and google-genai ships its own filter for exactly
#: this reason.
_ALLOWED_SCHEMA_KEYS = {
    "type",
    "description",
    "properties",
    "required",
    "items",
    "enum",
    "nullable",
}


# ---- configuration -----------------------------------------------------------


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    url: str
    auth: str = "oauth"  # oauth | header | none
    header_name: str = "x-api-key"
    api_key_env: str | None = None
    enabled: bool = True
    #: Whether to enforce the server's declared output schema on responses. Turn this off
    #: for a server whose schema contradicts its own behaviour — the SDK discards the
    #: entire response over a single violation, which is worse than not checking.
    validate_output: bool = True
    #: Which of the server's tools to load. Empty means all of them. Names or `prefix*`.
    #: Every declaration costs prefix tokens on every request and competes for the
    #: model's attention, so a big server is worth narrowing to the job.
    tools: tuple[str, ...] = ()

    @property
    def token_path(self) -> Path:
        return AUTH_DIR / f"{self.name}.json"


def load_servers(path: Path | None = None) -> list[MCPServerConfig]:
    """Read `agent/mcp.toml`. A missing file simply means no MCP servers."""
    path = path or (AGENT_DIR / "mcp.toml")
    if not path.is_file():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return [
        MCPServerConfig(
            name=name,
            url=entry["url"],
            auth=entry.get("auth", "oauth"),
            header_name=entry.get("header_name", "x-api-key"),
            api_key_env=entry.get("api_key_env"),
            enabled=entry.get("enabled", True),
            validate_output=entry.get("validate_output", True),
            tools=tuple(entry.get("tools", ())),
        )
        for name, entry in (data.get("servers") or {}).items()
    ]


# ---- OAuth -------------------------------------------------------------------


class FileTokenStorage(TokenStorage):
    """Tokens on disk so a browser sign-in survives the process exiting."""

    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> dict:
        if not self.path.is_file():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.path.chmod(0o600)

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._read().get("client")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read()
        data["client"] = client_info.model_dump(mode="json", exclude_none=True)
        self._write(data)


async def _open_browser(authorization_url: str) -> None:
    """Open the authorize page — but only if someone is there to complete it.

    Tokens refresh lazily, so an expiry can land in the middle of any run. Launching a
    browser nobody is watching would hang an unattended job on a prompt that will never be
    answered; failing with the command to run is the honest outcome.
    """
    if not sys.stdin.isatty():
        raise RuntimeError(
            "this MCP server needs authorization and there is no terminal to do it in — "
            "run `harness mcp <server>` interactively to sign in, then retry"
        )
    print(f"\n  opening browser to authorize:\n  {authorization_url}\n", flush=True)
    webbrowser.open(authorization_url)


async def _await_callback() -> AuthorizationCodeResult:
    """Catch the loopback redirect on one connection, then shut the listener down."""
    received: dict[str, str] = {}
    done = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = await reader.readline()
        target = request.decode("latin-1").split(" ")[1] if b" " in request else "/"
        query = parse_qs(urlparse(target).query)
        for key in ("code", "state", "iss", "error"):
            if key in query:
                received[key] = query[key][0]

        body = (
            b"<html><body><h3>Authorized. You can close this tab.</h3></body></html>"
            if "code" in received
            else b"<html><body><h3>Authorization failed.</h3></body></html>"
        )
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )
        await writer.drain()
        writer.close()
        done.set()

    server = await asyncio.start_server(handle, "localhost", CALLBACK_PORT)
    async with server:
        await done.wait()

    if "code" not in received:
        raise RuntimeError(
            f"authorization failed: {received.get('error', 'no code in the redirect')}"
        )
    return AuthorizationCodeResult(
        code=received["code"], state=received.get("state"), iss=received.get("iss")
    )


async def _validate_same_origin(server_url: str, prm_resource: str | None) -> None:
    """Accept a resource identifier that differs only in path, reject a different host.

    Triple Whale's Protected Resource Metadata advertises
    `https://mcp.triplewhale.com/sse` while its documented endpoint is
    `https://mcp.triplewhale.com/v1/mcp`. RFC 8707 validation rejects that outright, and
    the mismatch is the server's, not ours.

    The check still matters though: its real job is stopping a hostile PRM from pointing
    the token's audience at a *different* host, which would hand our credential to someone
    else. So this relaxes the path comparison and keeps the origin comparison.
    """
    if prm_resource is None:
        return
    server = urlparse(server_url)
    resource = urlparse(prm_resource)
    if (server.scheme, server.hostname, server.port) != (
        resource.scheme,
        resource.hostname,
        resource.port,
    ):
        raise RuntimeError(
            f"protected resource {prm_resource} is on a different origin than "
            f"{server_url} — refusing to authorize"
        )


def _oauth_provider(server: MCPServerConfig) -> OAuthClientProvider:
    return OAuthClientProvider(
        server_url=server.url,
        client_metadata=OAuthClientMetadata(
            client_name="agent-harness",
            redirect_uris=[CALLBACK_URI],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=FileTokenStorage(server.token_path),
        redirect_handler=_open_browser,
        callback_handler=_await_callback,
        validate_resource_url=_validate_same_origin,
    )


# ---- schema conversion -------------------------------------------------------


def sanitize_schema(schema: Any) -> dict:
    """Reduce a JSON Schema to what Gemini will accept.

    Dropping unknown keywords beats enumerating rejected ones: MCP servers emit full JSON
    Schema (`$ref`, `anyOf`, `format`, `default`, …) and a rejected declaration fails the
    whole request, not just the tool.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}

    cleaned: dict = {}
    for key, value in schema.items():
        if key not in _ALLOWED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {k: sanitize_schema(v) for k, v in value.items()}
        elif key == "items":
            cleaned[key] = sanitize_schema(value)
        else:
            cleaned[key] = value

    cleaned.setdefault("type", "object")
    if cleaned["type"] == "object":
        cleaned.setdefault("properties", {})
        cleaned.setdefault("required", [])
    return cleaned


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first present attribute.

    MCP 2.x renamed the wire names to snake_case (`input_schema`, `read_only_hint`,
    `is_error`). Accepting both spellings keeps this working against a 1.x server without
    a silent wrong answer — reading `isError` off a 2.x result returns False and reports
    every failure as a success.
    """
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _is_read_only(mcp_tool: Any) -> bool:
    """Trust the server's read-only hint; treat anything unannotated as unsafe.

    An unannotated tool might write, and the cost of being wrong in that direction is a
    race on someone's ad account. The cost of being wrong the other way is that it runs
    alone.
    """
    annotations = getattr(mcp_tool, "annotations", None)
    return bool(_field(annotations, "read_only_hint", "readOnlyHint", default=False))


def qualified_name(server: str, tool: str) -> str:
    """`mcp__triplewhale__get_metrics` — the shape the permission patterns expect."""
    return f"mcp__{server}__{tool}"


# ---- the bridge --------------------------------------------------------------


@dataclass
class MCPBridge:
    """Holds live connections to MCP servers for the length of a session."""

    servers: list[MCPServerConfig] = field(default_factory=load_servers)
    clients: dict[str, Client] = field(default_factory=dict, init=False)
    #: Servers that would not connect, name -> why. A dead source degrades the run; it
    #: does not end it. The same rule the agent's own prompt gives it about unreachable
    #: data sources applies to the runtime that feeds it.
    failures: dict[str, str] = field(default_factory=dict, init=False)
    #: Each server's own guidance, published at connect. Atria's explains things no tool
    #: description does — that a malformed id returns an empty result rather than an
    #: error, which otherwise reads as "nothing there" when it means "wrong key".
    instructions: dict[str, str] = field(default_factory=dict, init=False)
    _stack: AsyncExitStack | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> MCPBridge:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        # No `enabled` check here on purpose. Handing a server to the bridge is the
        # instruction to connect it; deciding *which* servers to hand over belongs to the
        # caller. Filtering in both places let a server be silently dropped — no tools and
        # no error, the most confusing failure there is.
        for server in self.servers:
            try:
                await self._connect(server)
            except Exception as exc:
                self.failures[server.name] = f"{type(exc).__name__}: {exc}"
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(*exc)
            self._stack = None
        self.clients.clear()

    async def _connect(self, server: MCPServerConfig) -> None:
        headers: dict[str, str] = {}
        auth = None

        if server.auth == "header":
            key = os.environ.get(server.api_key_env or "")
            if not key:
                raise RuntimeError(
                    f"{server.name}: auth is 'header' but {server.api_key_env} is unset"
                )
            headers[server.header_name] = key
        elif server.auth == "oauth":
            auth = _oauth_provider(server)

        http_client = create_mcp_http_client(headers=headers or None, auth=auth)
        transport = streamable_http_client(server.url, http_client=http_client)
        client = await self._stack.enter_async_context(Client(transport))
        self.clients[server.name] = client
        published = getattr(client, "instructions", None)
        if published:
            self.instructions[server.name] = published

    async def discover(self) -> list[Tool]:
        """Every connected server's tools, wrapped as ordinary local tools."""
        discovered: list[Tool] = []
        for name, client in self.clients.items():
            config = next((s for s in self.servers if s.name == name), None)
            wanted = config.tools if config else ()
            listing = await client.list_tools()
            for mcp_tool in listing.tools:
                if wanted and not any(matches_pattern(p, mcp_tool.name) for p in wanted):
                    continue
                discovered.append(self._wrap(name, client, mcp_tool))

            if config is not None and not config.validate_output:
                _disable_output_validation(client, name)
        return discovered

    def _wrap(self, server_name: str, client: Client, mcp_tool: Any) -> Tool:
        tool_name = mcp_tool.name

        async def call(**arguments) -> str:
            result = await client.call_tool(tool_name, arguments)
            return _render_result(result)

        call.__name__ = qualified_name(server_name, tool_name)

        return Tool(
            name=qualified_name(server_name, tool_name),
            description=(mcp_tool.description or "").strip(),
            parameters=sanitize_schema(_field(mcp_tool, "input_schema", "inputSchema")),
            fn=call,
            concurrency_safe=_is_read_only(mcp_tool),
            interrupt_behavior="cancel",
            wants_context=False,
        )


def _disable_output_validation(client: Client, server_name: str) -> None:
    """Stop the SDK discarding a whole response over one output-schema violation.

    Atria declares its metric values as `number` but returns `null` where a metric has no
    data — which its own tool descriptions state plainly ("`null` means no data for that
    metric in the window"). The schema contradicts the documented behaviour, and the
    client's reaction is to throw the entire response away, which took out
    `list_ad_account_creative_tags` completely.

    The cache is private to the SDK session, so this is deliberately best-effort: if the
    attribute moves in a future version we lose the workaround, not the connection.
    """
    session = getattr(client, "session", None)
    schemas = getattr(session, "_tool_output_schemas", None)
    if isinstance(schemas, dict):
        for tool_name in list(schemas):
            schemas[tool_name] = None


def _render_result(result: Any) -> str:
    """Flatten an MCP tool result into text the model can read."""
    if _field(result, "is_error", "isError", default=False):
        return f"tool error: {_render_content(result)}"
    structured = _field(result, "structured_content", "structuredContent")
    if structured:
        return json.dumps(structured, indent=2, default=str)
    return _render_content(result)


def _render_content(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else json.dumps(block, default=str))
    return "\n".join(parts) if parts else "(no content)"
