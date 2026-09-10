"""The MCP bridge, tested against an in-process server — no network, no credentials."""

import asyncio

from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from harness.executor import StreamingToolExecutor
from harness.events import ToolCallReady
from harness.mcp import (
    MCPBridge,
    MCPServerConfig,
    load_servers,
    qualified_name,
    sanitize_schema,
)
from harness.permissions import PermissionGate, PermissionPolicy
from harness.tools import ToolRegistry


def build_server() -> MCPServer:
    server = MCPServer("triplewhale")

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def get_metrics(window: str) -> str:
        """Fetch performance metrics for a window."""
        return f"metrics for {window}"

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def create_report(title: str) -> str:
        """Create a report. Writes."""
        return f"created {title}"

    @server.tool()
    def unannotated() -> str:
        """No annotations at all."""
        return "ok"

    return server


async def discover_from(server: MCPServer, name: str = "triplewhale"):
    """Wrap an in-process server's tools exactly as a remote one would be."""
    async with Client(server) as client:
        bridge = MCPBridge(servers=[])
        bridge.clients[name] = client
        return await bridge.discover()


def tools_by_name(tools):
    return {t.name: t for t in tools}


# ---- discovery and naming ----------------------------------------------------


def test_tools_are_namespaced_by_server():
    tools = tools_by_name(asyncio.run(discover_from(build_server())))
    assert "mcp__triplewhale__get_metrics" in tools
    assert "mcp__triplewhale__create_report" in tools


def test_qualified_name_matches_the_permission_pattern_shape():
    name = qualified_name("triplewhale", "get_metrics")
    policy = PermissionPolicy(allow=("mcp__triplewhale__*",))
    assert policy.evaluate(name).decision == "allow"


def test_description_and_schema_survive():
    tools = tools_by_name(asyncio.run(discover_from(build_server())))
    metrics = tools["mcp__triplewhale__get_metrics"]

    assert "performance metrics" in metrics.description
    assert metrics.parameters["properties"]["window"]["type"] == "string"
    assert metrics.parameters["required"] == ["window"]


# ---- concurrency classification ----------------------------------------------


def test_read_only_hint_makes_a_tool_concurrency_safe():
    tools = tools_by_name(asyncio.run(discover_from(build_server())))
    assert tools["mcp__triplewhale__get_metrics"].concurrency_safe is True


def test_a_writing_tool_is_not_safe():
    tools = tools_by_name(asyncio.run(discover_from(build_server())))
    assert tools["mcp__triplewhale__create_report"].concurrency_safe is False


def test_unannotated_tools_are_treated_as_unsafe():
    """An unannotated tool might write. Running it alone is the cheap mistake."""
    tools = tools_by_name(asyncio.run(discover_from(build_server())))
    assert tools["mcp__triplewhale__unannotated"].concurrency_safe is False


# ---- schema sanitizing -------------------------------------------------------


def test_sanitize_drops_keywords_gemini_rejects():
    cleaned = sanitize_schema(
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "q": {"type": "string", "format": "uri", "default": "x"},
                "n": {"type": "integer", "exclusiveMinimum": 0},
            },
            "required": ["q"],
        }
    )
    assert cleaned == {
        "type": "object",
        "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["q"],
    }


def test_sanitize_recurses_into_arrays():
    cleaned = sanitize_schema(
        {"type": "array", "items": {"type": "string", "format": "date"}, "minItems": 1}
    )
    assert cleaned == {"type": "array", "items": {"type": "string"}}


def test_sanitize_keeps_enum_and_description():
    cleaned = sanitize_schema(
        {"type": "string", "enum": ["a", "b"], "description": "pick one", "title": "X"}
    )
    assert cleaned == {"type": "string", "enum": ["a", "b"], "description": "pick one"}


def test_sanitize_survives_a_missing_schema():
    assert sanitize_schema(None) == {"type": "object", "properties": {}, "required": []}


# ---- execution through the real executor -------------------------------------


def call(call_id: str, name: str, **arguments) -> ToolCallReady:
    return ToolCallReady(index=0, call_id=call_id, name=name, arguments=arguments)


def test_a_bridged_tool_runs_through_the_executor_and_ledger():
    async def scenario():
        async with Client(build_server()) as client:
            bridge = MCPBridge(servers=[])
            bridge.clients["triplewhale"] = client
            registry = ToolRegistry(await bridge.discover())

            ex = StreamingToolExecutor(
                registry, gate=PermissionGate(PermissionPolicy(default="allow"))
            )
            ex.submit(call("c0", "mcp__triplewhale__get_metrics", window="7d"))
            return await ex.drain()

    outcomes = asyncio.run(scenario())
    assert len(outcomes) == 1
    assert not outcomes[0].is_error
    assert "7d" in outcomes[0].result


def test_permissions_gate_bridged_tools_like_any_other():
    async def scenario():
        async with Client(build_server()) as client:
            bridge = MCPBridge(servers=[])
            bridge.clients["triplewhale"] = client
            registry = ToolRegistry(await bridge.discover())

            # Read tools allowed by prefix, writes denied by prefix — the shape a real
            # permissions.toml would take for these servers.
            policy = PermissionPolicy(
                allow=("mcp__triplewhale__get_*",),
                deny=("mcp__triplewhale__create_*",),
            )
            ex = StreamingToolExecutor(registry, gate=PermissionGate(policy))
            ex.submit(call("c0", "mcp__triplewhale__create_report", title="x"))
            ex.submit(call("c1", "mcp__triplewhale__get_metrics", window="7d"))
            return await ex.drain()

    denied, allowed = asyncio.run(scenario())
    assert denied.reason == "denied"
    assert not allowed.is_error, "the deny pattern must not catch the read tool"
    assert "7d" in allowed.result


def test_an_unmatched_bridged_tool_falls_to_the_policy_default():
    """A newly discovered MCP tool nobody has written a rule for must not just run."""
    policy = PermissionPolicy(allow=("mcp__triplewhale__get_*",), default="ask")
    assert policy.evaluate("mcp__triplewhale__brand_new_tool").decision == "ask"


# ---- configuration -----------------------------------------------------------


def test_missing_config_means_no_servers(tmp_path):
    assert load_servers(tmp_path / "nope.toml") == []


def test_config_loads_servers(tmp_path):
    path = tmp_path / "mcp.toml"
    path.write_text(
        '[servers.triplewhale]\nurl = "https://mcp.triplewhale.com/v1/mcp"\nauth = "oauth"\n'
        '[servers.other]\nurl = "https://x/mcp"\nauth = "header"\n'
        'header_name = "x-api-key"\napi_key_env = "OTHER_KEY"\nenabled = false\n'
    )
    servers = {s.name: s for s in load_servers(path)}

    assert servers["triplewhale"].auth == "oauth"
    assert servers["other"].api_key_env == "OTHER_KEY"
    assert servers["other"].enabled is False


def test_the_shipped_config_names_both_servers():
    servers = {s.name: s for s in load_servers()}
    assert "triplewhale" in servers and "atria" in servers
    assert servers["triplewhale"].url == "https://mcp.triplewhale.com/v1/mcp"
    assert servers["atria"].url == "https://api.tryatria.com/mcp"


def test_tokens_land_in_the_gitignored_auth_dir():
    config = MCPServerConfig(name="triplewhale", url="https://x/mcp")
    assert config.token_path.name == "triplewhale.json"
    assert config.token_path.parent.name == ".mcp-auth"
