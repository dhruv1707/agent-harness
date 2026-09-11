"""`harness prompt` cannot connect to MCP servers, so it must not judge the cache floor
as if their declarations did not exist. That mistake once reported MISS on a prefix that
clears the floor four times over."""

import io
from contextlib import redirect_stdout

from harness import cli
from harness.mcp import MCPServerConfig
from harness.prompt import build_effective_system_prompt


class FakeCounter:
    """Stands in for the live token counter: one token per four characters."""

    model = "gemini-3.8-flash"  # floor 4,096
    note = None

    def tokens(self, text):
        return len(text) // 4 if text else None

    def measure(self, text):
        count = self.tokens(text)
        return "0 tok" if count is None else f"{count:,} tok"


def verdict(monkeypatch, servers):
    monkeypatch.setattr(cli, "load_servers", lambda *a, **k: servers)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        cli._render_prompt(build_effective_system_prompt(), FakeCounter())
    # Two spaces: the label column, not the "CACHE BREAKPOINT" banner.
    return next(line for line in buffer.getvalue().splitlines() if line.startswith(" CACHE  "))


def test_a_configured_server_makes_the_shortfall_unverified(monkeypatch):
    line = verdict(monkeypatch, [MCPServerConfig(name="atria", url="https://x/mcp")])

    assert "UNVERIFIED" in line
    assert "MISS" not in line, "the missing tools are the reason, not a real regression"


def test_a_disabled_server_does_not_excuse_the_shortfall(monkeypatch):
    """Nothing will declare tools at run time either, so the floor really is missed."""
    disabled = MCPServerConfig(name="atria", url="https://x/mcp", enabled=False)
    assert "MISS" in verdict(monkeypatch, [disabled])


def test_no_servers_means_the_verdict_is_final(monkeypatch):
    assert "MISS" in verdict(monkeypatch, [])
