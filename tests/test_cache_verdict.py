"""`harness prompt` cannot connect to MCP servers, so it must not judge the cache floor
as if their declarations did not exist. That mistake once reported MISS on a prefix that
clears the floor four times over."""

import io
from contextlib import redirect_stdout

from harness import cli
from harness.mcp import MCPServerConfig
from harness.prompt import build_effective_system_prompt


class FakeCounter:
    """Stands in for the live token counter.

    The ratio is a parameter, not the real one: these tests are about which verdict the
    shortfall produces, and must not fail every time the prompt files change size.
    """

    model = "gemini-3.8-flash"  # floor 4,096
    note = None

    def __init__(self, chars_per_token=400):
        self.chars_per_token = chars_per_token

    def tokens(self, text):
        return len(text) // self.chars_per_token if text else None

    def measure(self, text):
        count = self.tokens(text)
        return "0 tok" if count is None else f"{count:,} tok"


def verdict(monkeypatch, servers, chars_per_token=400):
    monkeypatch.setattr(cli, "load_servers", lambda *a, **k: servers)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        cli._render_prompt(build_effective_system_prompt(), FakeCounter(chars_per_token))
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


def test_a_prefix_over_the_floor_is_ok_regardless_of_servers(monkeypatch):
    line = verdict(monkeypatch, [MCPServerConfig(name="atria", url="https://x/mcp")],
                   chars_per_token=1)
    assert "OK" in line
