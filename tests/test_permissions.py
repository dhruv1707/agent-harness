"""Permission is a runtime object with three states, not a boolean."""

import asyncio

from harness.permissions import (
    PermissionGate,
    PermissionPolicy,
    PermissionResult,
)


def run(coro):
    return asyncio.run(coro)


# ---- policy evaluation -------------------------------------------------------


def test_exact_match_wins_its_decision():
    policy = PermissionPolicy(allow=("read_memory",), ask=("append_run_log",))
    assert policy.evaluate("read_memory").decision == "allow"
    assert policy.evaluate("append_run_log").decision == "ask"


def test_unmatched_tool_falls_to_default():
    policy = PermissionPolicy(allow=("read_memory",), default="ask")
    result = policy.evaluate("something_new")
    assert result.decision == "ask"
    assert "no rule matched" in result.reason
    assert result.rule is None


def test_most_restrictive_rule_wins():
    """A narrow allow cannot punch a hole in a broad deny."""
    policy = PermissionPolicy(allow=("dangerous_tool",), deny=("dangerous_*",))
    result = policy.evaluate("dangerous_tool")
    assert result.decision == "deny"


def test_ask_beats_allow():
    policy = PermissionPolicy(allow=("write_*",), ask=("write_file",))
    assert policy.evaluate("write_file").decision == "ask"


def test_deny_beats_ask_and_allow_together():
    policy = PermissionPolicy(
        allow=("mcp__*",), ask=("mcp__triplewhale__*",), deny=("mcp__triplewhale__delete",)
    )
    assert policy.evaluate("mcp__triplewhale__delete").decision == "deny"
    assert policy.evaluate("mcp__triplewhale__read").decision == "ask"
    assert policy.evaluate("mcp__atria__read").decision == "allow"


def test_wildcard_is_a_prefix_match():
    policy = PermissionPolicy(allow=("mcp__atria__*",))
    assert policy.evaluate("mcp__atria__search").decision == "allow"
    assert policy.evaluate("mcp__other__search").decision != "allow"


def test_the_matching_rule_is_recorded_for_audit():
    policy = PermissionPolicy(deny=("dangerous_*",))
    assert policy.evaluate("dangerous_thing").rule == "dangerous_*"


# ---- loading -----------------------------------------------------------------


def test_load_reads_a_policy_file(tmp_path):
    path = tmp_path / "permissions.toml"
    path.write_text(
        'default = "deny"\nallow = ["read_memory"]\nask = ["write_thing"]\ndeny = ["rm"]\n'
    )
    policy = PermissionPolicy.load(path)

    assert policy.default == "deny"
    assert policy.evaluate("read_memory").decision == "allow"
    assert policy.evaluate("write_thing").decision == "ask"
    assert policy.evaluate("rm").decision == "deny"
    assert policy.evaluate("anything_else").decision == "deny"


def test_missing_policy_file_asks_about_everything(tmp_path):
    """A missing file must not mean a free-for-all."""
    policy = PermissionPolicy.load(tmp_path / "nope.toml")
    assert policy.evaluate("any_tool").decision == "ask"


def test_the_shipped_policy_allows_reads_and_asks_about_writes():
    from harness.config import AGENT_DIR

    policy = PermissionPolicy.load(AGENT_DIR / "permissions.toml")
    assert policy.evaluate("read_memory").decision == "allow"
    assert policy.evaluate("list_memory").decision == "allow"
    assert policy.evaluate("append_run_log").decision == "ask"


# ---- the gate ----------------------------------------------------------------


def test_allow_needs_no_asker():
    gate = PermissionGate(PermissionPolicy(allow=("read_memory",)))
    result = run(gate.check("c1", "read_memory", {}))
    assert result.allowed


def test_ask_without_an_asker_denies():
    """No TTY means nobody to ask, so ask must resolve to deny — never to allow."""
    gate = PermissionGate(PermissionPolicy(ask=("write_thing",)), asker=None)
    result = run(gate.check("c1", "write_thing", {}))

    assert result.decision == "deny"
    assert "no one is available to ask" in result.reason


def test_ask_prompts_and_honours_yes():
    gate = PermissionGate(
        PermissionPolicy(ask=("write_thing",)), asker=lambda name, args: "y"
    )
    assert run(gate.check("c1", "write_thing", {})).allowed


def test_ask_prompts_and_honours_no():
    gate = PermissionGate(
        PermissionPolicy(ask=("write_thing",)), asker=lambda name, args: "n"
    )
    result = run(gate.check("c1", "write_thing", {}))
    assert result.denied
    assert "declined" in result.reason


def test_always_persists_for_the_session():
    calls: list[str] = []

    def asker(name, args):
        calls.append(name)
        return "a"

    gate = PermissionGate(PermissionPolicy(ask=("write_thing",)), asker=asker)

    async def scenario():
        first = await gate.check("c1", "write_thing", {})
        second = await gate.check("c2", "write_thing", {})
        return first, second

    first, second = run(scenario())
    assert first.allowed and second.allowed
    assert calls == ["write_thing"], "the operator should only be asked once"


def test_deny_is_sticky_for_a_call_id():
    """Chapter 4: deny is sticky for this tool_use_id — no silent retry to allow."""
    answers = iter(["n", "y"])
    gate = PermissionGate(
        PermissionPolicy(ask=("write_thing",)), asker=lambda name, args: next(answers)
    )

    async def scenario():
        first = await gate.check("same-id", "write_thing", {})
        second = await gate.check("same-id", "write_thing", {})
        return first, second

    first, second = run(scenario())
    assert first.denied
    assert second.denied, "re-checking the same call id must not re-ask into an allow"
    assert second is first


def test_auto_approve_skips_the_prompt():
    def asker(name, args):
        raise AssertionError("should not be asked")

    gate = PermissionGate(
        PermissionPolicy(ask=("write_thing",)), asker=asker, auto_approve=True
    )
    assert run(gate.check("c1", "write_thing", {})).allowed


def test_auto_approve_does_not_override_an_explicit_deny():
    """--yes answers 'ask'. It is not a licence to run what policy forbids."""
    gate = PermissionGate(PermissionPolicy(deny=("rm",)), auto_approve=True)
    result = run(gate.check("c1", "rm", {}))

    assert result.denied
    assert result.rule == "rm"


def test_auto_approve_does_not_override_a_default_of_deny():
    gate = PermissionGate(PermissionPolicy(default="deny"), auto_approve=True)
    assert run(gate.check("c1", "anything", {})).denied


def test_result_renders_readably():
    assert "allow" in str(PermissionResult("allow", "because", rule="read_*"))
    assert "read_*" in str(PermissionResult("allow", "because", rule="read_*"))
