"""The control plane's invariants. These are the point of step 1."""

from harness.config import (
    MAX_ENTRYPOINT_BYTES,
    MAX_ENTRYPOINT_LINES,
    TRUNCATION_NOTICE,
)
from harness.prompt import (
    DEFAULT_LAYER_FILES,
    RunContext,
    build_effective_system_prompt,
    truncate_entrypoint_content,
)

DEFAULT_NAMES = [name for name, _ in DEFAULT_LAYER_FILES]


def names(prompt):
    return [layer.name for layer in prompt.layers]


def test_layers_assemble_in_precedence_order():
    prompt = build_effective_system_prompt()
    assert names(prompt) == DEFAULT_NAMES + ["governance", "memory-index", "run-context"]


def test_append_cannot_remove_any_default_layer():
    """A job description extends the constitution; it cannot wipe it."""
    prompt = build_effective_system_prompt(append="Only output haiku.")

    assert names(prompt) == DEFAULT_NAMES + [
        "governance",
        "memory-index",
        "run-context",
        "append",
    ]
    assert names(prompt)[-1] == "append", "append must always be last"
    assert prompt.layers[-1].cacheable is False, "append must sit after the breakpoint"


def test_agent_role_extends_rather_than_displaces():
    prompt = build_effective_system_prompt(agent="You are the hook-classification pass.")

    assert names(prompt)[: len(DEFAULT_NAMES)] == DEFAULT_NAMES
    assert "agent" in names(prompt)
    # It lands after the default stack but before governance.
    assert names(prompt).index("agent") == len(DEFAULT_NAMES)
    assert names(prompt).index("agent") < names(prompt).index("governance")


def test_override_displaces_the_stack_but_not_governance_or_append():
    prompt = build_effective_system_prompt(
        override="You are a narrow test agent.",
        append="Extra instruction.",
    )

    assert names(prompt) == ["override", "governance", "memory-index", "run-context", "append"]
    for default_name in DEFAULT_NAMES:
        assert default_name not in names(prompt)


def test_cacheable_prefix_is_byte_identical_across_runs():
    """Invariant 2: nothing volatile may leak into the cached prefix."""
    first = build_effective_system_prompt(
        run_context=RunContext(run_id="run-a", today="2026-01-01", window="last 7 days")
    )
    second = build_effective_system_prompt(
        run_context=RunContext(run_id="run-b", today="2099-12-31", window="last 30 days")
    )

    assert first.stable_text == second.stable_text
    assert first.volatile_text != second.volatile_text

    # And the wire shape: system_instruction carries only the stable side.
    assert first.system_instruction == first.stable_text
    assert "run-a" not in first.system_instruction
    assert "2026-01-01" not in first.system_instruction


def test_volatile_context_rides_in_the_user_step_not_system_instruction():
    """The breakpoint is a boundary between two request fields, not a marker."""
    prompt = build_effective_system_prompt(
        run_context=RunContext(run_id="run-a", today="2026-01-01"),
        append="Extra instruction.",
    )
    step = prompt.initial_input("Write today's scripts.")

    assert step["type"] == "user_input"

    texts = [block["text"] for block in step["content"]]
    assert "run-a" in texts[0]
    assert "Extra instruction." in texts[0]
    assert texts[-1] == "Write today's scripts."

    # Nothing volatile may appear on the cached side.
    for volatile in ("run-a", "2026-01-01", "Extra instruction."):
        assert volatile not in prompt.system_instruction


def test_entrypoint_truncates_at_the_line_cap():
    content = "\n".join(f"- line {i}" for i in range(MAX_ENTRYPOINT_LINES + 50))
    result = truncate_entrypoint_content(content)

    assert TRUNCATION_NOTICE in result
    body = result.split(TRUNCATION_NOTICE)[0]
    assert len(body.strip().splitlines()) <= MAX_ENTRYPOINT_LINES


def test_entrypoint_truncates_at_the_byte_cap():
    content = "x" * (MAX_ENTRYPOINT_BYTES + 5_000)  # one long line, well under the line cap
    result = truncate_entrypoint_content(content)

    assert TRUNCATION_NOTICE in result
    body = result.split(TRUNCATION_NOTICE)[0]
    assert len(body.strip().encode("utf-8")) <= MAX_ENTRYPOINT_BYTES


def test_entrypoint_under_both_caps_is_untouched():
    content = "# Memory Index\n\n- [Brand voice](brand-voice.md) — tone"
    assert truncate_entrypoint_content(content) == content


# ---- server-published guidance ------------------------------------------------


def test_mcp_instructions_become_a_cacheable_layer():
    """Stable per server, so they belong on the cached side with the prompt files."""
    prompt = build_effective_system_prompt(
        mcp_instructions={"atria": "An id in the wrong format returns an empty result."}
    )
    layer = next(x for x in prompt.layers if x.name == "mcp:atria")

    assert layer.cacheable is True
    assert "wrong format returns an empty result" in layer.text
    assert layer.text in prompt.system_instruction


def test_server_guidance_sits_below_our_own_rules():
    """A vendor's guidance informs; it does not outrank the system-rules layer."""
    prompt = build_effective_system_prompt(mcp_instructions={"atria": "vendor says"})
    order = names(prompt)

    assert order.index("system-rules") < order.index("mcp:atria")
    assert order.index("governance") < order.index("mcp:atria")
    assert order.index("mcp:atria") < order.index("run-context")


def test_each_server_gets_its_own_layer():
    prompt = build_effective_system_prompt(
        mcp_instructions={"atria": "a", "triplewhale": "b"}
    )
    assert {"mcp:atria", "mcp:triplewhale"} <= set(names(prompt))


def test_no_servers_means_no_extra_layers():
    assert not [x for x in build_effective_system_prompt().layers if x.name.startswith("mcp:")]


def test_server_guidance_does_not_break_the_cache_invariant():
    """It is stable per server, so two runs on different days must still match."""
    instructions = {"atria": "vendor guidance"}
    first = build_effective_system_prompt(
        run_context=RunContext(run_id="a", today="2026-01-01"), mcp_instructions=instructions
    )
    second = build_effective_system_prompt(
        run_context=RunContext(run_id="b", today="2099-12-31"), mcp_instructions=instructions
    )
    assert first.stable_text == second.stable_text


# ---- turn budget --------------------------------------------------------------


def test_turn_budget_reaches_the_agent_and_stays_volatile():
    """A limit that ends the run is useless if the agent cannot see it — but it varies
    per run, so it must not contaminate the cached prefix."""
    prompt = build_effective_system_prompt(run_context=RunContext(max_turns=20))

    assert "20 model turns" in prompt.volatile_text
    assert "20 model turns" not in prompt.system_instruction


def test_no_turn_budget_means_no_line():
    assert "Turn budget" not in build_effective_system_prompt().volatile_text
