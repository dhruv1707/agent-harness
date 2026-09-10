"""The continuation brief and the gate that decides when to write it."""

from harness.config import (
    MAX_SECTION_TOKENS,
    MAX_SESSION_MEMORY_TOKENS,
    SESSION_MEMORY_FIRST_WRITE_TOKENS,
    approx_tokens,
)
from harness.session_memory import SECTIONS, TRIMMED, MemoryGate, SessionMemory


# ---- the brief ---------------------------------------------------------------


def test_render_always_carries_every_section():
    text = SessionMemory({"Current State": "ranked by roas"}).render()
    for name in SECTIONS:
        assert f"## {name}" in text


def test_parse_round_trips():
    original = SessionMemory(
        {"Current State": "read 8 hooks", "Worklog": "- ranked\n- transcribed"}
    )
    parsed = SessionMemory.parse(original.render())
    assert parsed.sections["Current State"] == "read 8 hooks"
    assert parsed.sections["Worklog"] == "- ranked\n- transcribed"


def test_parse_ignores_headings_outside_the_template():
    """The template may not drift, or the brief cannot be diffed or trusted on resume."""
    parsed = SessionMemory.parse("## Current State\nhere\n\n## Invented Section\nnope\n")
    assert parsed.sections["Current State"] == "here"
    assert "Invented Section" not in parsed.sections


def test_an_empty_section_round_trips_as_empty():
    assert SessionMemory.parse(SessionMemory().render()).sections["Worklog"] == ""


# ---- budgets -----------------------------------------------------------------


def test_an_oversized_section_loses_its_oldest_lines():
    lines = [f"entry {i} " + "x" * 200 for i in range(200)]
    brief = SessionMemory({"Worklog": "\n".join(lines)}).enforce_budgets()

    body = brief.sections["Worklog"]
    assert approx_tokens(body) <= MAX_SECTION_TOKENS + approx_tokens(TRIMMED) + 8
    assert TRIMMED in body
    assert "entry 199" in body, "the newest state is what continuation needs"
    assert "entry 0 " not in body


def test_the_section_caps_bound_the_whole_brief():
    """Why there is no whole-brief shedding step.

    Five sections capped at MAX_SECTION_TOKENS cannot exceed MAX_SESSION_MEMORY_TOKENS, so
    a shed path would be unreachable. Add a section and this fails — which is the signal to
    reinstate it, as the book's nine-section template requires.
    """
    assert len(SECTIONS) * MAX_SECTION_TOKENS <= MAX_SESSION_MEMORY_TOKENS


def test_every_section_over_budget_still_lands_within_the_whole_cap():
    filler = "y" * (MAX_SECTION_TOKENS * 8)
    brief = SessionMemory({name: filler for name in SECTIONS}).enforce_budgets()
    assert brief.tokens() <= MAX_SESSION_MEMORY_TOKENS


def test_a_small_brief_is_left_alone():
    brief = SessionMemory({"Current State": "all good"})
    assert brief.enforce_budgets().sections["Current State"] == "all good"


def test_save_and_load(tmp_path):
    SessionMemory({"Current State": "mid-run"}).save("s-1", runs_dir=tmp_path)
    assert SessionMemory.load("s-1", runs_dir=tmp_path).sections["Current State"] == "mid-run"
    assert SessionMemory.load("s-missing", runs_dir=tmp_path) is None


# ---- the gate ----------------------------------------------------------------


def test_nothing_is_written_below_the_first_threshold():
    """Below the threshold there is nothing worth compressing."""
    gate = MemoryGate()
    assert gate.decide(SESSION_MEMORY_FIRST_WRITE_TOKENS - 1, at_stopping_point=True) is None


def test_crossing_the_threshold_creates_the_brief():
    assert MemoryGate().decide(SESSION_MEMORY_FIRST_WRITE_TOKENS, False) == "create"


def test_growth_alone_does_not_trigger_an_update():
    gate = MemoryGate()
    gate.record_write(12_000)
    assert gate.decide(14_000, at_stopping_point=False) is None  # under the interval
    assert gate.decide(17_000, at_stopping_point=False) is None  # interval, but no work


def test_the_interval_plus_real_tool_activity_updates():
    gate = MemoryGate()
    gate.record_write(12_000)
    gate.observe_tool_calls(3)
    assert gate.decide(17_000, at_stopping_point=False) == "update"


def test_a_single_error_counts_as_activity_worth_recording():
    gate = MemoryGate()
    gate.record_write(12_000)
    gate.observe_tool_calls(1, errors=1)
    assert gate.decide(17_000, at_stopping_point=False) == "update"


def test_without_activity_it_waits_for_a_stopping_point():
    """Crossing the interval mid-tool-chain must not capture a half-formed picture."""
    gate = MemoryGate()
    gate.record_write(12_000)

    assert gate.decide(17_000, at_stopping_point=False) is None  # defer
    assert gate.decide(17_000, at_stopping_point=True) == "update"  # now it is coherent


def test_recording_a_write_resets_the_counters():
    gate = MemoryGate()
    gate.observe_tool_calls(5, errors=2)
    gate.record_write(20_000)

    assert gate.tool_calls_since == 0 and gate.errors_since == 0
    assert gate.decide(21_000, at_stopping_point=True) is None  # interval restarts
