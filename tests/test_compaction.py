"""Compaction: a controlled reboot, not a chat recap."""

import asyncio

import pytest

from harness.compaction import (
    ELIDE_OVER_BYTES,
    is_prompt_too_long,
    plan_cut,
    render_for_summary,
    strip_for_summary,
    summarize,
)
from harness.config import MAX_CONSECUTIVE_COMPACT_FAILURES
from harness.loop import LoopState, maybe_compact
from harness.session_memory import SessionMemory
from harness.transcript import Transcript


def result(name, text, call_id="c1", is_error=False):
    step = {
        "type": "function_result",
        "call_id": call_id,
        "name": name,
        "result": [{"type": "text", "text": text}],
    }
    if is_error:
        step["is_error"] = True
    return step


def text_of(step):
    return step["result"][0]["text"]


# ---- pre-summary cleaning ----------------------------------------------------


def test_a_large_tool_result_becomes_a_label():
    cleaned = strip_for_summary([result("list_ad_account_metrics", "x" * 31_559)])[0]
    assert text_of(cleaned) == "[tool result: list_ad_account_metrics — 31,559 bytes elided]"


def test_a_transcript_gets_its_own_label():
    cleaned = strip_for_summary([result("get_ad_account_video_transcript", "y" * 3_385)])[0]
    assert text_of(cleaned).startswith("[transcript:")


def test_a_small_result_is_left_intact():
    """Small results are cheap and often carry the actual finding."""
    cleaned = strip_for_summary([result("ping", "pong")])[0]
    assert text_of(cleaned) == "pong"


def test_the_error_flag_survives_elision():
    cleaned = strip_for_summary([result("boom", "z" * 9_000, is_error=True)])[0]
    assert cleaned["is_error"] is True


def test_thought_signatures_are_dropped_entirely():
    """Base64 signatures are large and carry nothing a summary can use."""
    cleaned = strip_for_summary([{"type": "thought", "signature": "A" * 4_000}])[0]
    assert cleaned == {"type": "thought", "elided": True}


def test_long_model_output_is_truncated_not_dropped():
    cleaned = strip_for_summary(
        [{"type": "model_output", "content": [{"type": "text", "text": "w" * 9_000}]}]
    )[0]
    body = cleaned["content"][0]["text"]
    assert body.startswith("w" * 100)
    assert "elided" in body
    assert len(body) < 9_000


def test_stripping_does_not_mutate_the_input():
    original = result("big", "x" * 20_000)
    strip_for_summary([original])
    assert len(text_of(original)) == 20_000


def test_rendering_flattens_calls_and_results():
    rendered = render_for_summary(
        [
            {"type": "function_call", "name": "list_ad_account_ads", "arguments": {"limit": 10}},
            result("list_ad_account_ads", "ten ads"),
            {"type": "model_output", "content": [{"type": "text", "text": "done"}]},
        ]
    )
    assert "CALL list_ad_account_ads" in rendered
    assert "RESULT list_ad_account_ads: ten ads" in rendered
    assert "ASSISTANT: done" in rendered


# ---- where to cut ------------------------------------------------------------


class FakeNode:
    def __init__(self, turn):
        self.turn = turn


def test_the_cut_snaps_back_to_a_turn_boundary():
    """A function_call must never be separated from its function_result."""
    nodes = [FakeNode(1)] * 4 + [FakeNode(2)] * 4 + [FakeNode(3)] * 2
    cut = plan_cut(nodes, keep_share=0.20)
    assert cut == 8
    assert nodes[cut].turn != nodes[cut - 1].turn


def test_the_cut_never_splits_a_turn_even_when_the_share_lands_mid_turn():
    nodes = [FakeNode(1)] * 2 + [FakeNode(2)] * 8
    cut = plan_cut(nodes, keep_share=0.20)
    assert cut == 2, "keeping 2 of 10 lands inside turn 2, so it must move to its start"


def test_an_empty_history_has_nothing_to_cut():
    assert plan_cut([]) == 0


# ---- the boundary ------------------------------------------------------------


def u(text):
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def m(text):
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def build_transcript():
    """Five nodes — enough to exercise the boundary mechanics."""
    t = Transcript("s-compact")
    t.append(u("original question"), turn=0)
    for i in range(1, 5):
        t.append(m(f"work {i}"), turn=i)
    return t


def build_long_transcript(name="s-long", steps=16):
    """Past MIN_STEPS_TO_COMPACT, so the compaction path actually engages."""
    t = Transcript(name)
    t.append(u("original question"), turn=0)
    for i in range(1, steps):
        t.append(m(f"work {i}"), turn=i)
    return t


def test_the_boundary_is_a_new_root_and_shortens_the_walk():
    t = build_transcript()
    nodes = t.path_to_root()
    t.compact_boundary("SUMMARY", nodes[-2:], meta={"pre_compact_tokens": 82_000})

    walked = [s["content"][0]["text"] for s in t.steps()]
    assert walked == ["SUMMARY", "work 3", "work 4"]
    assert len(t.roots()) == 2


def test_the_pre_compaction_branch_survives():
    """Compaction is non-destructive by construction — the tree makes it free."""
    t = build_transcript()
    nodes = t.path_to_root()
    old_head = t.head
    t.compact_boundary("SUMMARY", nodes[-2:])

    assert len(t.steps(old_head)) == 5, "the original walk is still there"
    assert len(t) == 8, "nothing was deleted"


def test_the_boundary_records_where_it_came_from():
    t = build_transcript()
    old_head = t.head
    boundary = t.compact_boundary("SUMMARY", t.path_to_root()[-2:], meta={"steps_compacted": 3})

    assert boundary.meta["compact_boundary"] is True
    assert boundary.meta["compacted_from"] == old_head
    assert boundary.meta["steps_compacted"] == 3


def test_render_marks_the_boundary():
    t = build_transcript()
    t.compact_boundary("SUMMARY", t.path_to_root()[-2:])
    assert "*COMPACTED*" in t.render()


# ---- prompt-too-long ---------------------------------------------------------


def test_ptl_is_recognised_from_provider_wording():
    for message in (
        "Request exceeds the maximum token limit",
        "input is too long for this model",
        "context length exceeded",
    ):
        assert is_prompt_too_long(message)


def test_an_ordinary_error_is_not_mistaken_for_ptl():
    assert not is_prompt_too_long("connection reset by peer")


class FakeInteractions:
    def __init__(self, fail_first_with=None):
        self.calls = []
        self.fail_first_with = fail_first_with

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first_with and len(self.calls) == 1:
            raise RuntimeError(self.fail_first_with)
        return type("R", (), {"output_text": "## Current State\nsummarized\n"})()


class FakeClient:
    def __init__(self, **kw):
        self.aio = type("Aio", (), {})()
        self.aio.interactions = FakeInteractions(**kw)


def test_summarize_returns_a_parsed_brief():
    client = FakeClient()
    brief = asyncio.run(summarize(client, "m", [result("t", "x" * 5_000)]))
    assert brief.sections["Current State"] == "summarized"


def test_ptl_on_the_summary_call_retries_once_with_a_truncated_head():
    client = FakeClient(fail_first_with="prompt is too long")
    steps = [m(f"turn {i}") for i in range(10)]
    brief = asyncio.run(summarize(client, "m", steps))

    assert brief.sections["Current State"] == "summarized"
    assert len(client.aio.interactions.calls) == 2, "exactly one retry, never a loop"
    first, second = (c["input"][0]["content"][0]["text"] for c in client.aio.interactions.calls)
    assert len(second) < len(first), "the retry dropped the oldest half"


def test_a_non_ptl_failure_is_not_retried():
    client = FakeClient(fail_first_with="connection reset")
    with pytest.raises(RuntimeError):
        asyncio.run(summarize(client, "m", [m("x")]))
    assert len(client.aio.interactions.calls) == 1


# ---- the circuit breaker -----------------------------------------------------


class AlwaysFails:
    def __init__(self):
        self.calls = 0
        self.aio = type("Aio", (), {})()
        self.aio.interactions = self

    async def create(self, **kwargs):
        self.calls += 1
        raise RuntimeError("connection reset")


def test_three_consecutive_failures_stop_further_attempts():
    """"You may fail, but you may not fail infinitely without memory.\""""
    client = AlwaysFails()
    state = LoopState(transcript=build_long_transcript("s-breaker"), context_tokens=999_999)

    async def scenario():
        attempts = []
        for _ in range(6):
            attempts.append(await maybe_compact(state, client, "m"))
        return attempts

    attempts = asyncio.run(scenario())
    assert not any(attempts)
    assert state.compact_failures == MAX_CONSECUTIVE_COMPACT_FAILURES
    assert client.calls == MAX_CONSECUTIVE_COMPACT_FAILURES, "it stopped calling after three"


def test_compaction_does_not_run_below_the_threshold():
    client = AlwaysFails()
    state = LoopState(transcript=build_long_transcript("s-under"), context_tokens=1_000)
    assert asyncio.run(maybe_compact(state, client, "m")) is False
    assert client.calls == 0


def test_a_forced_compaction_ignores_the_threshold():
    """The PTL path compacts regardless of where the accounting thinks it is."""
    client = FakeClient()
    state = LoopState(transcript=build_long_transcript("s-forced"), context_tokens=10)

    assert asyncio.run(maybe_compact(state, client, "m", forced=True)) is True
    assert state.compactions == 1
    assert isinstance(state.session_memory, SessionMemory)
    assert len(state.transcript.roots()) == 2


# ---- the threshold -----------------------------------------------------------


def test_a_normal_budget_subtracts_both_reserves():
    from harness.config import (
        AUTOCOMPACT_BUFFER_TOKENS,
        MAX_OUTPUT_TOKENS_FOR_SUMMARY,
        compact_threshold,
    )

    assert compact_threshold(120_000) == (
        120_000 - MAX_OUTPUT_TOKENS_FOR_SUMMARY - AUTOCOMPACT_BUFFER_TOKENS
    )


def test_a_small_budget_never_produces_a_negative_threshold():
    """The reserves total 33,000. A test budget below that would fire on turn one."""
    from harness.config import compact_threshold

    for budget in (5_000, 30_000, 33_000):
        assert 0 < compact_threshold(budget) <= budget


def test_a_trivial_history_is_not_worth_a_summarization_call():
    from harness.config import MIN_STEPS_TO_COMPACT

    client = FakeClient()
    t = Transcript("s-tiny")
    for i in range(MIN_STEPS_TO_COMPACT - 1):
        t.append(m(f"step {i}"), turn=i)
    state = LoopState(transcript=t, context_tokens=999_999)

    assert asyncio.run(maybe_compact(state, client, "m")) is False
    assert client.aio.interactions.calls == []


def test_compaction_does_not_repeat_without_growth():
    """If the irreducible context exceeds the budget, retrying every turn is a runaway."""
    client = FakeClient()
    t = Transcript("s-runaway")
    for i in range(20):
        t.append(m(f"step {i}"), turn=i)
    state = LoopState(transcript=t, context_tokens=50_000)

    assert asyncio.run(maybe_compact(state, client, "m", budget=10_000)) is True
    assert state.last_compact_tokens == 50_000

    # Same size, or smaller: compacting again cannot help.
    assert asyncio.run(maybe_compact(state, client, "m", budget=10_000)) is False
    state.context_tokens = 40_000
    assert asyncio.run(maybe_compact(state, client, "m", budget=10_000)) is False

    # Grown past the last point: worth another pass.
    state.context_tokens = 60_000
    for i in range(20):
        state.transcript.append(m(f"more {i}"), turn=100 + i)
    assert asyncio.run(maybe_compact(state, client, "m", budget=10_000)) is True
