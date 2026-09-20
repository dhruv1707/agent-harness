"""Microcompaction: what it clears, what it must not touch, and what it costs.

The charter: every clearing leaves a conversation the model can still read — every call
still has a result, thought signatures still replay — and the transcript is never touched,
because `verify` grounds against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from harness.microcompact import (
    CLEARED_MESSAGE,
    MicrocompactState,
    apply,
    gap_minutes,
    maybe_microcompact,
    reapply,
    select,
)


@dataclass
class FakeNode:
    step: dict
    ts: str


def call(cid: str, name: str = "list_ads") -> dict:
    return {"type": "function_call", "id": cid, "name": name, "arguments": {}}


def result(cid: str, body: str, name: str = "list_ads", error: bool = False) -> dict:
    step = {
        "type": "function_result",
        "call_id": cid,
        "name": name,
        "result": [{"type": "text", "text": body}],
    }
    if error:
        step["is_error"] = True
    return step


def conversation(n: int, size: int = 5_000) -> list[dict]:
    """n call/result pairs, each result `size` bytes, interleaved with model output."""
    steps: list[dict] = [{"type": "user_input", "content": [{"type": "text", "text": "go"}]}]
    for i in range(n):
        steps.append({"type": "model_output", "content": [{"type": "text", "text": f"m{i}"}]})
        steps.append(call(f"c{i}"))
        steps.append(result(f"c{i}", "x" * size))
    return steps


def nodes_for(steps: list[dict], *, minutes_ago: float = 0.0) -> list[FakeNode]:
    ts = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    return [FakeNode(step=s, ts=ts) for s in steps]


# ---- selection ---------------------------------------------------------------


def test_select_keeps_the_most_recent_five():
    ids = select(conversation(8), keep_recent=5)
    assert ids == ["c0", "c1", "c2"]


def test_select_keeps_everything_when_at_or_under_the_keep_count():
    assert select(conversation(5), keep_recent=5) == []
    assert select(conversation(1), keep_recent=5) == []


def test_keep_recent_of_zero_does_not_silently_clear_everything():
    """`list[-0:]` is the whole list, so a naive slice would keep all of them."""
    ids = select(conversation(4), keep_recent=0)
    assert ids == ["c0", "c1", "c2"]  # floors at 1, not "keep none" and not "keep all"


# ---- applying ----------------------------------------------------------------


def test_clearing_replaces_the_body_and_nothing_else():
    steps = conversation(8)
    out, freed = apply(steps, {"c0"})

    cleared = next(s for s in out if s.get("call_id") == "c0")
    assert cleared["result"][0]["text"] == CLEARED_MESSAGE
    assert cleared["name"] == "list_ads"
    assert cleared["call_id"] == "c0"
    assert freed == 5_000


def test_the_error_flag_survives_clearing():
    out, _ = apply([result("c0", "boom", error=True)], {"c0"})
    assert out[0]["is_error"] is True


def test_clearing_does_not_mutate_the_input():
    """The step dicts are shared with the transcript's nodes; editing one rewrites history."""
    steps = conversation(8)
    before = [s["result"][0]["text"] for s in steps if s.get("type") == "function_result"]

    apply(steps, {"c0", "c1", "c2"})

    after = [s["result"][0]["text"] for s in steps if s.get("type") == "function_result"]
    assert after == before


def test_every_call_still_has_a_result_after_clearing():
    steps = conversation(8)
    out, _ = apply(steps, set(select(steps, keep_recent=5)))

    calls = [s["id"] for s in out if s.get("type") == "function_call"]
    results = [s["call_id"] for s in out if s.get("type") == "function_result"]
    assert calls == results
    assert len(out) == len(steps)  # replaced in place, never dropped


def test_clearing_twice_is_idempotent_and_does_not_double_count():
    steps = conversation(8)
    once, first = apply(steps, {"c0"})
    twice, second = apply(once, {"c0"})

    assert first == 5_000
    assert second == 0
    assert twice == once


def test_thought_steps_are_never_touched():
    """Gemini 3 rejects the next request if a signature is not replayed verbatim."""
    steps = [{"type": "thought", "signature": "SIG123"}, *conversation(8)]
    out, _ = apply(steps, set(select(steps, keep_recent=5)))
    assert out[0] == {"type": "thought", "signature": "SIG123"}


# ---- the clock ---------------------------------------------------------------


def test_gap_is_measured_from_the_last_model_output():
    steps = conversation(3)
    assert gap_minutes(nodes_for(steps, minutes_ago=90)) == pytest.approx(90, abs=1)


def test_gap_is_none_when_the_model_has_not_spoken_yet():
    steps = [{"type": "user_input", "content": [{"type": "text", "text": "go"}]}]
    assert gap_minutes(nodes_for(steps)) is None


# ---- triggers ----------------------------------------------------------------


def test_time_based_fires_past_the_gap():
    steps, state = conversation(8), MicrocompactState()
    out = maybe_microcompact(steps, nodes_for(steps, minutes_ago=61), state, threshold=99)

    assert out is not None
    assert state.cleared == {"c0", "c1", "c2"}
    assert state.reclaimed_bytes == 15_000


def test_time_based_does_not_fire_under_the_gap():
    steps, state = conversation(8), MicrocompactState()
    assert maybe_microcompact(steps, nodes_for(steps, minutes_ago=59), state, threshold=99) is None
    assert state.cleared == set()


def test_time_based_ignores_the_reclaim_floor():
    """The cache is already gone; there is no miss to justify."""
    steps, state = conversation(8, size=10), MicrocompactState()
    out = maybe_microcompact(
        steps, nodes_for(steps, minutes_ago=61), state, threshold=99, min_reclaim=1_000_000
    )
    assert out is not None


def test_count_fires_over_the_threshold_on_a_warm_cache():
    steps, state = conversation(13), MicrocompactState()
    out = maybe_microcompact(steps, nodes_for(steps), state, threshold=12)

    assert out is not None
    assert len(state.cleared) == 8  # 13 results, keep 5


def test_count_does_not_fire_under_the_threshold():
    steps, state = conversation(11), MicrocompactState()
    assert maybe_microcompact(steps, nodes_for(steps), state, threshold=12) is None


def test_the_reclaim_floor_stops_a_warm_clearing_that_is_not_worth_the_miss():
    """Removing less than you keep is worse than doing nothing."""
    steps, state = conversation(13, size=100), MicrocompactState()
    out = maybe_microcompact(steps, nodes_for(steps), state, threshold=12, min_reclaim=20_000)

    assert out is None
    assert state.cleared == set()


def test_time_based_short_circuits_the_count_path():
    """A large gap establishes the cache is cold, so the miss is not priced twice."""
    steps, state = conversation(13, size=100), MicrocompactState()
    # min_reclaim would refuse this on the warm path, but the cold path ignores it.
    out = maybe_microcompact(
        steps, nodes_for(steps, minutes_ago=61), state, threshold=12, min_reclaim=20_000
    )
    assert out is not None


def test_disabled_does_nothing():
    steps, state = conversation(13), MicrocompactState()
    assert maybe_microcompact(steps, nodes_for(steps, minutes_ago=99), state, enabled=False) is None


def test_nothing_to_clear_is_not_a_clearing():
    steps, state = conversation(3), MicrocompactState()
    assert maybe_microcompact(steps, nodes_for(steps, minutes_ago=999), state) is None
    assert state.clearings == 0


# ---- surviving a rebuild -----------------------------------------------------


def test_reapply_restores_the_clearing_after_a_rebuild_from_the_tree():
    """`project()` walks a tree that still holds every result in full."""
    steps, state = conversation(8), MicrocompactState()
    maybe_microcompact(steps, nodes_for(steps, minutes_ago=61), state, threshold=99)

    fresh = conversation(8)  # what the tree would hand back
    restored = reapply(fresh, state)

    cleared = [
        s["call_id"]
        for s in restored
        if s.get("type") == "function_result" and s["result"][0]["text"] == CLEARED_MESSAGE
    ]
    assert cleared == ["c0", "c1", "c2"]


def test_reapply_does_not_double_count_the_bytes_it_already_banked():
    steps, state = conversation(8), MicrocompactState()
    maybe_microcompact(steps, nodes_for(steps, minutes_ago=61), state, threshold=99)
    banked = state.reclaimed_bytes

    reapply(conversation(8), state)

    assert state.reclaimed_bytes == banked


def test_reapply_is_a_no_op_when_nothing_has_been_cleared():
    steps = conversation(8)
    assert reapply(steps, MicrocompactState()) is steps


# ---- wiring into the loop and the session ------------------------------------


def test_project_reapplies_the_clearing_instead_of_undoing_it():
    """The tree still holds every result in full, so a plain walk would restore them."""
    from harness.loop import LoopState
    from harness.transcript import Transcript

    state = LoopState(transcript=Transcript("s-micro"))
    for step in conversation(8):
        state.record(step, turn=1)

    state.messages = maybe_microcompact(
        state.messages, nodes_for(conversation(8), minutes_ago=61), state.micro, threshold=99
    )
    state.project()

    bodies = [s["result"][0]["text"] for s in state.messages if s.get("type") == "function_result"]
    assert bodies[:3] == [CLEARED_MESSAGE] * 3
    assert bodies[3:] == ["x" * 5_000] * 5


def test_clearing_never_reaches_the_transcript():
    """`verify` grounds against the tree, so the tree must keep every byte."""
    from harness.loop import LoopState
    from harness.transcript import Transcript

    state = LoopState(transcript=Transcript("s-micro-tree"))
    for step in conversation(8):
        state.record(step, turn=1)

    state.messages = maybe_microcompact(
        state.messages, nodes_for(conversation(8), minutes_ago=61), state.micro, threshold=99
    )

    from_tree = [
        n.step["result"][0]["text"]
        for n in state.transcript.path_to_root()
        if n.step.get("type") == "function_result"
    ]
    assert from_tree == ["x" * 5_000] * 8
    assert CLEARED_MESSAGE not in "".join(from_tree)


def test_the_session_shares_one_cleared_set_across_submissions():
    """A fresh LoopState is built per submit; a set held there would forget everything."""
    from harness.loop import LoopState
    from harness.session import AgentSession
    from harness.transcript import Transcript

    session = AgentSession(transcript=Transcript("s-micro-session"))
    session._micro.cleared.add("c0")

    first = LoopState(transcript=session.transcript, micro=session._micro)
    second = LoopState(transcript=session.transcript, micro=session._micro)

    assert first.micro is second.micro is session._micro
    assert second.micro.cleared == {"c0"}
