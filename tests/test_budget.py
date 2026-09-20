"""The size gate: what gets persisted, what may never change, and what it costs.

The charter: a result too big on arrival never reaches the request, a result the model has
already seen is never altered afterwards, and the transcript keeps every byte either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from harness.budget import (
    ContentReplacementState,
    apply,
    build_replacement,
    groups,
    persist,
    preview,
    reapply,
    results_dir,
    select_fresh,
)
from harness.config import approx_tokens, bytes_per_token_for
from harness.tools import ToolContext, ToolRegistry, tool


@dataclass
class FakeNode:
    step: dict
    turn: int


@tool(concurrency_safe=True, read_only=True)
def big(ctx: ToolContext) -> str:
    """A tool with the default ceiling."""
    return "x"


@tool(concurrency_safe=True, read_only=True, max_result_chars=None)
def never(ctx: ToolContext) -> str:
    """A tool that opts out of persistence."""
    return "x"


@tool(concurrency_safe=True, read_only=True, max_result_chars=1_000)
def tight(ctx: ToolContext) -> str:
    """A tool that declares a lower ceiling."""
    return "x"


REGISTRY = ToolRegistry([big, never, tight])


def result(cid: str, body: str, name: str = "big") -> dict:
    return {
        "type": "function_result",
        "call_id": cid,
        "name": name,
        "result": [{"type": "text", "text": body}],
    }


def call(cid: str, name: str = "big") -> dict:
    return {"type": "function_call", "id": cid, "name": name, "arguments": {}}


def run(messages, nodes, state, tmp_path, **kw):
    return apply(
        messages,
        nodes,
        state,
        registry=REGISTRY,
        runs_dir=tmp_path,
        session_id="s-test",
        **kw,
    )


def nodes_for(steps, turns=None):
    turns = turns or [1] * len(steps)
    return [FakeNode(step=s, turn=t) for s, t in zip(steps, turns)]


def body_of(step):
    return step["result"][0]["text"]


# ---- grouping ----------------------------------------------------------------


def test_a_turns_results_form_one_group():
    steps = [call("c0"), call("c1"), result("c0", "a"), result("c1", "b")]
    assert groups(steps, nodes_for(steps)) == [[2, 3]]


def test_a_model_output_between_calls_does_not_split_the_results():
    """Calls interleave with text; results do not. Only the results are grouped."""
    steps = [
        {"type": "model_output", "content": [{"type": "text", "text": "hm"}]},
        call("c0"),
        {"type": "model_output", "content": [{"type": "text", "text": "and"}]},
        call("c1"),
        result("c0", "a"),
        result("c1", "b"),
    ]
    assert groups(steps, nodes_for(steps)) == [[4, 5]]


def test_two_turns_are_two_groups():
    steps = [result("c0", "a"), result("c1", "b"), call("d0"), result("d0", "c")]
    nodes = nodes_for(steps, turns=[1, 1, 2, 2])
    assert groups(steps, nodes) == [[0, 1], [3]]


def test_adjacent_results_from_different_turns_split_on_the_turn():
    """Contiguity alone would merge these; the turn stamp is what separates them."""
    steps = [result("c0", "a"), result("d0", "b")]
    assert groups(steps, nodes_for(steps, turns=[1, 2])) == [[0], [1]]


# ---- per-result gate ---------------------------------------------------------


def test_a_result_over_the_ceiling_is_persisted(tmp_path):
    steps = [result("c0", "J" * 60_000)]
    state = ContentReplacementState()
    out = run(steps, nodes_for(steps), state, tmp_path)

    assert body_of(out[0]).startswith("<persisted-output>")
    assert "60,000" not in body_of(out[0])  # the payload itself is gone
    assert (results_dir(tmp_path, "s-test") / "c0.txt").read_text() == "J" * 60_000
    assert state.persisted == 1


def test_a_result_under_the_ceiling_is_untouched(tmp_path):
    steps = [result("c0", "J" * 40_000)]
    out = run(steps, nodes_for(steps), ContentReplacementState(), tmp_path)
    assert body_of(out[0]) == "J" * 40_000


def test_a_tool_can_opt_out_of_persistence_at_any_size(tmp_path):
    steps = [result("c0", "J" * 500_000, name="never")]
    state = ContentReplacementState()
    out = run(steps, nodes_for(steps), state, tmp_path)

    assert body_of(out[0]) == "J" * 500_000
    assert state.persisted == 0


def test_a_tool_can_declare_a_lower_ceiling(tmp_path):
    steps = [result("c0", "J" * 5_000, name="tight")]
    out = run(steps, nodes_for(steps), ContentReplacementState(), tmp_path)
    assert body_of(out[0]).startswith("<persisted-output>")


def test_the_replacement_names_the_call_id_so_it_can_be_read_back(tmp_path):
    steps = [result("c0", "J" * 60_000)]
    out = run(steps, nodes_for(steps), ContentReplacementState(), tmp_path)
    assert 'read_tool_result(call_id="c0")' in body_of(out[0])


def test_the_transcript_input_is_never_mutated(tmp_path):
    """Step dicts are shared with the tree; editing one would rewrite history."""
    steps = [result("c0", "J" * 60_000)]
    run(steps, nodes_for(steps), ContentReplacementState(), tmp_path)
    assert body_of(steps[0]) == "J" * 60_000


# ---- per-group gate ----------------------------------------------------------


def test_a_group_over_budget_persists_the_largest_until_it_fits(tmp_path):
    """Five 45k results: each under the per-result ceiling, 225k together."""
    steps = [result(f"c{i}", "J" * 45_000) for i in range(5)]
    state = ContentReplacementState()
    out = run(steps, nodes_for(steps), state, tmp_path, group_limit=200_000)

    persisted = [s for s in out if body_of(s).startswith("<persisted-output>")]
    assert len(persisted) == 1  # one is enough to get under; no more than needed
    assert state.persisted == 1


def test_a_group_under_budget_is_left_alone(tmp_path):
    steps = [result(f"c{i}", "J" * 40_000) for i in range(4)]
    state = ContentReplacementState()
    run(steps, nodes_for(steps), state, tmp_path, group_limit=200_000)
    assert state.persisted == 0


def test_groups_are_judged_independently(tmp_path):
    """150k in one round and 150k in the next are each within budget."""
    first = [result(f"a{i}", "J" * 30_000) for i in range(5)]
    second = [result(f"b{i}", "J" * 30_000) for i in range(5)]
    steps = first + second
    state = ContentReplacementState()
    run(steps, nodes_for(steps, turns=[1] * 5 + [2] * 5), state, tmp_path, group_limit=200_000)
    assert state.persisted == 0


def test_select_fresh_takes_the_biggest_first_and_stops():
    chosen = select_fresh({0: 10, 1: 90, 2: 50}, frozen_total=0, limit=100)
    assert chosen == [1]  # 150 -> 60, already under


def test_select_fresh_gives_up_gracefully_when_frozen_alone_busts_the_budget():
    """Nothing can be reclaimed; the overage is accepted rather than looped over."""
    chosen = select_fresh({0: 10}, frozen_total=500, limit=100)
    assert chosen == [0]


# ---- the tri-state -----------------------------------------------------------


def test_a_result_sent_in_full_is_never_replaced_afterwards(tmp_path):
    """frozen: altering it now would move the prefix and cost the cache."""
    state = ContentReplacementState()
    first = [result("c0", "J" * 40_000)]
    run(first, nodes_for(first), state, tmp_path, group_limit=200_000)
    assert state.seen == {"c0"}

    # The same result, now in a group that busts the budget.
    steps = first + [result("c1", "J" * 190_000)]
    out = run(steps, nodes_for(steps), state, tmp_path, group_limit=200_000)

    assert body_of(out[0]) == "J" * 40_000  # frozen, untouched


def test_a_replacement_comes_back_verbatim_without_touching_the_disk(tmp_path, monkeypatch):
    """mustReapply: byte-identical, zero I/O. Regenerating risks a byte differing."""
    steps = [result("c0", "J" * 60_000)]
    state = ContentReplacementState()
    first = run(steps, nodes_for(steps), state, tmp_path)

    import harness.budget as budget

    monkeypatch.setattr(
        budget, "persist", lambda *a, **k: pytest.fail("re-persisted a known replacement")
    )
    second = run(steps, nodes_for(steps), state, tmp_path)

    assert body_of(second[0]) == body_of(first[0])


def test_persisting_the_same_call_twice_is_a_no_op(tmp_path):
    """History is replayed on every request build; a call id's content never changes."""
    directory = tmp_path / "out"
    persist("original", "c0", directory)
    persist("different", "c0", directory)
    assert (directory / "c0.txt").read_text() == "original"


# ---- rebuilds ----------------------------------------------------------------


def test_reapply_restores_replacements_after_a_rebuild_from_the_tree(tmp_path):
    steps = [result("c0", "J" * 60_000)]
    state = ContentReplacementState()
    run(steps, nodes_for(steps), state, tmp_path)

    restored = reapply([result("c0", "J" * 60_000)], state)
    assert body_of(restored[0]).startswith("<persisted-output>")


def test_reapply_is_a_no_op_when_nothing_was_persisted():
    steps = [result("c0", "small")]
    assert reapply(steps, ContentReplacementState()) is steps


# ---- previews and token estimation -------------------------------------------


def test_preview_cuts_at_a_newline_when_that_keeps_most_of_it():
    text = ("line\n" * 100) + "x" * 50
    assert preview(text, limit=200).endswith("line")


def test_preview_falls_back_to_the_byte_limit_for_one_long_line():
    """Single-line JSON is the common case, so the fallback is the normal path."""
    assert len(preview("x" * 5_000, limit=200)) == 200


def test_json_is_estimated_at_two_chars_per_token():
    assert bytes_per_token_for('{"a": 1, "b": [2, 3]}') == 2
    assert bytes_per_token_for("just some ordinary prose here") == 4


def test_json_with_a_derived_totals_block_appended_is_still_json():
    """The MCP layer appends to results, so a strict parse would miss the big ones."""
    payload = '{"ads": [1, 2, 3]}\n\n[harness-derived] total spend: 100'
    assert bytes_per_token_for(payload) == 2


def test_approx_tokens_is_unchanged_for_existing_callers():
    assert approx_tokens("a" * 400) == 100


def test_approx_tokens_doubles_the_estimate_for_json():
    assert approx_tokens("a" * 400, 2) == 200


def test_the_replacement_is_far_cheaper_than_what_it_replaces():
    body = '{"k": "' + "v" * 200_000 + '"}'
    replacement = build_replacement(body, "big", Path("/tmp/s-tool-results/c0.txt"))
    assert approx_tokens(replacement, 2) < approx_tokens(body, 2) / 20


# ---- wiring into the loop ----------------------------------------------------


def _loop_bits():
    import sys

    sys.path.insert(0, "tests")
    from test_loop import FakeClient, done, text, tool_call

    return FakeClient, done, text, tool_call


def test_a_turns_results_land_as_one_unbroken_run():
    """The group definition rests on this, and nothing else asserts it.

    Calls stream into the projection one at a time and interleave with text; results are
    buffered in the ledger and flushed together, so they never do.
    """
    import asyncio

    from harness.loop import LoopState, query_loop
    from harness.prompt import RunContext, build_effective_system_prompt
    from harness.transcript import Transcript

    FakeClient, done, text, tool_call = _loop_bits()

    @tool(concurrency_safe=True, read_only=True)
    def ping(ctx: ToolContext, tag: str) -> str:
        """ping"""
        return "pong:" + tag

    registry = ToolRegistry([ping])
    client = FakeClient(
        [
            [
                text("first "),
                *tool_call(0, "c0", "ping", '{"tag":"a"}'),
                text("and also "),
                *tool_call(1, "c1", "ping", '{"tag":"b"}'),
                *tool_call(2, "c2", "ping", '{"tag":"c"}'),
                done(),
            ],
            [text("done"), done()],
        ]
    )

    state = LoopState(transcript=Transcript("s-run"))
    prompt = build_effective_system_prompt(run_context=RunContext(run_id="s-run"))
    asyncio.run(query_loop(state, client=client, prompt=prompt, registry=registry, max_turns=5))

    kinds = [s.get("type") for s in state.messages]
    start = kinds.index("function_result")
    assert kinds[start : start + 3] == ["function_result"] * 3
    assert "function_result" not in kinds[start + 3 :]

    nodes = state.transcript.path_to_root()
    assert groups(state.messages, nodes) == [[start, start + 1, start + 2]]


def test_context_tokens_counts_results_recorded_after_the_response():
    """The lagging count was the bug: usage describes the request we sent, and this
    turn's tool results were recorded after it went out."""
    from harness.loop import LoopState, estimate_step
    from harness.transcript import Transcript

    state = LoopState(transcript=Transcript("s-count"))
    state.record({"type": "user_input", "content": [{"type": "text", "text": "go"}]}, turn=0)
    state.sent_through = len(state.messages)

    payload = '{"rows": [' + ",".join(str(i) for i in range(8_000)) + "]}"
    state.record(result("c0", payload), turn=1)
    state.usage = {"total_input_tokens": 10_000}
    state.note_usage()

    assert state.context_tokens > 10_000
    assert state.context_tokens == 10_000 + estimate_step(result("c0", payload))


def test_a_json_result_is_not_estimated_at_half_its_weight():
    from harness.loop import estimate_step

    payload = '{"k": "' + "v" * 4_000 + '"}'
    assert estimate_step(result("c0", payload)) == pytest.approx(len(payload) / 2, rel=0.01)


def test_microcompaction_beats_a_persisted_preview(tmp_path):
    """The size gate re-applies its replacement every turn. Without care that would put a
    preview back over a result cleared for age, undoing the clearing and re-counting the
    preview's bytes as reclaimed on every turn after."""
    from harness.microcompact import CLEARED_MESSAGE, MicrocompactState
    from harness.microcompact import apply as micro_apply
    from harness.microcompact import reapply as micro_reapply

    msgs = [result("c0", "J" * 60_000)]
    nodes = nodes_for(msgs)
    bstate, mstate = ContentReplacementState(), MicrocompactState()

    msgs = run(msgs, nodes, bstate, tmp_path)
    msgs, _ = micro_apply(msgs, {"c0"})
    mstate.cleared.add("c0")

    # A later turn: the gate restores its preview, then the clearing is re-asserted.
    msgs = micro_reapply(run(msgs, nodes, bstate, tmp_path), mstate)
    assert body_of(msgs[0]) == CLEARED_MESSAGE
