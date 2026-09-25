"""The query loop, driven by a scripted event stream. No network, no tokens."""

import asyncio

from harness.loop import LoopState, build_runtime, query_loop
from harness.prompt import RunContext, build_effective_system_prompt
from harness.tools import ToolContext, ToolRegistry, default_registry, tool
from harness.transcript import Transcript


# ---- scripted client ---------------------------------------------------------


class FakeStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        async def gen():
            for event in self._events:
                if isinstance(event, BaseException):
                    raise event
                yield event

        return gen()


class FakeInteractions:
    def __init__(self, turns, requests):
        self._turns = list(turns)
        self.requests = requests

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        assert self._turns, "loop asked for more turns than were scripted"
        return FakeStream(self._turns.pop(0))


class FakeClient:
    def __init__(self, turns):
        self.requests: list[dict] = []
        self.aio = type("Aio", (), {})()
        self.aio.interactions = FakeInteractions(turns, self.requests)


# ---- event helpers -----------------------------------------------------------


def text(chunk):
    return {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": chunk}}


def tool_call(index, call_id, name, args_json):
    return [
        {
            "event_type": "step.start",
            "index": index,
            "step": {"type": "function_call", "id": call_id, "name": name},
        },
        {
            "event_type": "step.delta",
            "index": index,
            "delta": {"type": "arguments_delta", "arguments": args_json},
        },
        {"event_type": "step.stop", "index": index},
    ]


def done(tokens=10):
    return {"event_type": "interaction.completed", "interaction": {"usage": {"total": tokens}}}


# ---- fixtures ----------------------------------------------------------------


@tool(concurrency_safe=True, read_only=True)
def ping(tag: str) -> str:
    """A trivial tool."""
    return f"pong:{tag}"


def registry():
    return ToolRegistry([ping])


def prompt():
    return build_effective_system_prompt(
        run_context=RunContext(run_id="test", today="2026-01-01")
    )


def fresh_state():
    return LoopState(transcript=Transcript("s-loop"))


def run(client, state=None, **kwargs):
    state = state or fresh_state()
    result = asyncio.run(
        query_loop(state, client=client, prompt=prompt(), registry=registry(), **kwargs)
    )
    return result, state


# ---- tests -------------------------------------------------------------------


def test_plain_answer_ends_the_turn():
    client = FakeClient([[text("hello "), text("world"), done()]])
    result, state = run(client)

    assert result.stop_reason == "end_turn"
    assert result.turns == 1
    assert result.text == "hello world"
    assert len(client.requests) == 1

    kinds = [n.kind for n in state.transcript.path_to_root()]
    assert kinds == ["model_output"]


def test_tool_call_runs_and_the_loop_continues():
    client = FakeClient(
        [
            [*tool_call(0, "c1", "ping", '{"tag": "a"}'), done()],
            [text("done"), done()],
        ]
    )
    result, state = run(client)

    assert result.stop_reason == "end_turn"
    assert result.turns == 2
    assert len(client.requests) == 2

    kinds = [n.kind for n in state.transcript.path_to_root()]
    assert kinds == ["function_call", "function_result", "model_output"]

    result_node = state.transcript.path_to_root()[1]
    assert result_node.step["result"][0]["text"] == "pong:a"
    assert "is_error" not in result_node.step


def test_runtime_context_is_rebuilt_from_the_transcript_each_turn():
    client = FakeClient(
        [
            [*tool_call(0, "c1", "ping", '{"tag": "a"}'), done()],
            [text("ok"), done()],
        ]
    )
    _result, _state = run(client)

    first, second = client.requests
    assert len(second["input"]) > len(first["input"])
    assert second["input"][:len(first["input"])] == first["input"]
    assert second["store"] is False
    assert second["system_instruction"] == first["system_instruction"]


def test_parallel_calls_both_close_before_the_next_turn():
    client = FakeClient(
        [
            [
                *tool_call(0, "c1", "ping", '{"tag": "a"}'),
                *tool_call(1, "c2", "ping", '{"tag": "b"}'),
                done(),
            ],
            [text("ok"), done()],
        ]
    )
    _result, state = run(client)

    kinds = [n.kind for n in state.transcript.path_to_root()]
    assert kinds.count("function_call") == 2
    assert kinds.count("function_result") == 2


def test_max_turns_terminates():
    turns = [[*tool_call(0, f"c{i}", "ping", '{"tag": "x"}'), done()] for i in range(5)]
    client = FakeClient(turns)
    result, _state = run(client, max_turns=3)

    assert result.stop_reason == "max_turns"
    assert result.turns == 3
    assert len(client.requests) == 3


def test_api_error_returns_without_retry():
    client = FakeClient([[text("partial"), {"event_type": "error", "error": {"message": "quota"}}]])
    result, _state = run(client)

    assert result.stop_reason == "api_error"
    assert "quota" in result.error
    assert len(client.requests) == 1, "API errors return directly, no retry"


def test_transport_exception_is_reported_not_raised():
    client = FakeClient([[RuntimeError("connection reset")]])
    result, _state = run(client)

    assert result.stop_reason == "api_error"
    assert "connection reset" in result.error


def test_interrupt_still_closes_the_ledger():
    """A call was issued; the stream then dies. The result must still exist."""
    client = FakeClient(
        [[*tool_call(0, "c1", "ping", '{"tag": "a"}'), KeyboardInterrupt()]]
    )
    result, state = run(client)

    assert result.stop_reason == "interrupted"

    kinds = [n.kind for n in state.transcript.path_to_root()]
    assert kinds.count("function_call") == kinds.count("function_result") == 1


def test_unknown_tool_closes_the_ledger_and_continues():
    client = FakeClient(
        [
            [*tool_call(0, "c1", "nope", "{}"), done()],
            [text("recovered"), done()],
        ]
    )
    result, state = run(client)

    assert result.stop_reason == "end_turn"
    error_node = state.transcript.path_to_root()[1]
    assert error_node.step["is_error"] is True
    assert "no such tool" in error_node.step["result"][0]["text"]


def test_build_runtime_carries_tools_and_system_instruction():
    state = fresh_state()
    runtime = build_runtime(state, prompt(), registry())
    request = runtime.request()

    assert request["store"] is False
    assert request["stream"] is True
    assert [t["name"] for t in request["tools"]] == ["ping"]
    assert request["system_instruction"].startswith("# Identity")


def test_thought_steps_are_replayed_in_the_next_request():
    """Dropping the signature made the live API reject turn 2 as invalid."""
    thought = [
        {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "thought_signature", "signature": "SIG123"},
        },
        {"event_type": "step.stop", "index": 0},
    ]
    client = FakeClient(
        [
            [*thought, *tool_call(1, "c1", "ping", '{"tag": "a"}'), done()],
            [text("ok"), done()],
        ]
    )
    _result, state = run(client)

    kinds = [n.kind for n in state.transcript.path_to_root()]
    assert kinds == ["thought", "function_call", "function_result", "model_output"]

    replayed = client.requests[1]["input"]
    assert replayed[0]["type"] == "thought"
    assert replayed[0]["signature"] == "SIG123"


def test_tool_body_starts_before_the_stream_finishes():
    """Mid-stream dispatch must actually execute mid-stream.

    submit() only creates a task; without an explicit yield the event loop never gets
    control while buffered events are iterated, so every tool body would wait for the
    stream to end. A live timeline caught that; this pins it.

    Events are tagged per turn: turn 2's stream would otherwise always log after the
    tool body and make this assertion vacuous.
    """
    log: list[str] = []

    @tool(concurrency_safe=True, read_only=True)
    async def watched() -> str:
        """Records when its body actually begins."""
        log.append("TOOL_BODY_START")
        return "ok"

    class LoggingStream(FakeStream):
        def __init__(self, events, label):
            super().__init__(events)
            self.label = label

        def __aiter__(self):
            async def gen():
                for i, event in enumerate(self._events):
                    log.append(f"{self.label}_event_{i}")
                    yield event

            return gen()

    class LoggingInteractions(FakeInteractions):
        turn = 0

        async def create(self, **kwargs):
            self.requests.append(kwargs)
            LoggingInteractions.turn += 1
            return LoggingStream(self._turns.pop(0), f"t{LoggingInteractions.turn}")

    client = FakeClient([])
    client.aio.interactions = LoggingInteractions(
        [
            [*tool_call(0, "c1", "watched", "{}"), text("trailing"), done()],
            [text("finished"), done()],
        ],
        client.requests,
    )

    asyncio.run(
        query_loop(
            LoopState(transcript=Transcript("s-midstream")),
            client=client,
            prompt=prompt(),
            registry=ToolRegistry([watched]),
        )
    )

    assert "TOOL_BODY_START" in log, "the tool never ran"
    turn_one = [i for i, e in enumerate(log) if e.startswith("t1_event_")]
    assert log.index("TOOL_BODY_START") < max(turn_one), (
        f"tool body waited for turn 1's stream to finish — mid-stream dispatch is "
        f"not working. log={log}"
    )


# ---- the in-memory projection -------------------------------------------------


def test_record_keeps_the_projection_and_the_tree_in_step():
    """One write path. Appending to one and not the other means the model sees a
    different conversation than the transcript records."""
    state = LoopState(transcript=Transcript("s-proj"))
    for i in range(5):
        state.record({"type": "model_output", "content": [{"type": "text", "text": f"m{i}"}]},
                     turn=i)

    assert state.messages == state.transcript.steps()
    assert len(state.messages) == 5


def test_project_rebuilds_from_the_tree_after_branching():
    t = Transcript("s-branch")
    state = LoopState(transcript=t)
    state.record({"type": "user_input", "content": [{"type": "text", "text": "shared"}]})
    fork = t.head
    state.record({"type": "model_output", "content": [{"type": "text", "text": "path A"}]})

    t.branch_from(fork)
    state.project()  # the tree's shape changed under us

    assert [s["content"][0]["text"] for s in state.messages] == ["shared"]
    assert state.messages == t.steps()


def test_project_rebuilds_from_the_tree_after_compaction():
    t = Transcript("s-proj-compact")
    state = LoopState(transcript=t)
    for i in range(6):
        state.record({"type": "model_output", "content": [{"type": "text", "text": f"m{i}"}]},
                     turn=i)

    t.compact_boundary("SUMMARY", t.path_to_root()[-2:])
    state.project()

    assert [s["content"][0]["text"] for s in state.messages] == ["SUMMARY", "m4", "m5"]
    assert state.messages == t.steps()


def test_the_request_input_matches_the_tree_walk_after_a_real_run():
    """The projection is what gets sent, so it must equal what the tree says happened."""
    client = FakeClient(
        [
            [*tool_call(0, "c1", "ping", '{"tag": "a"}'), done()],
            [text("finished"), done()],
        ]
    )
    result, state = run(client)

    assert result.stop_reason == "end_turn"
    assert state.messages == state.transcript.steps()
    # And the second request carried everything the first did, plus the turn's steps.
    first, second = (r["input"] for r in client.requests)
    assert second[: len(first)] == first
    assert len(second) > len(first)


# ---- children outstanding ------------------------------------------------------


def test_a_turn_will_not_end_while_children_are_still_working(tmp_path):
    """Evicting them would leak the work instead. Waiting is the better trade: it is
    already paid for, and a coordinator that forgets to collect would otherwise throw
    away three researchers it just spent two minutes on."""
    from harness.agents import AgentPool, ChildOutcome

    class Finished(AgentPool):
        """A pool with one child that is about to report."""

        def __init__(self):
            super().__init__(parent_id="s-p", registry=default_registry(), runs_dir=tmp_path)
            self._issued.append("s-p.researcher-1")
            self._meta["s-p.researcher-1"] = ("researcher", "read the account")
            self._live = True

        def pending(self):
            return ["s-p.researcher-1"] if self._live else []

        async def drain(self):
            self._live = False
            self._close(
                ChildOutcome(
                    child_id="s-p.researcher-1",
                    role="researcher",
                    task="read the account",
                    text="ad 6897420294631 spent $29.02",
                )
            )
            return self.ledger()

    client = FakeClient([[*text("spawned, done for now")], [*text("here is the synthesis")]])
    state = LoopState(transcript=Transcript.create("children-outstanding", runs_dir=tmp_path))
    state.project()

    result = asyncio.run(
        query_loop(
            state,
            client=client,
            prompt=build_effective_system_prompt(),
            registry=default_registry(),
            ctx=ToolContext(session_id="s-p", pool=Finished()),
        )
    )

    assert result.stop_reason == "end_turn"
    assert result.turns == 2, "the first turn was not allowed to be the last"
    injected = [
        s for s in state.messages if s.get("type") == "user_input" and "have finished" in
        "".join(b.get("text", "") for b in s.get("content") or [])
    ]
    assert injected, "the children's findings were recorded as a turn"
    assert "6897420294631" in "".join(b["text"] for b in injected[0]["content"])


def test_an_ordinary_run_is_untouched_by_the_child_check(tmp_path):
    """No pool means one attribute lookup and nothing else."""
    client = FakeClient([[*text("done")]])
    state = LoopState(transcript=Transcript.create("no-children", runs_dir=tmp_path))
    state.project()

    result = asyncio.run(
        query_loop(
            state,
            client=client,
            prompt=build_effective_system_prompt(),
            registry=default_registry(),
        )
    )

    assert result.stop_reason == "end_turn" and result.turns == 1


def test_findings_are_delivered_even_when_the_children_finished_long_ago(tmp_path):
    """The flaw async spawning introduced. Asking "is anything still running?" was right
    while spawning blocked; once the parent outlives the spawn, children finish while it is
    busy elsewhere. A live run burned three turns announcing it was waiting for reports it
    already had, because nothing was running so nothing was delivered."""
    from harness.agents import AgentPool, ChildOutcome

    pool = AgentPool(parent_id="s-p", registry=default_registry(), runs_dir=tmp_path)
    pool._issued.append("s-p.researcher-1")
    pool._meta["s-p.researcher-1"] = ("researcher", "read the account")
    pool._close(
        ChildOutcome(
            child_id="s-p.researcher-1",
            role="researcher",
            task="read the account",
            text="ad 6897420294631 spent $29.02",
        )
    )
    assert not pool.pending(), "already finished, nothing in flight"

    client = FakeClient([[*text("nothing more to do")], [*text("now I can synthesize")]])
    state = LoopState(transcript=Transcript.create("late-delivery", runs_dir=tmp_path))
    state.project()

    result = asyncio.run(
        query_loop(
            state,
            client=client,
            prompt=build_effective_system_prompt(),
            registry=default_registry(),
            ctx=ToolContext(session_id="s-p", pool=pool),
        )
    )

    assert result.turns == 2, "the findings were delivered rather than skipped"
    assert any(
        "6897420294631" in "".join(b.get("text", "") for b in s.get("content") or [])
        for s in state.messages
        if s.get("type") == "user_input"
    )


def test_findings_are_delivered_once(tmp_path):
    """`reported` is what stops a second collection repeating them."""
    from harness.agents import AgentPool, ChildOutcome

    pool = AgentPool(parent_id="s-p", registry=default_registry(), runs_dir=tmp_path)
    pool._issued.append("s-p.researcher-1")
    pool._meta["s-p.researcher-1"] = ("researcher", "x")
    pool._close(ChildOutcome(child_id="s-p.researcher-1", role="researcher", task="x", text="y"))

    client = FakeClient([[*text("a")], [*text("b")]])
    state = LoopState(transcript=Transcript.create("delivered-once", runs_dir=tmp_path))
    state.project()
    asyncio.run(
        query_loop(
            state,
            client=client,
            prompt=build_effective_system_prompt(),
            registry=default_registry(),
            ctx=ToolContext(session_id="s-p", pool=pool),
        )
    )

    injected = [
        s for s in state.messages
        if s.get("type") == "user_input"
        and "have finished" in "".join(b.get("text", "") for b in s.get("content") or [])
    ]
    assert len(injected) == 1
