"""Normalizing the SDK's SSE into events the loop can consume.

The adapter owns argument accumulation: `arguments_delta` carries partial JSON, and a call
is only whole at its own `step.stop`.
"""

import asyncio

from harness.events import (
    StreamDone,
    StreamError,
    TextDelta,
    ToolCallReady,
    ToolCallStarted,
    normalize,
)


def run(raw_events):
    async def gen():
        for event in raw_events:
            yield event

    async def collect():
        return [event async for event in normalize(gen())]

    return asyncio.run(collect())


def call_start(index, call_id, name):
    return {
        "event_type": "step.start",
        "index": index,
        "step": {"type": "function_call", "id": call_id, "name": name},
    }


def args_delta(index, fragment):
    return {
        "event_type": "step.delta",
        "index": index,
        "delta": {"type": "arguments_delta", "arguments": fragment},
    }


def test_text_deltas_pass_through():
    events = run(
        [
            {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "he"}},
            {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "llo"}},
        ]
    )
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["he", "llo"]
    assert isinstance(events[-1], StreamDone)


def test_arguments_accumulate_across_split_deltas():
    """The JSON arrives in fragments that individually do not parse."""
    events = run(
        [
            call_start(0, "c1", "read_memory"),
            args_delta(0, '{"na'),
            args_delta(0, 'me": "brief-'),
            args_delta(0, 'samples.md"}'),
            {"event_type": "step.stop", "index": 0},
        ]
    )
    ready = [e for e in events if isinstance(e, ToolCallReady)]
    assert len(ready) == 1
    assert ready[0].arguments == {"name": "brief-samples.md"}
    assert ready[0].parse_error is None


def test_call_is_ready_at_its_own_step_stop_not_at_stream_end():
    """Call 0 must be dispatchable while call 1 is still streaming."""
    events = run(
        [
            call_start(0, "c1", "list_memory"),
            args_delta(0, "{}"),
            {"event_type": "step.stop", "index": 0},
            call_start(1, "c2", "read_memory"),
            args_delta(1, '{"name": "x"}'),
            {"event_type": "step.stop", "index": 1},
        ]
    )
    kinds = [type(e).__name__ for e in events]
    # ToolCallReady for c1 lands before c2 even starts.
    assert kinds.index("ToolCallReady") < kinds.index("ToolCallStarted", 1)


def test_parallel_calls_are_tracked_by_index():
    events = run(
        [
            call_start(0, "c1", "list_memory"),
            call_start(1, "c2", "read_memory"),
            args_delta(1, '{"name": "a"}'),
            args_delta(0, "{}"),
            {"event_type": "step.stop", "index": 1},
            {"event_type": "step.stop", "index": 0},
        ]
    )
    ready = [e for e in events if isinstance(e, ToolCallReady)]
    assert [r.call_id for r in ready] == ["c2", "c1"]
    assert ready[0].arguments == {"name": "a"}
    assert ready[1].arguments == {}


def test_malformed_arguments_still_yield_a_call():
    """Dropping it would leave a function_call with no possible result."""
    events = run(
        [
            call_start(0, "c1", "read_memory"),
            args_delta(0, '{"name": '),
            {"event_type": "step.stop", "index": 0},
        ]
    )
    ready = [e for e in events if isinstance(e, ToolCallReady)]
    assert len(ready) == 1
    assert ready[0].parse_error is not None


def test_stream_ending_mid_call_still_yields_the_call():
    events = run([call_start(0, "c1", "read_memory"), args_delta(0, '{"name"')])
    ready = [e for e in events if isinstance(e, ToolCallReady)]
    assert len(ready) == 1
    assert "stream ended" in ready[0].parse_error


def test_error_event_stops_the_stream():
    events = run(
        [
            {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "hi"}},
            {"event_type": "error", "error": {"message": "quota exceeded"}},
            {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "no"}},
        ]
    )
    assert isinstance(events[-1], StreamError)
    assert "quota exceeded" in events[-1].message
    assert not any(isinstance(e, StreamDone) for e in events)


def test_usage_is_carried_on_done():
    events = run(
        [{"event_type": "interaction.completed", "interaction": {"usage": {"total_tokens": 42}}}]
    )
    assert isinstance(events[-1], StreamDone)
    assert events[-1].usage == {"total_tokens": 42}


def test_started_is_emitted_before_ready():
    events = run(
        [call_start(0, "c1", "list_memory"), args_delta(0, "{}"), {"event_type": "step.stop", "index": 0}]
    )
    assert isinstance(events[0], ToolCallStarted)
    assert isinstance(events[1], ToolCallReady)


# ---- regressions found by a live run, not by the scripted tests ---------------


def test_thought_step_is_captured_with_its_signature():
    """Gemini 3 signs thought steps and rejects the next request if they are dropped."""
    from harness.events import ThoughtReady

    events = run(
        [
            {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "thought_signature", "signature": "SIG123"},
            },
            {"event_type": "step.stop", "index": 0},
        ]
    )
    ready = [e for e in events if isinstance(e, ThoughtReady)]
    assert len(ready) == 1
    assert ready[0].signature == "SIG123"


def test_arguments_present_on_step_start_survive_with_no_deltas():
    """A call can arrive complete on step.start with no arguments_delta at all."""
    events = run(
        [
            {
                "event_type": "step.start",
                "index": 0,
                "step": {
                    "type": "function_call",
                    "id": "c1",
                    "name": "ping",
                    "arguments": {"tag": "seeded"},
                },
            },
            {"event_type": "step.stop", "index": 0},
        ]
    )
    ready = [e for e in events if isinstance(e, ToolCallReady)]
    assert ready[0].arguments == {"tag": "seeded"}
    assert ready[0].parse_error is None


def test_argument_deltas_override_the_seed():
    events = run(
        [
            {
                "event_type": "step.start",
                "index": 0,
                "step": {"type": "function_call", "id": "c1", "name": "ping", "arguments": {}},
            },
            args_delta(0, '{"tag": "streamed"}'),
            {"event_type": "step.stop", "index": 0},
        ]
    )
    ready = [e for e in events if isinstance(e, ToolCallReady)]
    assert ready[0].arguments == {"tag": "streamed"}
