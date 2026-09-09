"""The streaming tool executor: barrier scheduling and ledger closure."""

import asyncio
import time

import pytest

from harness.events import ToolCallReady
from harness.executor import StreamingToolExecutor
from harness.tools import ToolRegistry, tool

DELAY = 0.08
INTERVALS: list[tuple[str, float, float]] = []


@tool(concurrency_safe=True)
def safe_tool(tag: str) -> str:
    """A concurrency-safe tool."""
    start = time.monotonic()
    time.sleep(DELAY)
    INTERVALS.append((f"safe:{tag}", start, time.monotonic()))
    return f"safe:{tag}"


@tool(concurrency_safe=False)
def unsafe_tool(tag: str) -> str:
    """A tool that must run alone."""
    start = time.monotonic()
    time.sleep(DELAY)
    INTERVALS.append((f"unsafe:{tag}", start, time.monotonic()))
    return f"unsafe:{tag}"


@tool(concurrency_safe=True)
def boom() -> str:
    """Always raises."""
    raise ValueError("tool exploded")


@tool(concurrency_safe=True)
async def forever() -> str:
    """Never finishes in time.

    Async on purpose: an asyncio.sleep is genuinely cancellable, whereas a blocking
    time.sleep in a worker thread keeps running after wait_for gives up on it.
    """
    await asyncio.sleep(30)
    return "unreachable"


def registry() -> ToolRegistry:
    return ToolRegistry([safe_tool, unsafe_tool, boom, forever])


def call(call_id: str, name: str, **arguments) -> ToolCallReady:
    return ToolCallReady(index=0, call_id=call_id, name=name, arguments=arguments)


def overlaps(a: tuple[str, float, float], b: tuple[str, float, float]) -> bool:
    return a[1] < b[2] and b[1] < a[2]


@pytest.fixture(autouse=True)
def _reset():
    INTERVALS.clear()
    yield


def test_safe_tools_run_in_parallel():
    async def scenario():
        ex = StreamingToolExecutor(registry())
        for i in range(3):
            ex.submit(call(f"c{i}", "safe_tool", tag=str(i)))
        started = time.monotonic()
        outcomes = await ex.drain()
        return time.monotonic() - started, outcomes

    elapsed, outcomes = asyncio.run(scenario())

    assert len(outcomes) == 3
    assert all(not o.is_error for o in outcomes)
    # Serial would be 3 * DELAY. Parallel stays near one DELAY.
    assert elapsed < DELAY * 2, f"took {elapsed:.3f}s — tools did not overlap"


def test_unsafe_tool_is_a_barrier():
    """[safe, unsafe, safe] must not let the unsafe call overlap anything."""

    async def scenario():
        ex = StreamingToolExecutor(registry())
        ex.submit(call("c0", "safe_tool", tag="a"))
        ex.submit(call("c1", "unsafe_tool", tag="b"))
        ex.submit(call("c2", "safe_tool", tag="c"))
        return await ex.drain()

    outcomes = asyncio.run(scenario())
    assert [o.call_id for o in outcomes] == ["c0", "c1", "c2"]

    unsafe = next(i for i in INTERVALS if i[0].startswith("unsafe"))
    for other in INTERVALS:
        if other is unsafe:
            continue
        assert not overlaps(unsafe, other), f"{other[0]} overlapped the barrier"


def test_consecutive_safe_calls_after_a_barrier_still_parallelise():
    async def scenario():
        ex = StreamingToolExecutor(registry())
        ex.submit(call("c0", "unsafe_tool", tag="first"))
        ex.submit(call("c1", "safe_tool", tag="a"))
        ex.submit(call("c2", "safe_tool", tag="b"))
        return await ex.drain()

    asyncio.run(scenario())
    safes = [i for i in INTERVALS if i[0].startswith("safe")]
    assert len(safes) == 2
    assert overlaps(safes[0], safes[1]), "safe calls after the barrier should overlap"


def test_ledger_closes_on_success_failure_and_unknown_tool():
    async def scenario():
        ex = StreamingToolExecutor(registry())
        ex.submit(call("ok", "safe_tool", tag="x"))
        ex.submit(call("raises", "boom"))
        ex.submit(call("missing", "no_such_tool"))
        ex.submit(
            ToolCallReady(
                index=0,
                call_id="bad-args",
                name="safe_tool",
                arguments={},
                parse_error="could not parse arguments",
            )
        )
        return await ex.drain()

    outcomes = asyncio.run(scenario())

    assert [o.call_id for o in outcomes] == ["ok", "raises", "missing", "bad-args"]
    assert [o.is_error for o in outcomes] == [False, True, True, True]
    assert "tool exploded" in outcomes[1].result
    assert "no such tool" in outcomes[2].result


def test_every_issued_call_gets_exactly_one_result():
    async def scenario():
        ex = StreamingToolExecutor(registry())
        for i in range(5):
            ex.submit(call(f"c{i}", "safe_tool", tag=str(i)))
        outcomes = await ex.drain()
        return ex.issued, outcomes

    issued, outcomes = asyncio.run(scenario())
    assert issued == len(outcomes) == 5
    assert len({o.call_id for o in outcomes}) == 5


def test_cancel_still_closes_the_ledger():
    async def scenario():
        ex = StreamingToolExecutor(registry(), max_parallel=1)
        for i in range(4):
            ex.submit(call(f"c{i}", "safe_tool", tag=str(i)))
        await asyncio.sleep(0.01)  # let the first one start
        return ex.issued, await ex.cancel()

    issued, outcomes = asyncio.run(scenario())
    assert issued == len(outcomes) == 4
    assert any("cancel" in o.result for o in outcomes)


def test_timeout_produces_an_error_outcome_not_a_hang():
    async def scenario():
        ex = StreamingToolExecutor(registry(), timeout=0.05)
        ex.submit(call("slow", "forever"))
        return await ex.drain()

    outcomes = asyncio.run(scenario())
    assert len(outcomes) == 1
    assert outcomes[0].is_error
    assert "timed out" in outcomes[0].result


def test_outcome_converts_to_a_function_result_step():
    outcome_step = asyncio.run(_one_outcome()).to_step()
    assert outcome_step["type"] == "function_result"
    assert outcome_step["call_id"] == "c0"
    assert outcome_step["result"][0]["text"].startswith("safe:")
    assert "is_error" not in outcome_step


async def _one_outcome():
    ex = StreamingToolExecutor(registry())
    ex.submit(call("c0", "safe_tool", tag="z"))
    return (await ex.drain())[0]
