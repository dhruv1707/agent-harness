"""What is true when a tool call does not complete normally.

Much of this was settled in step 3 — the ledger closes from every exit, `block` tools are
awaited rather than killed, seven distinct reasons. These cover the six holes an audit
found afterwards.
"""

import asyncio

import pytest

from harness.config import MAX_TOOL_ERROR_BYTES
from harness.events import ToolCallReady
from harness.executor import StreamingToolExecutor
from harness.permissions import PermissionGate, PermissionPolicy
from harness.tools import Tool, ToolContext, ToolError, ToolRegistry


def run(coro):
    return asyncio.run(coro)


def call(call_id, name="t", **arguments):
    return ToolCallReady(call_id=call_id, name=name, arguments=arguments, index=0)


def make(fn, *, name="t", read_only=True, concurrency_safe=True, interrupt_behavior="cancel"):
    return Tool(
        name=name,
        description="",
        parameters={"type": "object", "properties": {}},
        fn=fn,
        concurrency_safe=concurrency_safe,
        read_only=read_only,
        interrupt_behavior=interrupt_behavior,
    )


def executor(tool, *, policy=None, asker=None, **kwargs):
    return StreamingToolExecutor(
        ToolRegistry([tool]),
        ctx=ToolContext(session_id="s"),
        gate=PermissionGate(policy or PermissionPolicy(allow=("*",)), asker=asker),
        **kwargs,
    )


# ---- 1. a remote error is an error --------------------------------------------


def test_a_tool_error_is_flagged_and_keeps_its_own_words():
    """An MCP server's error used to arrive as an ordinary return value, so a remote 401
    closed the ledger as a success with no error flag — and the memory gate counted it as
    one. `_field` exists because getting `isError` wrong reports failures as successes;
    this is the other half of that wiring."""

    def boom(**_kwargs):
        raise ToolError("tool error: upstream returned 401 unauthorized")

    async def scenario():
        ex = executor(make(boom))
        ex.submit(call("c1"))
        return await ex.drain()

    outcome = run(scenario())[0]

    assert outcome.is_error is True
    assert outcome.reason == "tool_error"
    assert outcome.result == "tool error: upstream returned 401 unauthorized", (
        "rendered verbatim, not wrapped in a class name the model does not need"
    )


def test_the_mcp_bridge_raises_rather_than_returning_its_errors():
    from harness.mcp import _render_result

    class Failed:
        is_error = True
        content = [type("B", (), {"text": "upstream returned 401"})()]

    with pytest.raises(ToolError, match="401"):
        _render_result(Failed())


# ---- 2. nothing waits forever -------------------------------------------------


def test_an_unanswered_prompt_denies_instead_of_hanging():
    """The gate serializes interactive prompts behind a lock, so one unanswered question
    used to park every sibling waiting to ask, and `drain()` with them."""

    def never_answers(_request):
        import time

        # Long relative to the 0.05s approval timeout, short enough not to stall the
        # suite — the thread outlives the timeout either way, which is the limitation
        # this fix works around rather than removes.
        time.sleep(2)
        return "y"

    async def scenario():
        ex = executor(
            make(lambda **_k: "ran"),
            policy=PermissionPolicy(ask=("t",)),
            asker=never_answers,
            approval_timeout=0.05,
        )
        ex.submit(call("c1"))
        return await asyncio.wait_for(ex.drain(), 5)

    outcome = run(scenario())[0]

    assert outcome.reason == "denied"
    assert "no answer within" in outcome.result


# ---- 3. a second interrupt keeps what closed ----------------------------------


def test_close_all_keeps_outcomes_that_already_closed():
    """The last resort when even cancelling is being interrupted: it awaits nothing and
    cannot raise, so a second Ctrl-C costs the remainder, never the lot."""
    async def scenario():
        ex = executor(make(lambda **_k: "done"))
        ex.submit(call("c1"))
        await ex.drain()
        ex.submit(call("c2"))
        return ex.close_all()

    outcomes = run(scenario())

    assert [o.call_id for o in outcomes] == ["c1", "c2"]
    assert outcomes[0].result == "done" and not outcomes[0].is_error
    assert outcomes[1].is_error and outcomes[1].reason == "user_interrupt"


def test_close_all_never_raises_on_an_incomplete_ledger():
    """`ledger()` raises when a call is unaccounted for — which is right, except inside
    interrupt handling, where raising is the one thing that cannot help."""
    async def scenario():
        ex = executor(make(lambda **_k: "done"))
        ex.submit(call("c1"))
        with pytest.raises(RuntimeError):
            ex.ledger()
        return ex.close_all()

    assert len(run(scenario())) == 1


def test_a_stuck_block_tool_cannot_make_the_interrupt_unkillable():
    async def forever(**_kwargs):
        await asyncio.sleep(30)
        return "finished"

    tool = make(forever, concurrency_safe=False, interrupt_behavior="block", read_only=False)

    async def scenario():
        ex = executor(tool, drain_timeout=0.05)
        ex.submit(call("c1"))
        await asyncio.sleep(0.01)
        return await ex.cancel()

    outcomes = run(asyncio.wait_for(scenario(), 5))

    assert len(outcomes) == 1 and outcomes[0].is_error


# ---- 5. say the true thing ----------------------------------------------------


def test_a_cancelled_write_admits_it_may_have_landed():
    """Cancelling unwinds our side of the call. It does not reach an MCP server already
    executing, and it cannot kill a worker thread."""

    async def forever(**_kwargs):
        await asyncio.sleep(30)

    async def scenario(read_only):
        ex = executor(make(forever, read_only=read_only))
        ex.submit(call("c1"))
        await asyncio.sleep(0.01)
        return await ex.cancel()

    wrote = run(scenario(read_only=False))[0]
    only_read = run(scenario(read_only=True))[0]

    assert "may have completed anyway" in wrote.result
    assert "may have completed anyway" not in only_read.result, "a read has nothing to undo"


# ---- 6. an error cannot flood the context -------------------------------------


def test_error_text_is_capped():
    def boom(**_kwargs):
        raise ValueError("x" * 100_000)

    async def scenario():
        ex = executor(make(boom))
        ex.submit(call("c1"))
        return await ex.drain()

    outcome = run(scenario())[0]

    assert len(outcome.result.encode("utf-8")) < MAX_TOOL_ERROR_BYTES + 200
    assert "elided" in outcome.result


def test_a_large_successful_result_is_not_capped():
    """A transcript is legitimately long, and pre-summary elision already bounds those."""
    async def scenario():
        ex = executor(make(lambda **_k: "y" * 100_000))
        ex.submit(call("c1"))
        return await ex.drain()

    assert len(run(scenario())[0].result) == 100_000
