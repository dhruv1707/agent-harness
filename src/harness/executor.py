"""The streaming tool executor.

Calls are submitted the moment they are complete — at their own `step.stop`, while the rest
of the model's output is still streaming. Three rules govern what happens next.

**Unsafe tools are barriers.** Walking the calls in arrival order, a concurrency-safe call
starts immediately (up to `max_parallel`); an unsafe call waits for everything in flight to
drain, runs alone, and only then does the queue resume. That parallelises consecutive safe
calls without reordering work around a call that cannot tolerate it — `[read, write, read]`
will not race both reads around the write.

**The runtime authorizes, the model only proposes.** Every call passes the permission gate
after partitioning and before execution, matching Claude Code's order (`runTools()` →
`partitionToolCalls()` → per-tool `runToolUse()`, which wraps permission around the call).

**A sibling's failure does not stop its siblings.** Each call is its own task; one raising
or being denied leaves the others running. That is deliberate and matches chapter 4's
matrix — *"one tool fails in parallel batch … keep others"* — rather than an absence. A
tool that must not run after a sibling failed does not exist yet; when one does, it needs a
way to say so, not a blanket fail-fast.

**The ledger always closes.** Every submitted call produces exactly one outcome, whether it
succeeded, was denied, raised, timed out, was never started, or was cancelled — each with a
distinct reason, because the model reads these and should be able to tell "a human declined
this" from "this timed out".
"""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from .config import (
    APPROVAL_TIMEOUT_SECONDS,
    INTERRUPT_DRAIN_SECONDS,
    MAX_PARALLEL_TOOLS,
    MAX_TOOL_ERROR_BYTES,
    TOOL_TIMEOUT_SECONDS,
)
from .events import ToolCallReady
from .permissions import PermissionGate, PermissionPolicy
from .tools import Tool, ToolContext, ToolError, ToolRegistry


class CallState(StrEnum):
    """Where a call is in its life.

    `yielded` — a tool emitting partial results mid-execution — is deliberately absent.
    None of our tools stream; it gets added when one does.
    """

    QUEUED = "queued"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    COMPLETED = "completed"
    DENIED = "denied"
    CANCELLED = "cancelled"


def _with_caveat(text: str, tool: Tool) -> str:
    """Say the true thing about what we did and did not stop.

    Cancelling a task unwinds our side of it — the HTTP request, the await. It does not
    reach an MCP server already executing, and it cannot kill a worker thread. For a tool
    that only reads, that distinction is academic. For one that changes something, telling
    the model nothing happened is telling it something false.
    """
    if tool.read_only:
        return text
    return f"{text} — the work may have completed anyway; check before retrying"


def _cap(text: str) -> str:
    """Bound an error message. A tool raising with a megabyte message would otherwise
    write a megabyte into the context, and the useful part is at the front."""
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_TOOL_ERROR_BYTES:
        return text
    head = encoded[:MAX_TOOL_ERROR_BYTES].decode("utf-8", "ignore")
    return f"{head}\n[… {len(encoded) - MAX_TOOL_ERROR_BYTES:,} bytes of error text elided]"


@dataclass(frozen=True)
class ToolOutcome:
    """One closed ledger entry."""

    call_id: str
    name: str
    result: str
    is_error: bool = False
    #: denied | user_interrupt | timeout | tool_error | not_started | unknown_tool | bad_arguments
    reason: str | None = None

    def to_step(self) -> dict:
        """The `function_result` step to send back to the model."""
        step: dict = {
            "type": "function_result",
            "call_id": self.call_id,
            "name": self.name,
            "result": [{"type": "text", "text": self.result}],
        }
        if self.is_error:
            step["is_error"] = True
        return step


class StreamingToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        ctx: ToolContext | None = None,
        gate: PermissionGate | None = None,
        max_parallel: int = MAX_PARALLEL_TOOLS,
        timeout: float = TOOL_TIMEOUT_SECONDS,
        approval_timeout: float = APPROVAL_TIMEOUT_SECONDS,
        drain_timeout: float = INTERRUPT_DRAIN_SECONDS,
    ):
        self.registry = registry
        self.ctx = ctx or ToolContext(session_id="local")
        # An explicit allow-everything policy rather than a None check, so a reader can
        # see that an ungated executor is ungated. The session always supplies a real one.
        self.gate = gate or PermissionGate(PermissionPolicy(default="allow"))
        self.max_parallel = max(1, max_parallel)
        self.timeout = timeout
        self.approval_timeout = approval_timeout
        self.drain_timeout = drain_timeout

        self._queue: deque[tuple[ToolCallReady, Tool]] = deque()
        self._running: dict[str, tuple[asyncio.Task, Tool]] = {}
        self._barrier_active = False
        self._issued: list[str] = []
        self._names: dict[str, str] = {}
        self._states: dict[str, CallState] = {}
        self._outcomes: dict[str, ToolOutcome] = {}
        self._idle = asyncio.Event()
        self._idle.set()

    # ---- submission ----------------------------------------------------------

    def submit(self, call: ToolCallReady) -> None:
        """Accept a completed call. Returns immediately; work starts as capacity allows."""
        self._issued.append(call.call_id)
        self._names[call.call_id] = call.name

        if call.parse_error:
            self._close(
                call.call_id, call.name, call.parse_error, True, reason="bad_arguments"
            )
            return

        try:
            tool = self.registry.get(call.name)
        except KeyError:
            self._close(
                call.call_id,
                call.name,
                f"no such tool: {call.name}",
                True,
                reason="unknown_tool",
            )
            return

        self._states[call.call_id] = CallState.QUEUED
        self._queue.append((call, tool))
        self._idle.clear()
        self._pump()

    @property
    def issued(self) -> int:
        return len(self._issued)

    @property
    def pending(self) -> int:
        return len(self._queue) + len(self._running)

    def states(self) -> dict[str, CallState]:
        """What every issued call is currently doing."""
        return dict(self._states)

    # ---- scheduling ----------------------------------------------------------

    def _pump(self) -> None:
        """Start whatever the barrier rule currently permits."""
        while self._queue:
            call, tool = self._queue[0]

            if self._barrier_active:
                return  # an unsafe tool owns the executor right now
            if not tool.concurrency_safe:
                if self._running:
                    return  # let the in-flight batch drain first
                self._barrier_active = True
            elif len(self._running) >= self.max_parallel:
                return

            self._queue.popleft()
            task = asyncio.create_task(self._run(call, tool))
            self._running[call.call_id] = (task, tool)

        if not self._running:
            self._idle.set()

    async def _run(self, call: ToolCallReady, tool: Tool) -> None:
        verdict_reached = False
        try:
            # Partitioned already; authorize before executing.
            self._states[call.call_id] = CallState.AWAITING_APPROVAL
            # Bounded. The gate serializes interactive prompts behind a lock, so one
            # unanswered question parks every sibling waiting to ask, and `drain()` with
            # them. Unbounded approval was the one place a call could hang forever.
            verdict = await asyncio.wait_for(
                self.gate.check(
                    call.call_id, call.name, call.arguments, read_only=tool.read_only
                ),
                self.approval_timeout,
            )
            verdict_reached = True
            if not verdict.allowed:
                self._states[call.call_id] = CallState.DENIED
                self._close(
                    call.call_id,
                    call.name,
                    f"permission denied: {verdict.reason}",
                    True,
                    reason="denied",
                )
                return

            self._states[call.call_id] = CallState.EXECUTING
            if inspect.iscoroutinefunction(tool.fn):
                result = await asyncio.wait_for(
                    tool.invoke(self.ctx, **call.arguments), self.timeout
                )
            else:
                # to_thread keeps a blocking tool off the event loop. Note the thread
                # itself cannot be killed on cancellation — we stop awaiting it, which is
                # enough to close the ledger and exit.
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.invoke, self.ctx, **call.arguments), self.timeout
                )
            self._states[call.call_id] = CallState.COMPLETED
            self._close(call.call_id, call.name, str(result))

        except asyncio.TimeoutError:
            self._states[call.call_id] = CallState.CANCELLED
            if not verdict_reached:  # the wait was for a person, not the tool
                self._close(
                    call.call_id,
                    call.name,
                    f"permission denied: no answer within {self.approval_timeout:g}s",
                    True,
                    reason="denied",
                )
            else:
                self._close(
                    call.call_id,
                    call.name,
                    _with_caveat(f"tool timed out after {self.timeout:g}s", tool),
                    True,
                    reason="timeout",
                )
        except asyncio.CancelledError:
            # Deliberately not re-raised: cancellation still owes the ledger an entry.
            self._states[call.call_id] = CallState.CANCELLED
            self._close(
                call.call_id,
                call.name,
                _with_caveat("interrupted by the operator", tool),
                True,
                reason="user_interrupt",
            )
        except ToolError as exc:
            # The tool wrote this message for the model; pass it through rather than
            # wrapping it in a class name it does not need to see.
            self._states[call.call_id] = CallState.CANCELLED
            self._close(call.call_id, call.name, str(exc), True, reason="tool_error")
        except Exception as exc:
            self._states[call.call_id] = CallState.CANCELLED
            self._close(
                call.call_id,
                call.name,
                f"{type(exc).__name__}: {exc}",
                True,
                reason="tool_error",
            )
        finally:
            self._running.pop(call.call_id, None)
            if not tool.concurrency_safe:
                self._barrier_active = False
            self._pump()

    def close_all(self, reason: str = "user_interrupt") -> list[ToolOutcome]:
        """Close every unclosed call and return the ledger, awaiting nothing.

        The last resort, for when even cancelling is being interrupted. It cannot hang and
        cannot raise, which is the point: a second Ctrl-C should cost the outcomes that had
        not closed yet, never the ones that had.
        """
        for call_id in self._issued:
            self._states.setdefault(call_id, CallState.CANCELLED)
            self._close(
                call_id,
                self._names.get(call_id, "?"),
                "interrupted before this call could be closed",
                True,
                reason=reason,
            )
        return [self._outcomes[cid] for cid in self._issued]

    def _close(
        self,
        call_id: str,
        name: str,
        result: str,
        is_error: bool = False,
        *,
        reason: str | None = None,
    ) -> None:
        self._outcomes.setdefault(
            call_id,
            ToolOutcome(
                call_id=call_id,
                name=name,
                result=_cap(result) if is_error else result,
                is_error=is_error,
                reason=reason,
            ),
        )

    # ---- completion ----------------------------------------------------------

    async def drain(self) -> list[ToolOutcome]:
        """Wait for every submitted call to finish, then return outcomes in issue order."""
        self._pump()
        while self._queue or self._running:
            await self._idle.wait()
        return self.ledger()

    @property
    def blocking(self) -> list[str]:
        """Calls that must finish before the runtime may accept anything new."""
        return [
            call_id
            for call_id, (_task, tool) in self._running.items()
            if tool.interrupt_behavior == "block"
        ]

    async def cancel(self) -> list[ToolOutcome]:
        """Abandon in-flight work but still close the ledger.

        A tool declaring `interrupt_behavior="block"` is awaited rather than cancelled —
        killing a half-written file is worse than waiting for it.
        """
        for call, _tool in self._queue:
            self._states[call.call_id] = CallState.CANCELLED
            self._close(
                call.call_id, call.name, "not started before the interrupt", True,
                reason="not_started",
            )
        self._queue.clear()

        for _call_id, (task, tool) in list(self._running.items()):
            if tool.interrupt_behavior == "cancel":
                task.cancel()
        if self._running:
            # Bounded. Letting a half-written file finish is the point of `block`; letting
            # a stuck writer make the interrupt unkillable is not. Past the deadline the
            # sweep below closes whatever is left.
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(task for task, _tool in self._running.values()),
                        return_exceptions=True,
                    ),
                    self.drain_timeout,
                )
            except asyncio.TimeoutError:
                pass
        self._running.clear()
        self._barrier_active = False
        self._idle.set()

        # A task cancelled before it ever ran never enters its body, so `_run` gets no
        # chance to record anything. Sweep here so ledger closure does not depend on the
        # coroutine having started.
        for call_id in self._issued:
            self._states.setdefault(call_id, CallState.CANCELLED)
            self._close(
                call_id, self._names.get(call_id, "?"), "interrupted by the operator", True,
                reason="user_interrupt",
            )

        return self.ledger()

    def ledger(self) -> list[ToolOutcome]:
        """Outcomes in the order the calls were issued — never completion order.

        This is what keeps context evolution deterministic while execution is parallel: a
        slow first call and a fast second call still land in the transcript in the order
        the model asked for them.

        Raises if any submitted call is unaccounted for — that would be a silent hole in
        the conversation the model is about to read.
        """
        missing = [cid for cid in self._issued if cid not in self._outcomes]
        if missing:
            raise RuntimeError(f"ledger not closed for calls: {missing}")
        return [self._outcomes[cid] for cid in self._issued]
