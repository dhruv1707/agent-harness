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

from .config import MAX_PARALLEL_TOOLS, TOOL_TIMEOUT_SECONDS
from .events import ToolCallReady
from .permissions import PermissionGate, PermissionPolicy
from .tools import Tool, ToolContext, ToolRegistry


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
    ):
        self.registry = registry
        self.ctx = ctx or ToolContext(session_id="local")
        # An explicit allow-everything policy rather than a None check, so a reader can
        # see that an ungated executor is ungated. The session always supplies a real one.
        self.gate = gate or PermissionGate(PermissionPolicy(default="allow"))
        self.max_parallel = max(1, max_parallel)
        self.timeout = timeout

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
        try:
            # Partitioned already; authorize before executing.
            self._states[call.call_id] = CallState.AWAITING_APPROVAL
            verdict = await self.gate.check(
                call.call_id, call.name, call.arguments, read_only=tool.read_only
            )
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
            self._close(
                call.call_id,
                call.name,
                f"tool timed out after {self.timeout:g}s",
                True,
                reason="timeout",
            )
        except asyncio.CancelledError:
            # Deliberately not re-raised: cancellation still owes the ledger an entry.
            self._states[call.call_id] = CallState.CANCELLED
            self._close(
                call.call_id, call.name, "interrupted by the operator", True,
                reason="user_interrupt",
            )
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
                call_id=call_id, name=name, result=result, is_error=is_error, reason=reason
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
            await asyncio.gather(
                *(task for task, _tool in self._running.values()), return_exceptions=True
            )
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
