"""The streaming tool executor.

Calls are submitted the moment they are complete — at their own `step.stop`, while the rest
of the model's output is still streaming. Scheduling follows one rule:

    **unsafe tools are barriers.**

Walking the calls in arrival order, a concurrency-safe call starts immediately (up to
`max_parallel`); an unsafe call waits for everything in flight to drain, runs alone, and
only then does the queue resume. That parallelises consecutive safe calls without ever
reordering work around a call that cannot tolerate it — `[read, write, read]` will not race
both reads around the write.

The other guarantee is the ledger: **every submitted call produces exactly one outcome**,
whether it succeeded, raised, timed out, was never started, or was cancelled. A turn that
cannot close its ledger is a bug.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from dataclasses import dataclass

from .config import MAX_PARALLEL_TOOLS, TOOL_TIMEOUT_SECONDS
from .events import ToolCallReady
from .tools import Tool, ToolRegistry


@dataclass(frozen=True)
class ToolOutcome:
    """One closed ledger entry."""

    call_id: str
    name: str
    result: str
    is_error: bool = False

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
        max_parallel: int = MAX_PARALLEL_TOOLS,
        timeout: float = TOOL_TIMEOUT_SECONDS,
    ):
        self.registry = registry
        self.max_parallel = max(1, max_parallel)
        self.timeout = timeout

        self._queue: deque[tuple[ToolCallReady, Tool]] = deque()
        self._running: dict[str, asyncio.Task] = {}
        self._barrier_active = False
        self._issued: list[str] = []
        self._names: dict[str, str] = {}
        self._outcomes: dict[str, ToolOutcome] = {}
        self._idle = asyncio.Event()
        self._idle.set()

    # ---- submission ----------------------------------------------------------

    def submit(self, call: ToolCallReady) -> None:
        """Accept a completed call. Returns immediately; work starts as capacity allows."""
        self._issued.append(call.call_id)
        self._names[call.call_id] = call.name

        if call.parse_error:
            self._close(call.call_id, call.name, call.parse_error, is_error=True)
            return

        try:
            tool = self.registry.get(call.name)
        except KeyError:
            self._close(
                call.call_id, call.name, f"no such tool: {call.name}", is_error=True
            )
            return

        self._queue.append((call, tool))
        self._idle.clear()
        self._pump()

    @property
    def issued(self) -> int:
        return len(self._issued)

    @property
    def pending(self) -> int:
        return len(self._queue) + len(self._running)

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
            self._running[call.call_id] = task

        if not self._running:
            self._idle.set()

    async def _run(self, call: ToolCallReady, tool: Tool) -> None:
        try:
            if inspect.iscoroutinefunction(tool.fn):
                result = await asyncio.wait_for(tool.fn(**call.arguments), self.timeout)
            else:
                # to_thread keeps a blocking tool off the event loop. Note the thread
                # itself cannot be killed on cancellation — we stop awaiting it, which is
                # enough to close the ledger and exit.
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.fn, **call.arguments), self.timeout
                )
            self._close(call.call_id, call.name, str(result))
        except asyncio.TimeoutError:
            self._close(
                call.call_id,
                call.name,
                f"tool timed out after {self.timeout:g}s",
                is_error=True,
            )
        except asyncio.CancelledError:
            # Deliberately not re-raised: cancellation still owes the ledger an entry.
            self._close(call.call_id, call.name, "cancelled", is_error=True)
        except Exception as exc:
            self._close(
                call.call_id, call.name, f"{type(exc).__name__}: {exc}", is_error=True
            )
        finally:
            self._running.pop(call.call_id, None)
            if not tool.concurrency_safe:
                self._barrier_active = False
            self._pump()

    def _close(self, call_id: str, name: str, result: str, is_error: bool = False) -> None:
        self._outcomes.setdefault(
            call_id, ToolOutcome(call_id=call_id, name=name, result=result, is_error=is_error)
        )

    # ---- completion ----------------------------------------------------------

    async def drain(self) -> list[ToolOutcome]:
        """Wait for every submitted call to finish, then return outcomes in issue order."""
        self._pump()
        while self._queue or self._running:
            await self._idle.wait()
        return self.ledger()

    async def cancel(self) -> list[ToolOutcome]:
        """Abandon in-flight work but still close the ledger."""
        for call, _tool in self._queue:
            self._close(call.call_id, call.name, "cancelled before start", is_error=True)
        self._queue.clear()

        for task in list(self._running.values()):
            task.cancel()
        if self._running:
            await asyncio.gather(*self._running.values(), return_exceptions=True)
        self._running.clear()
        self._barrier_active = False
        self._idle.set()

        # A task cancelled before it ever ran never enters its body, so `_run` gets no
        # chance to record anything. Sweep here so ledger closure does not depend on the
        # coroutine having started.
        for call_id in self._issued:
            self._close(call_id, self._names.get(call_id, "?"), "cancelled", is_error=True)

        return self.ledger()

    def ledger(self) -> list[ToolOutcome]:
        """Outcomes in the order the calls were issued.

        Raises if any submitted call is unaccounted for — that would be a silent hole in
        the conversation the model is about to read.
        """
        missing = [cid for cid in self._issued if cid not in self._outcomes]
        if missing:
            raise RuntimeError(f"ledger not closed for calls: {missing}")
        return [self._outcomes[cid] for cid in self._issued]
