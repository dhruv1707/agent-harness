"""The query loop — the heartbeat.

Every iteration rebuilds the entire runtime context from state: system instruction, tool
declarations, and the conversation derived by walking the transcript from its head back to
the root. Nothing is carried implicitly between turns and nothing is stored server-side, so
the loop can be pointed at any node in the tree and will simply build a different request.

Termination is deliberate and enumerated. Retry and recovery branches belong to chapter 6;
until then the loop fails loudly rather than guessing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config import MAX_TURNS, MODEL
from .events import (
    StreamDone,
    StreamError,
    TextDelta,
    ThoughtDelta,
    ThoughtReady,
    ToolCallReady,
    ToolCallStarted,
    normalize,
)
from .executor import StreamingToolExecutor
from .prompt import AssembledPrompt
from .tools import ToolRegistry
from .transcript import Transcript


@dataclass(frozen=True)
class RuntimeContext:
    """Everything one model call needs. Rebuilt from scratch each iteration."""

    model: str
    system_instruction: str
    input: list[dict]
    tools: list[dict]
    store: bool = False

    def request(self) -> dict:
        return {
            "model": self.model,
            "system_instruction": self.system_instruction,
            "input": self.input,
            "tools": self.tools,
            "store": self.store,
            "stream": True,
        }


@dataclass
class LoopState:
    """What changes across turns. The transcript is the history; nothing shadows it."""

    transcript: Transcript
    turn: int = 0
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    text: str = ""


@dataclass(frozen=True)
class LoopResult:
    stop_reason: str
    turns: int
    text: str
    usage: dict
    error: str | None = None


def build_runtime(
    state: LoopState,
    prompt: AssembledPrompt,
    registry: ToolRegistry,
    model: str = MODEL,
) -> RuntimeContext:
    """Assemble the full runtime context for one turn."""
    return RuntimeContext(
        model=model,
        system_instruction=prompt.system_instruction,
        input=state.transcript.steps(),
        tools=registry.declarations(),
        store=False,
    )


async def query_loop(
    state: LoopState,
    *,
    client: Any,
    prompt: AssembledPrompt,
    registry: ToolRegistry,
    model: str = MODEL,
    max_turns: int = MAX_TURNS,
    on_text: Callable[[str], None] | None = None,
    on_event: Callable[[Any], None] | None = None,
) -> LoopResult:
    """Run until the model stops asking for tools, or a termination condition fires.

    `client` needs one thing: an awaitable `aio.interactions.create(**request)` returning an
    async iterator of SSE events. Tests supply a fake.
    """
    last_text = ""

    while True:
        if state.turn >= max_turns:
            state.stop_reason = "max_turns"
            return LoopResult("max_turns", state.turn, last_text, state.usage)

        state.turn += 1
        runtime = build_runtime(state, prompt, registry, model)
        executor = StreamingToolExecutor(registry)

        text_buffer: list[str] = []
        stream_error: str | None = None

        def flush_text() -> None:
            """Write buffered assistant text into the tree before any call that follows."""
            if not text_buffer:
                return
            joined = "".join(text_buffer)
            state.transcript.append(
                {"type": "model_output", "content": [{"type": "text", "text": joined}]},
                turn=state.turn,
            )
            text_buffer.clear()

        try:
            stream = await client.aio.interactions.create(**runtime.request())

            async for event in normalize(stream):
                if on_event is not None:
                    on_event(event)

                if isinstance(event, TextDelta):
                    text_buffer.append(event.text)
                    last_text += event.text
                    if on_text is not None:
                        on_text(event.text)

                elif isinstance(event, ThoughtDelta):
                    pass  # display only; the replayable form is ThoughtReady

                elif isinstance(event, ThoughtReady):
                    # Gemini 3 signs its thought steps and rejects the next request if the
                    # signature is not replayed with the history. We carry it verbatim.
                    flush_text()
                    thought_step: dict = {"type": "thought"}
                    if event.signature:
                        thought_step["signature"] = event.signature
                    if event.summary:
                        thought_step["summary"] = event.summary
                    state.transcript.append(thought_step, turn=state.turn)

                elif isinstance(event, ToolCallStarted):
                    pass  # arguments still streaming — nothing to do yet

                elif isinstance(event, ToolCallReady):
                    # Dispatch mid-stream. This is the whole point of the executor.
                    flush_text()
                    state.transcript.append(
                        {
                            "type": "function_call",
                            "id": event.call_id,
                            "name": event.name,
                            "arguments": event.arguments,
                        },
                        turn=state.turn,
                    )
                    executor.submit(event)

                elif isinstance(event, StreamError):
                    stream_error = event.message

                elif isinstance(event, StreamDone):
                    if event.usage:
                        state.usage = event.usage

        except (asyncio.CancelledError, KeyboardInterrupt):
            await _close_ledger(state, executor)
            state.stop_reason = "interrupted"
            return LoopResult("interrupted", state.turn, last_text, state.usage)

        except Exception as exc:  # transport, auth, malformed request
            await _close_ledger(state, executor)
            state.stop_reason = "api_error"
            return LoopResult(
                "api_error", state.turn, last_text, state.usage, f"{type(exc).__name__}: {exc}"
            )

        flush_text()

        if stream_error is not None:
            # Per chapter 3: API errors return directly. No retry policy exists yet.
            await _close_ledger(state, executor)
            state.stop_reason = "api_error"
            return LoopResult("api_error", state.turn, last_text, state.usage, stream_error)

        if executor.issued == 0:
            state.stop_reason = "end_turn"
            return LoopResult("end_turn", state.turn, last_text, state.usage)

        try:
            outcomes = await executor.drain()
        except (asyncio.CancelledError, KeyboardInterrupt):
            await _close_ledger(state, executor)
            state.stop_reason = "interrupted"
            return LoopResult("interrupted", state.turn, last_text, state.usage)

        for outcome in outcomes:
            state.transcript.append(outcome.to_step(), turn=state.turn)


async def _close_ledger(state: LoopState, executor: StreamingToolExecutor) -> None:
    """Write a result for every issued call, however the turn ended."""
    if executor.issued == 0:
        return
    try:
        outcomes = await executor.cancel()
    except Exception:  # noqa: BLE001 - a failed cancel must not mask the original exit
        return
    for outcome in outcomes:
        state.transcript.append(outcome.to_step(), turn=state.turn)
