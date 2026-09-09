"""Normalized stream events, and the adapter that produces them from the SDK.

The loop consumes these, never the SDK's own event objects. Two reasons: we have already
swapped providers once on this project, and — more usefully — a normalized event type means
the loop and executor can be driven by a scripted stream in tests, with no network and no
tokens spent.

The adapter also owns argument accumulation. `arguments_delta` events carry partial JSON
that must be concatenated; by the time the loop sees a call it is whole, parsed, and safe
to execute.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


# ---- normalized events -------------------------------------------------------


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThoughtDelta:
    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    index: int
    call_id: str
    name: str


@dataclass(frozen=True)
class ToolCallReady:
    """A complete, parsed call. Emitted at the call's own step.stop — mid-stream."""

    index: int
    call_id: str
    name: str
    arguments: dict
    parse_error: str | None = None


@dataclass(frozen=True)
class ThoughtReady:
    """A completed thought step.

    Gemini 3 returns a signed `thought` step alongside tool calls, and the signature must
    be replayed with the history or the next request is rejected as invalid. We never read
    it — we just carry it faithfully.
    """

    index: int
    signature: str | None = None
    summary: list = field(default_factory=list)


@dataclass(frozen=True)
class StreamError:
    message: str
    raw: Any = None


@dataclass(frozen=True)
class StreamDone:
    usage: dict = field(default_factory=dict)
    interaction_id: str | None = None


Event = (
    TextDelta
    | ThoughtDelta
    | ThoughtReady
    | ToolCallStarted
    | ToolCallReady
    | StreamError
    | StreamDone
)


# ---- adapter -----------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from an SDK object or a plain dict.

    The SDK returns pydantic models; tests feed dicts. Both work.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class _PendingThought:
    index: int
    signature: str | None = None
    summary: list = field(default_factory=list)


@dataclass
class _PendingCall:
    index: int
    call_id: str
    name: str
    fragments: list[str] = field(default_factory=list)
    #: Some calls arrive complete on step.start with no argument deltas at all.
    seeded: dict | None = None

    def parse(self) -> tuple[dict, str | None]:
        raw = "".join(self.fragments).strip()
        if not raw:
            return dict(self.seeded or {}), None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {}, f"could not parse arguments: {exc}"
        if not isinstance(parsed, dict):
            return {}, f"arguments were {type(parsed).__name__}, expected object"
        return parsed, None


async def normalize(stream: AsyncIterator[Any]) -> AsyncIterator[Event]:
    """Translate the SDK's SSE events into normalized events.

    Yields `ToolCallReady` as soon as a call's `step.stop` arrives, which is what lets the
    executor start work while the rest of the stream is still arriving.
    """
    pending: dict[int, _PendingCall] = {}
    thoughts: dict[int, _PendingThought] = {}
    usage: dict = {}
    interaction_id: str | None = None

    async for raw_event in stream:
        kind = _get(raw_event, "event_type") or _get(raw_event, "type")
        index = _get(raw_event, "index", 0) or 0

        if kind == "interaction.created":
            interaction_id = _get(_get(raw_event, "interaction"), "id") or _get(raw_event, "id")

        elif kind == "step.start":
            step = _get(raw_event, "step")
            step_type = _get(step, "type")
            if step_type == "thought":
                thoughts[index] = _PendingThought(index=index)
            elif step_type == "function_call":
                seeded = _get(step, "arguments")
                pending[index] = _PendingCall(
                    index=index,
                    call_id=_get(step, "id") or "",
                    name=_get(step, "name") or "",
                    seeded=seeded if isinstance(seeded, dict) else None,
                )
                yield ToolCallStarted(
                    index=index, call_id=pending[index].call_id, name=pending[index].name
                )

        elif kind == "step.delta":
            delta = _get(raw_event, "delta")
            delta_type = _get(delta, "type")
            if delta_type == "arguments_delta":
                fragment = _get(delta, "arguments") or _get(delta, "partial_arguments") or ""
                if index in pending:
                    pending[index].fragments.append(fragment)
            elif delta_type == "text":
                text = _get(delta, "text") or ""
                if text:
                    yield TextDelta(text)
            elif delta_type == "thought_signature":
                signature = _get(delta, "signature")
                if index in thoughts and signature:
                    thoughts[index].signature = signature
            elif delta_type == "thought_summary":
                text = _get(delta, "text") or _get(delta, "thought_summary") or ""
                if index in thoughts and text:
                    thoughts[index].summary.append({"type": "text", "text": text})
                if text:
                    yield ThoughtDelta(text)

        elif kind == "step.stop":
            thought = thoughts.pop(index, None)
            if thought is not None:
                yield ThoughtReady(
                    index=thought.index,
                    signature=thought.signature,
                    summary=thought.summary,
                )

            call = pending.pop(index, None)
            if call is not None:
                arguments, parse_error = call.parse()
                yield ToolCallReady(
                    index=call.index,
                    call_id=call.call_id,
                    name=call.name,
                    arguments=arguments,
                    parse_error=parse_error,
                )

        elif kind == "interaction.completed":
            interaction = _get(raw_event, "interaction")
            usage = _get(interaction, "usage") or _get(raw_event, "usage") or {}
            if not isinstance(usage, dict):
                usage = {
                    key: getattr(usage, key)
                    for key in dir(usage)
                    if not key.startswith("_") and isinstance(getattr(usage, key, None), int)
                }
            interaction_id = interaction_id or _get(interaction, "id")

        elif kind == "error":
            error = _get(raw_event, "error") or raw_event
            yield StreamError(message=str(_get(error, "message") or error), raw=error)
            return

    # A call left pending means the stream ended mid-arguments. Surface it rather than
    # dropping it — the ledger has to close either way.
    for call in pending.values():
        arguments, parse_error = call.parse()
        # The truncation is the cause; any JSON error is only its symptom. Lead with the
        # cause and keep the detail.
        detail = f" ({parse_error})" if parse_error else ""
        yield ToolCallReady(
            index=call.index,
            call_id=call.call_id,
            name=call.name,
            arguments=arguments,
            parse_error=f"stream ended before the call completed{detail}",
        )

    yield StreamDone(usage=usage, interaction_id=interaction_id)
