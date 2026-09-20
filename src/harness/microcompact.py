"""Clearing stale tool results from the working context.

Compaction is the nuclear option: a summarization call, a new root, the old history gone.
Microcompaction is the cheap one. No model call, no summary — old tool results are simply
replaced with a placeholder, in place, keeping the call/result pairing intact.

Two triggers, one mechanism:

**Time-based.** The gap since the last model output exceeds an hour, so the prompt cache
has expired and the whole prefix is about to be rewritten anyway. Clearing here is free:
we never force a miss that would not have happened. This is the resume-after-lunch case.

**Count.** More tool results have accumulated than the threshold allows. This fires while
the cache is still warm and does cost one uncached turn — but a cache miss costs you
whatever you send *after* the clearing, and clearing makes that much smaller. On an 80k
context where 60k is stale results, the miss turn processes 20k, which is less than a
cached 80k turn costs. The saving then repeats every turn.

That only holds while we remove more than we keep, which is what `MIN_RECLAIM_BYTES`
guards: clearing 5k while keeping 75k is three times worse than doing nothing.

Nothing here touches the transcript. The tree stays the complete record of what happened —
`verify` reads it, so every figure still grounds against the full tool output even after
the model has lost sight of it. Clearing weakens what the model reads, never what the
harness checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import (
    MICROCOMPACT_ENABLED,
    MICROCOMPACT_GAP_MINUTES,
    MICROCOMPACT_KEEP_RECENT,
    MICROCOMPACT_MIN_RECLAIM_BYTES,
    MICROCOMPACT_TOOL_THRESHOLD,
)

#: What replaces a cleared result. It has to read as "the answer was removed", not "there
#: was no answer" — the model should re-call the tool, not conclude the call failed or
#: invent what it would have said.
CLEARED_MESSAGE = "[tool result cleared to reclaim context — call this tool again if you still need it]"


@dataclass
class MicrocompactState:
    """Which results have been cleared, and what that bought.

    Lives on the `Session`, not the `LoopState`: a fresh `LoopState` is built for every
    `submit()`, so a cleared set held there would be forgotten between submissions and the
    next turn would restore everything the last one cleared.
    """

    #: Call ids already cleared. Sticky — once cleared, always cleared, so the prefix
    #: changes once and is byte-identical afterwards. A set that flip-flopped would
    #: invalidate the cache every single turn and never recover.
    cleared: set[str] = field(default_factory=set)
    clearings: int = 0
    reclaimed_bytes: int = 0


def _body(step: dict) -> str:
    """The text of a `function_result` step."""
    return " ".join(
        block.get("text", "")
        for block in step.get("result") or []
        if isinstance(block, dict)
    )


def _call_id(step: dict) -> str:
    """The call id, tolerating the shape asymmetry.

    A `function_call` carries `id`; its `function_result` carries `call_id`. Both appear
    in the projection, so anything walking it has to accept either.
    """
    return str(step.get("call_id") or step.get("id") or "")


def _results(messages: list[dict]) -> list[tuple[str, int]]:
    """Every uncleared `function_result` in order, as (call_id, body size)."""
    out: list[tuple[str, int]] = []
    for step in messages:
        if step.get("type") != "function_result":
            continue
        body = _body(step)
        if body == CLEARED_MESSAGE:
            continue  # already cleared; not a candidate, and not countable again
        out.append((_call_id(step), len(body)))
    return out


def select(messages: list[dict], keep_recent: int = MICROCOMPACT_KEEP_RECENT) -> list[str]:
    """Call ids to clear: everything but the most recent `keep_recent` results.

    `max(1, ...)` because `list[-0:]` is the whole list, which would keep everything and
    silently make the whole mechanism a no-op.
    """
    results = _results(messages)
    keep = max(1, keep_recent)
    return [call_id for call_id, _ in results[:-keep]] if len(results) > keep else []


def last_model_output_at(nodes: list[Any]) -> datetime | None:
    """When the model last said something, from the tree.

    `Node.ts` rather than an in-memory clock, because the gap that matters is measured
    across a resume — the three-hours-at-lunch case *is* a resume, and an in-process clock
    would read zero there.
    """
    for node in reversed(nodes):
        step = getattr(node, "step", None)
        if isinstance(step, dict) and step.get("type") == "model_output":
            try:
                return datetime.fromisoformat(getattr(node, "ts", ""))
            except (TypeError, ValueError):
                return None
    return None


def gap_minutes(nodes: list[Any], now: datetime | None = None) -> float | None:
    """Minutes since the last model output, or None if there has not been one."""
    last = last_model_output_at(nodes)
    if last is None:
        return None
    now = now or datetime.now(timezone.utc)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (now - last).total_seconds() / 60.0


def apply(messages: list[dict], call_ids: set[str]) -> tuple[list[dict], int]:
    """Replace the named results with the placeholder. Returns new messages and bytes freed.

    Builds replacements rather than editing in place. This is not style: `state.messages`
    holds the *same dict objects* as the transcript's nodes, so mutating one would rewrite
    history. Returns a new list, leaving the input untouched.
    """
    freed = 0
    out: list[dict] = []
    for step in messages:
        if step.get("type") != "function_result" or _call_id(step) not in call_ids:
            out.append(step)
            continue
        body = _body(step)
        if body == CLEARED_MESSAGE:
            out.append(step)  # idempotent: clearing twice must not double-count
            continue
        freed += len(body)
        out.append(
            {
                "type": "function_result",
                "call_id": step.get("call_id"),
                "name": step.get("name"),
                "result": [{"type": "text", "text": CLEARED_MESSAGE}],
                **({"is_error": True} if step.get("is_error") else {}),
            }
        )
    return out, freed


def reapply(messages: list[dict], state: MicrocompactState) -> list[dict]:
    """Re-clear what was already cleared, after the projection is rebuilt from the tree.

    `project()` walks the tree, and the tree still holds every result in full — so without
    this a compaction or a resume would silently restore everything microcompaction had
    cleared. The byte count is discarded: those bytes were already banked when the
    clearing first fired.
    """
    if not state.cleared:
        return messages
    out, _ = apply(messages, state.cleared)
    return out


def maybe_microcompact(
    messages: list[dict],
    nodes: list[Any],
    state: MicrocompactState,
    *,
    now: datetime | None = None,
    enabled: bool = MICROCOMPACT_ENABLED,
    gap_threshold: float = MICROCOMPACT_GAP_MINUTES,
    keep_recent: int = MICROCOMPACT_KEEP_RECENT,
    threshold: int = MICROCOMPACT_TOOL_THRESHOLD,
    min_reclaim: int = MICROCOMPACT_MIN_RECLAIM_BYTES,
) -> list[dict] | None:
    """Clear stale results if either trigger fires. Returns new messages, or None.

    Time-based is evaluated first and short-circuits, which is not merely an ordering
    preference: the count path assumes a warm cache and weighs a miss against the saving,
    and a large gap has just established the cache is cold. Running both would price a
    miss that already happened.
    """
    if not enabled:
        return None

    candidates = select(messages, keep_recent)
    if not candidates:
        return None

    gap = gap_minutes(nodes, now)
    cold = gap is not None and gap >= gap_threshold

    if not cold:
        # Warm cache: only worth a miss if we remove more than we keep.
        results = _results(messages)
        if len(results) <= threshold:
            return None
        reclaimable = sum(size for call_id, size in results if call_id in set(candidates))
        if reclaimable < min_reclaim:
            return None

    out, freed = apply(messages, set(candidates))
    if not freed:
        return None

    state.cleared.update(candidates)
    state.clearings += 1
    state.reclaimed_bytes += freed
    return out
