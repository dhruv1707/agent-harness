"""A size gate in front of the context, for tool results that are too big on arrival.

Compaction summarizes what is already there; microcompaction clears what has gone stale.
Both act on results the model has been carrying for a while. This acts before that: a
single 284 KB listing is a quarter of the budget the moment it lands, and no amount of
later cleanup gets that turn back.

Two gates:

**Per result.** Over `DEFAULT_MAX_RESULT_SIZE_CHARS` the full text is written to a file
and the model is sent a preview plus the path, which it can read back with
`read_tool_result`. A tool may declare a lower ceiling or opt out entirely.

**Per group.** One round of parallel calls may return several results that are each under
the per-result ceiling and still crush the context together. Over
`MAX_TOOL_RESULTS_PER_MESSAGE_CHARS` the largest are persisted, biggest first, until the
group fits.

Both run before every request rather than once when results are recorded, which keeps the
transcript byte-for-byte complete — `verify` grounds figures against the tree, so the tree
must keep everything the model is no longer being shown.

Enforcing repeatedly is what makes the tri-state necessary. Once the model has seen a
result, that decision can never change: altering it would move the prefix and throw away
the cache for everything after it. So each candidate is in exactly one state:

    fresh        never sent      -> may be replaced
    frozen       sent in full    -> must stay in full
    mustReapply  sent replaced   -> must get the identical replacement back

If the frozen results alone bust the budget, we take the overage. Microcompaction clears
them later on age; there is no correct way to reclaim them now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_MAX_RESULT_SIZE_CHARS,
    MAX_TOOL_RESULTS_PER_MESSAGE_CHARS,
    PREVIEW_BYTES,
)


@dataclass
class ContentReplacementState:
    """What the model has been shown, so it can never be shown something else.

    Lives on the `Session` beside `PlanState` and `MicrocompactState`, because a fresh
    `LoopState` is built per submission and a record's fate must outlive that.
    """

    #: Call ids that have been in a request. Everything here is frozen or mustReapply.
    seen: set[str] = field(default_factory=set)
    #: call_id -> the exact replacement string. Re-applied verbatim, never regenerated:
    #: regenerating risks a byte differing, and a byte differing costs the cache.
    replacements: dict[str, str] = field(default_factory=dict)
    persisted: int = 0
    reclaimed_chars: int = 0


def _body(step: dict) -> str:
    return " ".join(
        block.get("text", "")
        for block in step.get("result") or []
        if isinstance(block, dict)
    )


def _call_id(step: dict) -> str:
    """Tolerates the shape asymmetry: calls carry `id`, results carry `call_id`."""
    return str(step.get("call_id") or step.get("id") or "")


def _replace(step: dict, text: str) -> dict:
    """A result step with a new body, rebuilt rather than mutated.

    The step dicts in the projection are the same objects the transcript's nodes hold, so
    editing one in place would rewrite history.
    """
    return {
        "type": "function_result",
        "call_id": step.get("call_id"),
        "name": step.get("name"),
        "result": [{"type": "text", "text": text}],
        **({"is_error": True} if step.get("is_error") else {}),
    }


# ---- grouping ----------------------------------------------------------------


def groups(messages: list[dict], nodes: list[Any]) -> list[list[int]]:
    """Indices of each round of parallel results, as the API will see them.

    A group is a run of consecutive `function_result` steps sharing a turn. Consecutive
    because the executor's ledger is flushed as a unit, in issue order, so a turn's
    results land together with nothing between them. Same turn because the loop stamps
    one turn per model response — that is the equivalent of an assistant message id, and
    it is already on the node when the step is written.

    Requiring both matters: `Node.turn` restarts at every submission, so two rounds can
    both be "turn 1" on one path — but they can never be adjacent, because a `user_input`
    separates them.
    """
    out: list[list[int]] = []
    current: list[int] = []
    current_turn: Any = None

    for index, step in enumerate(messages):
        if step.get("type") != "function_result":
            if current:
                out.append(current)
                current, current_turn = [], None
            continue
        turn = getattr(nodes[index], "turn", None) if index < len(nodes) else None
        if current and turn != current_turn:
            out.append(current)
            current = []
        current, current_turn = current + [index], turn

    if current:
        out.append(current)
    return out


# ---- persistence -------------------------------------------------------------


def results_dir(runs_dir: Path, session_id: str) -> Path:
    """Where persisted results live, following the `<runs_dir>/<session_id>-…` idiom."""
    return runs_dir / f"{session_id}-tool-results"


def preview(text: str, limit: int = PREVIEW_BYTES) -> str:
    """The head of `text`, cut at a newline when that does not lose too much.

    Falling back to the exact limit is the common path rather than the edge case here:
    our largest results are single-line JSON with no newline to cut at.
    """
    head = text[:limit]
    cut = head.rfind("\n")
    return head[:cut] if cut > limit // 2 else head


def persist(text: str, call_id: str, directory: Path) -> Path:
    """Write the full result to a file. Returns the path.

    Exclusive create, so re-persisting the same call is a no-op instead of an error. The
    history is replayed on every request build, and a `call_id` is unique per invocation,
    so the content behind an id never changes — skipping an existing file is always safe.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{call_id}.txt"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError:
        pass
    return path


def build_replacement(text: str, name: str, path: Path) -> str:
    """What the model sees instead of the full result."""
    kb = len(text.encode("utf-8")) / 1024
    return (
        "<persisted-output>\n"
        f"Output too large ({kb:,.1f} KB). Full output saved to:\n"
        f"  {path}\n\n"
        f"Read it back with read_tool_result(call_id=\"{path.stem}\").\n\n"
        f"Preview (first {PREVIEW_BYTES / 1024:.1f} KB):\n"
        f"{preview(text)}\n"
        "…\n"
        "</persisted-output>"
    )


# ---- the gates ---------------------------------------------------------------


def ceiling_for(name: str, registry: Any) -> int | None:
    """Chars this tool's results may occupy, or None if it opts out of persistence.

    A tool declaring a limit gets the lower of its own and the global one; a tool
    declaring `None` is never persisted at any size. `read_tool_result` opts out for the
    reason the chapter's Read tool does: persisting the thing that reads persisted output
    sends the model round the same loop again.
    """
    try:
        declared = getattr(registry.get(name), "max_result_chars", DEFAULT_MAX_RESULT_SIZE_CHARS)
    except Exception:  # noqa: BLE001 - an MCP tool that has since gone away takes the default
        return DEFAULT_MAX_RESULT_SIZE_CHARS
    if declared is None:
        return None
    return min(declared, DEFAULT_MAX_RESULT_SIZE_CHARS)


def select_fresh(sizes: dict[int, int], frozen_total: int, limit: int) -> list[int]:
    """Which fresh results to persist: largest first, until the group fits.

    Stops as soon as the remainder is under the limit, so we persist the fewest results
    that solve the problem rather than everything over some line.
    """
    total = frozen_total + sum(sizes.values())
    chosen: list[int] = []
    for index in sorted(sizes, key=lambda i: sizes[i], reverse=True):
        if total <= limit:
            break
        chosen.append(index)
        total -= sizes[index]
    return chosen


def apply(
    messages: list[dict],
    nodes: list[Any],
    state: ContentReplacementState,
    *,
    registry: Any,
    runs_dir: Path,
    session_id: str,
    group_limit: int = MAX_TOOL_RESULTS_PER_MESSAGE_CHARS,
) -> list[dict]:
    """Enforce both gates over the projection. Returns new messages; the input is untouched.

    Marking and replacement happen together. The chapter marks unselected candidates as
    seen synchronously but selected ones only after an async write, because a mismatch
    between the two would classify a result as frozen while its preview was already in
    flight. Our write is synchronous inside a single request build, so the window in which
    they could disagree does not exist.
    """
    out = list(messages)
    directory = results_dir(runs_dir, session_id)

    for group in groups(messages, nodes):
        fresh: dict[int, int] = {}
        frozen_total = 0

        for index in group:
            step = messages[index]
            call_id = _call_id(step)

            if call_id in state.replacements:  # mustReapply — verbatim, no I/O
                out[index] = _replace(step, state.replacements[call_id])
                continue

            size = len(_body(step))
            if call_id in state.seen:  # frozen — replacing now would move the prefix
                frozen_total += size
                continue

            ceiling = ceiling_for(step.get("name") or "", registry)
            if ceiling is None:
                # Opted out. It still occupies the group's budget — it is simply not a
                # candidate for reclaiming it, which is the same arithmetic as frozen.
                frozen_total += size
                state.seen.add(call_id)
                continue
            if size > ceiling:
                out[index] = _persist_one(step, state, directory)  # over on its own
                continue
            fresh[index] = size

        for index in select_fresh(fresh, frozen_total, group_limit):
            out[index] = _persist_one(messages[index], state, directory)
            del fresh[index]

        # Everything still in `fresh` is going out at full size, so it freezes here.
        for index in fresh:
            state.seen.add(_call_id(messages[index]))

    return out


def _persist_one(step: dict, state: ContentReplacementState, directory: Path) -> dict:
    """Write one result out and record the decision. Returns the replacement step."""
    call_id = _call_id(step)
    body = _body(step)
    path = persist(body, call_id, directory)
    replacement = build_replacement(body, step.get("name") or "tool", path)

    state.replacements[call_id] = replacement
    state.seen.add(call_id)
    state.persisted += 1
    state.reclaimed_chars += len(body) - len(replacement)
    return _replace(step, replacement)


def reapply(messages: list[dict], state: ContentReplacementState) -> list[dict]:
    """Re-apply known replacements after the projection is rebuilt from the tree.

    `project()` walks a tree that still holds every result in full, so without this a
    branch, compaction or resume would quietly put the payloads back.
    """
    if not state.replacements:
        return messages
    out: list[dict] = []
    for step in messages:
        replacement = (
            state.replacements.get(_call_id(step))
            if step.get("type") == "function_result"
            else None
        )
        out.append(_replace(step, replacement) if replacement else step)
    return out
