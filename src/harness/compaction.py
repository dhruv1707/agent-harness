"""Compaction: a controlled reboot, not a chat recap.

Old context is translated into new operating substrate rather than summarized for a
reader. Four phases:

1. **Pre-summary cleaning.** Expensive, low-summary-value content is replaced by labels
   before the summarization call sees it: `[tool result: …]`, `[transcript: …]` and
   `[thought]`. This copy is
   throwaway — the real transcript is never rewritten.
2. **Summarize.** One model call, no tools, filling the session-memory template. If that
   call itself hits prompt-too-long, the head is truncated and it retries once.
3. **Boundary.** A new root node holding the summary.
4. **Reinject.** The retained recent steps, re-appended under it.
"""

from __future__ import annotations

import json
from typing import Any

from .config import KEEP_RECENT_SHARE
from .verify import (
    ids_mentioned,
    index_sources,
    mark_unverified_quotes,
    render_measurements,
    repair,
    sources_from_steps,
)
from .session_memory import SessionMemory, WRITE_PROMPT, render_previous

#: Tool results and outputs longer than this are labelled rather than sent for
#: summarization. Small results are cheap and often carry the actual finding.
ELIDE_OVER_BYTES = 1_500

#: Phrases the providers use for a context-window overflow.
_PTL_MARKERS = ("too long", "token limit", "exceeds the maximum", "context length", "too large")


def is_prompt_too_long(error: BaseException | str) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _PTL_MARKERS)


# ---- 1. pre-summary cleaning -------------------------------------------------


def _step_text(step: dict) -> str:
    if step.get("type") in ("user_input", "model_output"):
        return " ".join(
            block.get("text", "")
            for block in step.get("content") or []
            if isinstance(block, dict)
        )
    if step.get("type") == "function_result":
        return " ".join(
            block.get("text", "")
            for block in step.get("result") or []
            if isinstance(block, dict)
        )
    return ""


def strip_for_summary(steps: list[dict]) -> list[dict]:
    """Replace expensive content with labels. Returns a copy; the input is untouched."""
    cleaned: list[dict] = []
    for index, step in enumerate(steps):
        kind = step.get("type")

        # The first step is the root of the walk, and on every compaction after the first
        # that root is the *previous boundary* — the task, the attachments and the brief.
        # Eliding it at 1,500 bytes threw away everything a 12,000-token brief had just
        # been written to preserve, `## Next` included, on the second compaction of any
        # long run. A summary of a summary is the one thing that must not be truncated.
        if index == 0 and kind == "user_input":
            cleaned.append(step)
            continue

        if kind == "thought":
            # Signatures are large base64 blobs and carry nothing a summary can use.
            cleaned.append({"type": "thought", "elided": True})
            continue

        if kind == "function_result":
            body = _step_text(step)
            if len(body) > ELIDE_OVER_BYTES:
                name = step.get("name") or step.get("call_id") or "tool"
                label = (
                    f"[transcript: {name} — {len(body):,} bytes elided]"
                    if "transcript" in str(name)
                    else f"[tool result: {name} — {len(body):,} bytes elided]"
                )
                cleaned.append(
                    {
                        "type": "function_result",
                        "call_id": step.get("call_id"),
                        "name": step.get("name"),
                        "result": [{"type": "text", "text": label}],
                        **({"is_error": True} if step.get("is_error") else {}),
                    }
                )
                continue

        if kind in ("user_input", "model_output"):
            body = _step_text(step)
            if len(body) > ELIDE_OVER_BYTES:
                head = body[:ELIDE_OVER_BYTES]
                cleaned.append(
                    {
                        "type": kind,
                        "content": [
                            {
                                "type": "text",
                                "text": f"{head}\n[… {len(body) - len(head):,} bytes elided]",
                            }
                        ],
                    }
                )
                continue

        cleaned.append(step)
    return cleaned


def render_for_summary(steps: list[dict]) -> str:
    """Flatten the cleaned history into something a summarizer can read."""
    lines: list[str] = []
    for step in steps:
        kind = step.get("type")
        if kind == "thought":
            continue
        if kind == "function_call":
            args = json.dumps(step.get("arguments", {}))[:400]
            lines.append(f"CALL {step.get('name')}({args})")
        elif kind == "function_result":
            flag = " [error]" if step.get("is_error") else ""
            lines.append(f"RESULT {step.get('name')}{flag}: {_step_text(step)[:1500]}")
        elif kind == "user_input":
            lines.append(f"USER: {_step_text(step)}")
        elif kind == "model_output":
            lines.append(f"ASSISTANT: {_step_text(step)}")
    return "\n\n".join(lines)


# ---- 2. where to cut ---------------------------------------------------------


def plan_cut(nodes: list[Any], keep_share: float = KEEP_RECENT_SHARE) -> int:
    """Index of the first node to keep verbatim, snapped to a turn boundary.

    A `function_call` must never be separated from its `function_result` — the model would
    read a call with no answer. Snapping backwards to the start of the turn keeps every
    pair intact, since the loop stamps both with the same turn number.
    """
    if not nodes:
        return 0
    keep = max(1, int(len(nodes) * keep_share))
    index = max(0, len(nodes) - keep)

    turn = getattr(nodes[index], "turn", None)
    while index > 0 and getattr(nodes[index - 1], "turn", None) == turn:
        index -= 1
    return index


# ---- 3. the summarization call -----------------------------------------------


async def summarize(
    client: Any,
    model: str,
    steps: list[dict],
    previous: SessionMemory | None = None,
) -> SessionMemory:
    """One model call, no tools. Retries once with a truncated head on prompt-too-long."""
    cleaned = strip_for_summary(steps)

    async def attempt(history: list[dict]) -> str:
        request = {
            "model": model,
            "system_instruction": WRITE_PROMPT,
            "input": [
                {
                    "type": "user_input",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{render_previous(previous)}\n\n"
                                f"SESSION HISTORY:\n\n{render_for_summary(history)}"
                            ),
                        }
                    ],
                }
            ],
            "store": False,
            "stream": False,
        }
        response = await client.aio.interactions.create(**request)
        return _extract_text(response)

    try:
        text = await attempt(cleaned)
    except Exception as exc:
        if not is_prompt_too_long(exc):
            raise
        # The compact request itself overflowed. Drop the oldest half and try once more.
        text = await attempt(cleaned[len(cleaned) // 2 :])

    brief = SessionMemory.parse(text)

    # The summarizer read a history whose large tool results were truncated, so its prose
    # can carry figures it never saw. Everything below is transcription from the untouched
    # steps: contradictions in the prose are corrected against the record, and the metrics
    # section is written from the record outright.
    sources = sources_from_steps(steps)
    for name, body in list(brief.sections.items()):
        if body:
            body, _ = repair(body, sources)
            brief.sections[name], _ = mark_unverified_quotes(body, sources)
    brief.sections["Measurements"] = render_measurements(
        index_sources(sources), ids_mentioned(brief.render())
    )
    return brief.enforce_budgets()


def _extract_text(response: Any) -> str:
    """Pull the assistant text out of a non-streaming interaction response."""
    for step in getattr(response, "steps", None) or []:
        if getattr(step, "type", None) == "model_output":
            return " ".join(
                getattr(part, "text", "") or ""
                for block in (getattr(step, "content", None) or [])
                for part in (getattr(block, "parts", None) or [block])
            ).strip()
    return str(getattr(response, "output_text", "") or "")
