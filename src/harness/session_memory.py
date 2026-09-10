"""Session memory: an operational continuation brief.

Not a second copy of the conversation. The chapter is explicit that this "distills the
session into an operational continuation brief" — status, pitfalls, what changed, and the
next actionable step. A transcript tells you what was said; this tells you where things
stand.

Writing costs a model call, so it happens at deliberate points rather than every turn.
`MemoryGate` owns that decision and nothing else does.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    MAX_SECTION_TOKENS,
    MAX_SESSION_MEMORY_TOKENS,
    RUNS_DIR,
    SESSION_MEMORY_FIRST_WRITE_TOKENS,
    SESSION_MEMORY_MIN_TOOL_CALLS,
    SESSION_MEMORY_UPDATE_INTERVAL,
    approx_tokens,
)

#: Fixed template. The model fills these in and may not invent or drop one — a brief with
#: a moving shape cannot be diffed against the last version or trusted on resume.
SECTIONS: tuple[str, ...] = (
    "Current State",
    "Task Specification",
    "Errors & Corrections",
    "Key Results",
    "Worklog",
)

TRIMMED = "_[earlier entries trimmed to stay within budget]_"


@dataclass
class SessionMemory:
    """The brief itself."""

    sections: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in SECTIONS:
            self.sections.setdefault(name, "")

    # ---- rendering and parsing ----------------------------------------------

    def render(self) -> str:
        parts = ["# Session Memory", ""]
        for name in SECTIONS:
            parts.append(f"## {name}")
            body = self.sections.get(name, "").strip()
            parts.append(body if body else "_(nothing yet)_")
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    @classmethod
    def parse(cls, text: str) -> SessionMemory:
        """Read a rendered brief back. Unknown headings are ignored, missing ones empty."""
        sections: dict[str, str] = {}
        current: str | None = None
        buffer: list[str] = []
        for line in text.splitlines():
            heading = re.match(r"^##\s+(.+?)\s*$", line)
            if heading:
                if current is not None:
                    sections[current] = "\n".join(buffer).strip()
                name = heading.group(1)
                current = name if name in SECTIONS else None
                buffer = []
                continue
            if current is not None:
                buffer.append(line)
        if current is not None:
            sections[current] = "\n".join(buffer).strip()

        for name, body in list(sections.items()):
            if body == "_(nothing yet)_":
                sections[name] = ""
        return cls(sections=sections)

    # ---- budgets -------------------------------------------------------------

    def tokens(self) -> int:
        return approx_tokens(self.render())

    def enforce_budgets(self) -> SessionMemory:
        """Condense rather than truncate: a section over its cap loses its oldest lines.

        There is deliberately no whole-brief shedding step. With five sections at
        `MAX_SECTION_TOKENS` each, the total cannot exceed `MAX_SESSION_MEMORY_TOKENS` —
        the arithmetic makes it unreachable, and `test_the_section_caps_bound_the_whole_brief`
        fails if a future section breaks that. The book's template has nine sections, where
        shedding is genuinely needed; ours does not, so shipping the branch would be
        shipping dead code.
        """
        return SessionMemory(
            sections={name: _trim_section(body) for name, body in self.sections.items()}
        )

    # ---- persistence ---------------------------------------------------------

    @staticmethod
    def path_for(session_id: str, runs_dir: Path | None = None) -> Path:
        return (runs_dir or RUNS_DIR) / f"{session_id}-memory.md"

    def save(self, session_id: str, runs_dir: Path | None = None) -> Path:
        path = self.path_for(session_id, runs_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.enforce_budgets().render(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, session_id: str, runs_dir: Path | None = None) -> SessionMemory | None:
        path = cls.path_for(session_id, runs_dir)
        if not path.is_file():
            return None
        return cls.parse(path.read_text(encoding="utf-8"))


def _trim_section(body: str) -> str:
    """Drop oldest lines until the section fits. Newest state is what continuation needs."""
    if approx_tokens(body) <= MAX_SECTION_TOKENS:
        return body
    lines = body.splitlines()
    while lines and approx_tokens("\n".join(lines)) > MAX_SECTION_TOKENS:
        lines.pop(0)
    return "\n".join([TRIMMED, *lines]).strip()


# ---- when to write -----------------------------------------------------------

#: What the gate decided: create a brief, update the existing one, or do nothing.
Decision = str  # "create" | "update" | None


@dataclass
class MemoryGate:
    """Decides when the brief is worth a model call.

    Below the first-write threshold there is nothing worth compressing. Above it, updates
    wait for a moment where the state is actually coherent: real tool activity has
    accumulated, or the run has paused. Crossing the interval mid-tool-chain defers rather
    than capturing a half-formed picture.
    """

    exists: bool = False
    last_write_tokens: int = 0
    tool_calls_since: int = 0
    errors_since: int = 0

    def observe_tool_calls(self, count: int, errors: int = 0) -> None:
        self.tool_calls_since += count
        self.errors_since += errors

    def decide(self, context_tokens: int, at_stopping_point: bool) -> Decision | None:
        if context_tokens < SESSION_MEMORY_FIRST_WRITE_TOKENS:
            return None
        if not self.exists:
            return "create"
        if context_tokens - self.last_write_tokens < SESSION_MEMORY_UPDATE_INTERVAL:
            return None
        if self.tool_calls_since >= SESSION_MEMORY_MIN_TOOL_CALLS or self.errors_since:
            return "update"
        if at_stopping_point:
            return "update"
        return None  # defer — re-checked next turn

    def record_write(self, context_tokens: int) -> None:
        self.exists = True
        self.last_write_tokens = context_tokens
        self.tool_calls_since = 0
        self.errors_since = 0


# ---- the writer --------------------------------------------------------------
#
# A plain callable so chapter 7 can replace the inline model call with a forked sub-agent
# without touching the loop.

Writer = Callable[[list[dict], "SessionMemory | None"], "SessionMemory"]

WRITE_PROMPT = """\
You are maintaining an operational continuation brief for an agent session.

Fill in every section of the template below and no others. This is not a chat log: it is
what someone would need to pick the work up cold. Record status, pitfalls, what changed,
and the next actionable step.

Rules:
- Do not talk about note-taking itself.
- Do not alter the template structure or invent sections.
- Keep Current State aligned with the latest work.
- Keep every section dense. Prefer specifics — ids, metrics, file names — over narration.
- Worklog is a compressed list of what was done, not a transcript.

Template:

## Current State
## Task Specification
## Errors & Corrections
## Key Results
## Worklog
"""


def render_previous(previous: SessionMemory | None) -> str:
    if previous is None:
        return "There is no previous brief. Write the first one."
    return "The previous brief follows. Update it; do not start over.\n\n" + previous.render()
