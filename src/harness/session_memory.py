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
    "Task",
    "Findings",
    "Measurements",
    "Examined",
    "Failed approaches",
    "Current state",
    "Next",
)

#: Budget per section, allocated by value rather than evenly. Findings is what makes the
#: brief worth having — it is the knowledge the compacted history no longer holds — so it
#: gets the most room. Next needs one line. Six equal sections would also sit exactly on
#: MAX_SESSION_MEMORY_TOKENS, leaving no room for the headings themselves.
SECTION_BUDGETS: dict[str, int] = {
    "Task": 1_000,
    "Findings": 3_000,
    #: Written by the harness from tool results rather than by the model, so this is a
    #: transcription budget, not a summarization one — it bounds how many entities carry
    #: forward. See `verify.render_measurements`.
    "Measurements": 2_000,
    "Examined": 1_500,
    "Failed approaches": 1_500,
    "Current state": 1_500,
    "Next": 500,
}

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

        There is deliberately no whole-brief shedding step. `SECTION_BUDGETS` sums to 9,000
        against a 12,000 total, so the arithmetic makes it unreachable, and
        `test_the_section_budgets_bound_the_whole_brief` fails if that stops being true.
        The book's nine-section template is where shedding earns its place; ours does not,
        so shipping the branch would be shipping dead code.
        """
        return SessionMemory(
            sections={
                name: _trim_section(body, SECTION_BUDGETS.get(name, MAX_SECTION_TOKENS))
                for name, body in self.sections.items()
            }
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


def _trim_section(body: str, budget: int = MAX_SECTION_TOKENS) -> str:
    """Drop oldest lines until the section fits. Newest state is what continuation needs."""
    if approx_tokens(body) <= budget:
        return body
    lines = body.splitlines()
    while lines and approx_tokens("\n".join(lines)) > budget:
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

Someone picking this up cold must be able to continue the work from this alone. Once the
session is compacted this brief *replaces* the history it summarises, so anything not
written here is gone.

That means recording what was **learned**, not what was done.

Rules:
- Findings are facts, not activity. "Top ads all open on a problem callout, strongest is
  ad 52553174233835 at 6.77 roas" is a finding. "Read brand-voice.md (3,684 bytes)" is
  not — it tells the reader nothing usable and makes them open the file again.
- Examined is a bare list of what has already been looked at, so it is not fetched twice.
- Failed approaches covers anything tried that did not work, including things that ran
  without error. It is what stops the next turn repeating them.
- Next is one concrete action. Not a plan, not a list of options.
- Leave `Measurements` empty. The harness fills it in from the tool results directly,
  because the history you are reading has large tool results truncated — you do not have
  the figures in front of you and must not reconstruct them. Cite ad ids freely in
  Findings; the numbers will be attached for you.
- Do not talk about note-taking itself.
- Do not add, rename, reorder or drop fields.
- Be dense. Prefer concrete values — ad ids, metric numbers, verbatim hooks — over
  description of activity.

Template:

## Task
## Findings
## Measurements
## Examined
## Failed approaches
## Current state
## Next
"""


def render_previous(previous: SessionMemory | None) -> str:
    if previous is None:
        return "There is no previous brief. Write the first one."
    return "The previous brief follows. Update it; do not start over.\n\n" + previous.render()
