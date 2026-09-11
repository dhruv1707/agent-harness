"""What the agent was working from, carried across a compaction.

Compaction replaces the history with a written brief, and everything not in that brief is
gone. Numbers were solved by transcribing them from the tool results instead of asking a
summarizer to remember them. Everything else still passes through a writer that sees tool
results truncated to a fixed byte budget: the request it was given, any plan it committed
to, the files it had open.

So those are re-attached rather than summarised. An attachment says what the work *is*;
the brief keeps saying where it stands.

**Derived, never stored.** All three already live in the tree — the task is the opening
`user_input`, the plan is the `submit_plan` call, the files are the `read_memory` calls.
A separate `runs/<id>-attachments.json` would be a second durable copy of facts the
transcript already holds, and a second copy is a thing that can disagree with the first.
`loop.record` says the same about its own two writes.

Files are held **by name and re-read at render time**. A copy taken when the file was
first opened goes stale the moment the agent appends to it, and would cost context every
turn instead of only at a boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .config import (
    AGENT_DIR,
    MAX_ATTACHED_FILE_TOKENS,
    MAX_ATTACHMENT_TOKENS,
    approx_tokens,
)

Kind = Literal["task", "plan", "file"]

#: Tools whose call attaches a file, and the argument carrying its name. The tool records
#: nothing itself: its call step already is the record, on the single write path, and a
#: side-channel would be a second copy that can drift from the tree.
FILE_ATTACHING_TOOLS: dict[str, str] = {"read_memory": "name"}

PLAN_TOOL = "submit_plan"

#: Neither of these is context the agent can rebuild. A file it can re-read; the request
#: it was given and the plan it agreed to, it cannot.
IRREDUCIBLE: tuple[Kind, ...] = ("task", "plan")

SHED = "_[not attached, over budget: {names} — read them again if you need them]_"
TRUNCATED = "_[{name} truncated to fit the attachment budget]_"
MISSING = "_[{name} is no longer on disk]_"


@dataclass(frozen=True)
class Attachment:
    kind: Kind
    #: For a file, its name in the memory directory. Otherwise a label for the reader.
    name: str
    #: Carried for `task` and `plan`. Empty for `file`, whose body is read at render.
    body: str = ""

    def resolve(self, memory_dir: Path) -> str:
        if self.kind != "file":
            return self.body
        path = memory_dir / self.name
        if not path.is_file():
            return MISSING.format(name=self.name)
        return _head(path.read_text(encoding="utf-8"), self.name)

    def heading(self) -> str:
        return f"[attachment: {self.kind}] {self.name}".rstrip()


def _head(body: str, name: str) -> str:
    """Trim a file from the *tail*, keeping its opening.

    The opposite of `_trim_section`, which drops a brief's oldest lines because newest
    state is what continuation needs. A memory file is not a state log — it is a document
    whose headings and framing sit at the top, and one trimmed from the front is
    unreadable.
    """
    if approx_tokens(body) <= MAX_ATTACHED_FILE_TOKENS:
        return body
    lines = body.splitlines()
    while lines and approx_tokens("\n".join(lines)) > MAX_ATTACHED_FILE_TOKENS:
        lines.pop()
    return "\n".join([*lines, TRUNCATED.format(name=name)])


@dataclass
class Attachments:
    items: list[Attachment] = field(default_factory=list)

    def add(self, kind: Kind, name: str, body: str = "") -> None:
        """Later wins. A file read twice attaches once; a second plan replaces the first."""
        self.items = [i for i in self.items if not (i.kind == kind and i.name == name)]
        if kind in IRREDUCIBLE:
            self.items = [i for i in self.items if i.kind != kind]
        self.items.append(Attachment(kind=kind, name=name, body=body))

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    @property
    def files(self) -> list[Attachment]:
        return [i for i in self.items if i.kind == "file"]

    def render(
        self, memory_dir: Path | None = None, budget: int = MAX_ATTACHMENT_TOKENS
    ) -> str:
        """The block to re-attach. Empty when there is nothing, so a boundary with no
        attachments is byte-identical to one written before this existed."""
        directory = memory_dir if memory_dir is not None else (AGENT_DIR / "memory")
        kept, dropped = self._within(directory, budget)
        if not kept:
            return ""

        parts = [
            "Attached to this session and carried through the compaction below. This is "
            "the work itself, not a summary of it.",
            "",
        ]
        for attachment in kept:
            parts += [attachment.heading(), "", attachment.resolve(directory), ""]
        if dropped:
            parts.append(SHED.format(names=", ".join(d.name for d in dropped)))
        return "\n".join(parts).rstrip() + "\n"

    def _within(
        self, memory_dir: Path, budget: int
    ) -> tuple[list[Attachment], list[Attachment]]:
        spend = sum(
            approx_tokens(i.body) for i in self.items if i.kind in IRREDUCIBLE
        )
        keep = [i for i in self.items if i.kind in IRREDUCIBLE]
        if spend >= budget:
            # Task and plan alone exceed the budget. Shedding files cannot help and
            # dropping either would lose what the run is for, so the budget bends.
            return keep, self.files

        # Largest first, because the point is to fit: one 5,000-token file would
        # otherwise crowd out three small ones that together say more.
        fits: list[Attachment] = []
        for attachment in sorted(
            self.files, key=lambda a: approx_tokens(a.resolve(memory_dir))
        ):
            cost = approx_tokens(attachment.resolve(memory_dir))
            if spend + cost > budget:
                continue
            spend += cost
            fits.append(attachment)
        dropped = [f for f in self.files if f not in fits]
        ordered = [i for i in self.items if i in keep or i in fits]
        return ordered, dropped


def from_lineage(nodes: list[Any]) -> Attachments:
    """Read the attachment set out of the transcript.

    Takes nodes rather than steps because it must tell the original request from a
    compaction boundary, and only the node carries that. Use `Transcript.lineage()`: the
    live path forgets what a boundary replaced, and the full node set remembers branches
    this run abandoned.
    """
    attachments = Attachments()
    failed: set[str] = set()
    for node in nodes:
        step = node.step
        if step.get("type") == "function_result" and _is_error(step):
            failed.add(str(step.get("call_id") or step.get("id") or ""))

    for node in nodes:
        step = node.step
        kind = step.get("type")
        if kind == "user_input" and not node.meta.get("compact_boundary"):
            # The opening turn. `initial_input` puts the run context first and the user's
            # own words last, and a resume may prepend more still.
            blocks = [b.get("text", "") for b in (step.get("content") or [])]
            if blocks and blocks[-1].strip():
                attachments.add("task", "the request", blocks[-1].strip())
        elif kind == "function_call":
            call_id = str(step.get("id") or step.get("call_id") or "")
            if call_id in failed:
                continue  # a refused or failed call attaches nothing
            arguments = step.get("arguments") or {}
            name = step.get("name") or ""
            if name == PLAN_TOOL and arguments.get("plan"):
                attachments.add("plan", "approved plan", str(arguments["plan"]).strip())
            elif name in FILE_ATTACHING_TOOLS:
                value = arguments.get(FILE_ATTACHING_TOOLS[name])
                if value:
                    attachments.add("file", _as_memory_name(str(value)))
    return attachments


def _as_memory_name(value: str) -> str:
    """`read_memory` accepts a name with or without the suffix; the file has one."""
    name = Path(value).name
    return name if name.endswith(".md") else name + ".md"


def _is_error(step: dict) -> bool:
    if step.get("is_error"):
        return True
    text = "".join(b.get("text", "") for b in (step.get("result") or []))
    return text.startswith("tool error:") or text.startswith("permission denied")


def render_boundary(nodes: list[Any], brief: str, memory_dir: Path | None = None) -> str:
    """Compose what becomes the new root: what the work is, then where it stands.

    Order is not taste. The task and the plan come first because they are the only parts
    nothing can regenerate — a file can be read again, and the brief is rewritten at every
    compaction, but nobody can reconstruct what was asked for. The brief follows, and file
    bodies come last, being both the bulk and the one part that costs nothing to lose.
    """
    attached = from_lineage(nodes).render(memory_dir)
    return f"{attached}\n{brief}" if attached else brief
