"""Tools the agent can call, and the registry that holds them.

Two pieces of metadata beyond the callable itself:

`concurrency_safe` — whether this tool may overlap others. The executor treats an unsafe
tool as a barrier. A tool that touches shared state, writes a file, or must observe an
earlier call's effects is not safe.

`interrupt_behavior` — how the tool ends when a run is interrupted. `cancel` stops it
immediately; `block` lets it finish first. Killing a half-written file is worse than
waiting for it.

Tools that need ambient state (which session, which directories) annotate their first
parameter as `ToolContext`. The executor supplies it and the schema builder omits it, so
the model never sees it and cannot forge one.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_type_hints

from .config import (
    AGENT_DIR,
    AGENT_TIMEOUT_SECONDS,
    DEFAULT_MAX_RESULT_SIZE_CHARS,
    RUNS_DIR,
)
from .permissions import PlanState

#: Memory files the agent may append to. Everything else under agent/memory/ is curated
#: by a human and read-only: brief-samples.md is ground truth from scripts that actually
#: shipped, and brand-voice.md carries compliance boundaries. An agent that can rewrite
#: its own evidence has no evidence.
WRITABLE_MEMORY: frozenset[str] = frozenset({"hook-patterns.md"})

#: A topic file is read in full whenever it is opened, so unbounded growth is a context
#: leak. Past this the tool refuses and says to consolidate.
MAX_MEMORY_FILE_BYTES = 20_000

InterruptBehavior = Literal["cancel", "block"]

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


class ToolError(Exception):
    """A tool failed in a way it understands, and says so in its own words.

    Raising this instead of returning an error string is what gets the outcome flagged
    `is_error`. The executor renders it verbatim rather than as `ToolError: ...`, because
    the message is already written for the model to read.

    It exists because an MCP server's own error arrived as an ordinary return value, so a
    remote 401 closed the ledger as a success with no error flag — and the memory gate
    counted it as one.
    """


@dataclass(frozen=True)
class ToolContext:
    """Ambient state a tool runs against.

    Supplied by the runtime, never by the model. Only fields a tool actually reads live
    here; permissions and MCP handles arrive when something supplies them.
    """

    session_id: str
    agent_dir: Path = AGENT_DIR
    runs_dir: Path = RUNS_DIR
    turn: int = 0
    #: Shared with the permission gate, which enforces plan mode while this tool ends it.
    #: None means plan mode is unavailable in this run.
    plan: PlanState | None = None
    #: The child-agent pool, when this run may delegate. None means it may not, and
    #: `spawn_agent` says so rather than failing obscurely.
    pool: Any = None

    @property
    def memory_dir(self) -> Path:
        return self.agent_dir / "memory"


def _resolved_hints(fn: Callable) -> dict:
    """Real types, not the strings `from __future__ import annotations` leaves behind."""
    try:
        return get_type_hints(fn)
    except Exception:  # a forward reference we cannot resolve; fall back to raw
        return getattr(fn, "__annotations__", {}) or {}


def _wants_context(fn: Callable) -> bool:
    parameters = list(inspect.signature(fn).parameters.values())
    if not parameters:
        return False
    first = parameters[0]
    hints = _resolved_hints(fn)
    return hints.get(first.name) is ToolContext or first.annotation in (
        ToolContext,
        "ToolContext",
    )


def _schema_from_signature(fn: Callable) -> dict:
    """Build a JSON Schema from the signature, omitting the context parameter."""
    hints = _resolved_hints(fn)
    skip_first = _wants_context(fn)

    properties: dict[str, dict] = {}
    required: list[str] = []
    for index, (name, param) in enumerate(inspect.signature(fn).parameters.items()):
        if index == 0 and skip_first:
            continue  # runtime-supplied; the model must never see or set it
        json_type = _JSON_TYPES.get(hints.get(name, str), "string")
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class Tool:
    """A callable the model may invoke."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]
    concurrency_safe: bool
    #: Does calling this change anything? Distinct from `concurrency_safe`, which asks
    #: whether it is safe to run alongside others — one is a scheduling question and the
    #: other is a safety boundary, and plan mode is built on this one.
    #:
    #: Defaulted, unlike `concurrency_safe`, because the default is the *safe* direction:
    #: False means "treat it as a write", so a tool nobody thought about is refused in
    #: plan mode rather than quietly permitted.
    read_only: bool = False
    #: Seconds this tool may run, overriding the executor's default. For the rare tool
    #: whose work is not tool-shaped — spawning a child agent is a whole run, and the
    #: 120s default would kill one and call it hung.
    timeout: float | None = None
    interrupt_behavior: InterruptBehavior = "cancel"
    wants_context: bool = False
    #: Chars of result this tool may put in the context before the rest is written to a
    #: file and replaced by a preview. `None` opts out of persistence entirely, which is
    #: right for a tool that reads persisted output: persisting its result would hand the
    #: model a path to the thing it just asked to be read.
    #:
    #: The effective ceiling is the lower of this and the global default, so declaring a
    #: number can only tighten the gate, never loosen it.
    max_result_chars: int | None = DEFAULT_MAX_RESULT_SIZE_CHARS

    def declaration(self) -> dict:
        """The wire shape the Interactions API expects in `tools`."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def invoke(self, ctx: ToolContext, /, **kwargs):
        """Call the underlying function, supplying context only if it asked for one."""
        if self.wants_context:
            return self.fn(ctx, **kwargs)
        return self.fn(**kwargs)


def tool(
    *,
    concurrency_safe: bool,
    read_only: bool,
    timeout: float | None = None,
    interrupt_behavior: InterruptBehavior = "cancel",
    max_result_chars: int | None = DEFAULT_MAX_RESULT_SIZE_CHARS,
) -> Callable[[Callable], Tool]:
    """Turn a function into a Tool. The docstring becomes the model-facing description.

    `concurrency_safe` and `read_only` are both required rather than defaulted — deciding
    them is the point, and a default would be silently wrong half the time. `Tool` itself
    defaults `read_only` to False so that a tool built by other means fails closed, but an
    author writing one here is made to answer.
    """

    def wrap(fn: Callable) -> Tool:
        return Tool(
            name=fn.__name__,
            description=inspect.getdoc(fn) or "",
            parameters=_schema_from_signature(fn),
            fn=fn,
            concurrency_safe=concurrency_safe,
            read_only=read_only,
            timeout=timeout,
            interrupt_behavior=interrupt_behavior,
            wants_context=_wants_context(fn),
            max_result_chars=max_result_chars,
        )

    return wrap


class ToolRegistry:
    """The set of tools available for a run."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for entry in tools or []:
            self.register(entry)

    def register(self, entry: Tool) -> None:
        if entry.name in self._tools:
            raise ValueError(f"duplicate tool name: {entry.name}")
        self._tools[entry.name] = entry

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def declarations(self) -> list[dict]:
        return [entry.declaration() for entry in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


# ---- the first real tools ---------------------------------------------------
#
# These close the third memory tier: MEMORY.md has been pointing at topic files that
# nothing could actually open.


def _resolve_memory(memory_dir: Path, name: str) -> Path:
    """Keep reads inside the memory directory, whatever the model asks for."""
    candidate = (memory_dir / name).resolve()
    if not candidate.is_relative_to(memory_dir.resolve()):
        raise ValueError(f"path escapes the memory directory: {name}")
    if candidate.suffix != ".md":
        candidate = candidate.with_suffix(".md")
    return candidate


@tool(concurrency_safe=True, read_only=True)
def list_memory(ctx: ToolContext) -> str:
    """List the memory topic files available to read.

    Use this when you need to know what the agent already knows before answering.
    """
    if not ctx.memory_dir.is_dir():
        return "no memory directory"
    names = sorted(p.name for p in ctx.memory_dir.glob("*.md"))
    return "\n".join(names) if names else "memory directory is empty"


@tool(concurrency_safe=True, read_only=True)
def read_memory(ctx: ToolContext, name: str) -> str:
    """Read one memory topic file in full.

    Args:
        name: The file name from list_memory, e.g. "brief-samples.md". The .md suffix is
            optional.
    """
    path = _resolve_memory(ctx.memory_dir, name)
    if not path.is_file():
        return f"no such memory file: {name}"
    return path.read_text(encoding="utf-8")


#: Set by the runtime when a run may delegate. A tool cannot reach the pool any other way,
#: which is deliberate: spawning is the one capability the model must not be able to
#: manufacture for itself.
# Async, but it awaits nothing. The point is only to stay on the event loop: a sync tool
# runs in a worker thread via `to_thread`, and `create_task` there raises "no running event
# loop" — which put child ids in the ledger with no task behind them, so nothing waited for
# them and teardown reported them cancelled.
@tool(concurrency_safe=True, read_only=False)
async def spawn_agent(ctx: ToolContext, role: str, task: str) -> str:
    """Hand a piece of work to a child agent. Returns straight away; it works in the
    background while you carry on.

    The child starts fresh: it sees the same rules and tools you do but none of your
    conversation, so the task has to stand alone. Give it one job and the specifics it
    needs — the account, the window, the ads or files.

    Spawn everything that does not depend on something else in the same turn, then get on
    with whatever you can genuinely do meanwhile.

    **Their findings are delivered to you the moment you stop calling tools.** Do not call
    other tools to pass the time and do not announce that you are waiting — that only
    delays delivery. When you have nothing else useful to do, simply end your turn and the
    answers will be in front of you. They arrive in full rather than summarised, because
    deciding what matters across them is your job and you cannot do it on material already
    squeezed. You cannot finish the run without them.

    Args:
        role: `researcher` gathers and may only read; `implementer` produces the
            deliverable; `verifier` checks one against the evidence.
        task: What this worker should do, written so someone with no other context could
            act on it.
    """
    if ctx.pool is None:
        return "delegation is not available in this run; do the work yourself"
    try:
        child_id = ctx.pool.spawn(role, task)
    except (ValueError, FileNotFoundError) as exc:
        return f"cannot spawn: {exc}"
    return (
        f"started {role} `{child_id}`. It is running now — spawn anything else independent "
        "in this turn, and its findings will reach you when every child has finished."
    )


# `concurrency_safe=False` is load-bearing rather than cautious: the executor treats an
# unsafe tool as a barrier, so everything in flight drains, this runs alone, and calls
# queued behind it resume *after* the mode has flipped. Without the barrier a sibling call
# could race its own permission check against the approval that would have allowed it.
@tool(concurrency_safe=False, read_only=False, interrupt_behavior="block")
def submit_plan(ctx: ToolContext, plan: str) -> str:
    """Propose your plan for approval. Call this in plan mode once the plan is ready.

    A person reads it and decides. Approved, plan mode lifts and you continue in the same
    session with everything you have already learned — so do not re-research. Refused, you
    are told why and may revise once.

    Args:
        plan: The plan itself. What you will do and why, not an account of your research.
    """
    if ctx.plan is None:
        return "plan mode is not available in this run; there is nothing to approve"
    if not ctx.plan.active:
        return "not in plan mode — nothing to approve here; just do the work"

    # Reaching the body at all means the gate approved it: a refusal never calls us, and
    # the plan text was recorded on the call step before the gate ever ran.
    ctx.plan.active = False
    return (
        "Plan approved. Plan mode has lifted and the tools you were refused are available "
        "again. Carry the plan out; do not restate it first."
    )


@tool(concurrency_safe=True, read_only=True, max_result_chars=None)
def read_tool_result(ctx: ToolContext, call_id: str, offset: int = 0, limit: int = 20000) -> str:
    """Read back a tool result that was too large to keep in the conversation.

    When a result is replaced by a `<persisted-output>` block, the full text is on disk.
    This reads a slice of it. Ask for the part you need rather than the whole thing — the
    reason it was moved out is that all of it does not fit.

    Args:
        call_id: The id in the persisted-output block.
        offset: Character offset to start from.
        limit: How many characters to return.
    """
    from .budget import results_dir

    path = results_dir(ctx.runs_dir, ctx.session_id) / f"{call_id}.txt"
    if not path.exists():
        return f"no persisted result for call_id {call_id!r}"

    text = path.read_text(encoding="utf-8")
    limit = max(1, min(limit, DEFAULT_MAX_RESULT_SIZE_CHARS))
    chunk = text[offset : offset + limit]
    end = offset + len(chunk)
    tail = (
        f"\n[… {len(text) - end:,} more chars; "
        f"read_tool_result(call_id={call_id!r}, offset={end})]"
        if end < len(text)
        else ""
    )
    return f"[{call_id} chars {offset}-{end} of {len(text):,}]\n{chunk}{tail}"


@tool(concurrency_safe=False, read_only=False, interrupt_behavior="block")
def append_run_log(ctx: ToolContext, text: str) -> str:
    """Append a line to this run's log. Use it to record a decision worth keeping.

    Args:
        text: The line to append.
    """
    ctx.runs_dir.mkdir(parents=True, exist_ok=True)
    log = ctx.runs_dir / f"{ctx.session_id}-log.md"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
    return f"appended {len(text)} chars to {log.name}"


@tool(concurrency_safe=False, read_only=False, interrupt_behavior="block")
def append_memory(ctx: ToolContext, name: str, section: str, entry: str) -> str:
    """Append one entry to a section of a writable memory file, so a finding survives.

    Use this when a run learns something the next run should not have to rediscover — a
    hook pattern the taxonomy did not have, or the result of an iteration that was tested.
    Without it every session re-derives the same conclusions and the taxonomy never settles.

    Append-only, and only to files that are meant to accumulate. It cannot create a
    section, reword an existing line, or touch the curated files.

    Args:
        name: The memory file, e.g. "hook-patterns.md".
        section: An existing "## " heading in that file, e.g. "Tested".
        entry: One line to append. A table row if the section holds a table.
    """
    if name not in WRITABLE_MEMORY:
        return (
            f"{name} is read-only. Writable memory files: {', '.join(sorted(WRITABLE_MEMORY))}. "
            "Report the finding in your output instead."
        )

    path = _resolve_memory(ctx.memory_dir, name)
    if not path.is_file():
        return f"no such memory file: {name}"

    text = path.read_text(encoding="utf-8")
    if len(text.encode("utf-8")) > MAX_MEMORY_FILE_BYTES:
        return (
            f"{name} is at its {MAX_MEMORY_FILE_BYTES:,} byte cap and is read in full every "
            "time it is opened. Consolidate it before adding more."
        )

    lines = text.splitlines()
    heading = f"## {section.lstrip('# ').strip()}"
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        existing = [ln[3:] for ln in lines if ln.startswith("## ")]
        return f"no section '{section}' in {name}. Sections: {', '.join(existing)}"

    # End of this section is the next heading, or the end of the file.
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines)
    )
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1  # keep the blank line that separates sections

    lines.insert(end, entry.rstrip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"appended to '{section}' in {name}"


def default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            list_memory,
            read_memory,
            read_tool_result,
            append_memory,
            append_run_log,
            submit_plan,
            spawn_agent,
        ]
    )
