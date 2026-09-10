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
from typing import Literal, get_type_hints

from .config import AGENT_DIR, RUNS_DIR

InterruptBehavior = Literal["cancel", "block"]

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


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
    interrupt_behavior: InterruptBehavior = "cancel"
    wants_context: bool = False

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
    interrupt_behavior: InterruptBehavior = "cancel",
) -> Callable[[Callable], Tool]:
    """Turn a function into a Tool. The docstring becomes the model-facing description.

    `concurrency_safe` is required rather than defaulted — deciding it is the point, and a
    default would be silently wrong half the time.
    """

    def wrap(fn: Callable) -> Tool:
        return Tool(
            name=fn.__name__,
            description=inspect.getdoc(fn) or "",
            parameters=_schema_from_signature(fn),
            fn=fn,
            concurrency_safe=concurrency_safe,
            interrupt_behavior=interrupt_behavior,
            wants_context=_wants_context(fn),
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
# These close chapter 2's third memory tier: MEMORY.md has been pointing at topic files
# that nothing could actually open.


def _resolve_memory(memory_dir: Path, name: str) -> Path:
    """Keep reads inside the memory directory, whatever the model asks for."""
    candidate = (memory_dir / name).resolve()
    if not candidate.is_relative_to(memory_dir.resolve()):
        raise ValueError(f"path escapes the memory directory: {name}")
    if candidate.suffix != ".md":
        candidate = candidate.with_suffix(".md")
    return candidate


@tool(concurrency_safe=True)
def list_memory(ctx: ToolContext) -> str:
    """List the memory topic files available to read.

    Use this when you need to know what the agent already knows before answering.
    """
    if not ctx.memory_dir.is_dir():
        return "no memory directory"
    names = sorted(p.name for p in ctx.memory_dir.glob("*.md"))
    return "\n".join(names) if names else "memory directory is empty"


@tool(concurrency_safe=True)
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


@tool(concurrency_safe=False, interrupt_behavior="block")
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


def default_registry() -> ToolRegistry:
    return ToolRegistry([list_memory, read_memory, append_run_log])
