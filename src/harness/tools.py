"""Tools the agent can call, and the registry that holds them.

Every tool declares whether it is safe to run concurrently. The executor uses that flag to
decide what may overlap: safe tools run in parallel, unsafe tools act as barriers. A tool
that touches shared state, writes a file, or must observe the effects of an earlier call in
the same turn is **not** safe.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import AGENT_DIR, RUNS_DIR

MEMORY_DIR = AGENT_DIR / "memory"

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


@dataclass(frozen=True)
class Tool:
    """A callable the model may invoke."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]
    concurrency_safe: bool

    def declaration(self) -> dict:
        """The wire shape the Interactions API expects in `tools`."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __call__(self, **kwargs) -> str:
        return self.fn(**kwargs)


def _schema_from_signature(fn: Callable) -> dict:
    """Build a JSON Schema from the function signature. Deliberately minimal."""
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        annotation = param.annotation
        json_type = _JSON_TYPES.get(annotation, "string")
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def tool(*, concurrency_safe: bool) -> Callable[[Callable], Tool]:
    """Turn a function into a Tool. The docstring becomes the model-facing description.

    `concurrency_safe` is required rather than defaulted — deciding it is the point, and a
    default would be silently wrong half the time.
    """

    def wrap(fn: Callable) -> Tool:
        description = inspect.getdoc(fn) or ""
        return Tool(
            name=fn.__name__,
            description=description,
            parameters=_schema_from_signature(fn),
            fn=fn,
            concurrency_safe=concurrency_safe,
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


def _resolve_memory(name: str) -> Path:
    """Keep reads inside agent/memory/, whatever the model asks for."""
    candidate = (MEMORY_DIR / name).resolve()
    if not candidate.is_relative_to(MEMORY_DIR.resolve()):
        raise ValueError(f"path escapes the memory directory: {name}")
    if candidate.suffix != ".md":
        candidate = candidate.with_suffix(".md")
    return candidate


@tool(concurrency_safe=True)
def list_memory() -> str:
    """List the memory topic files available to read.

    Use this when you need to know what the agent already knows before answering.
    """
    if not MEMORY_DIR.is_dir():
        return "no memory directory"
    names = sorted(p.name for p in MEMORY_DIR.glob("*.md"))
    return "\n".join(names) if names else "memory directory is empty"


@tool(concurrency_safe=True)
def read_memory(name: str) -> str:
    """Read one memory topic file in full.

    Args:
        name: The file name from list_memory, e.g. "brief-samples.md". The .md suffix is
            optional.
    """
    path = _resolve_memory(name)
    if not path.is_file():
        return f"no such memory file: {name}"
    return path.read_text(encoding="utf-8")


@tool(concurrency_safe=False)
def append_run_log(text: str) -> str:
    """Append a line to this run's log. Use it to record a decision worth keeping.

    Args:
        text: The line to append.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log = RUNS_DIR / "run-log.md"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
    return f"appended {len(text)} chars to run-log.md"


def default_registry() -> ToolRegistry:
    return ToolRegistry([list_memory, read_memory, append_run_log])
