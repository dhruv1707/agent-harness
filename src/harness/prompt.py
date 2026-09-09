"""Prompt assembly: the control plane, not personality.

The system prompt is built from ordered layers with an explicit precedence structure. Two
invariants hold, and `tests/test_prompt.py` enforces both:

1. A job description extends the constitution; it cannot wipe it. `custom` and `append` are
   strictly additive. Only an explicit `override` displaces the default layer stack, and
   even then governance, the memory index and `append` survive.
2. The cacheable prefix is byte-identical across runs. Everything that varies per run lives
   after the cache breakpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path

from .config import (
    AGENT_DIR,
    MAX_ENTRYPOINT_BYTES,
    MAX_ENTRYPOINT_LINES,
    TRUNCATION_NOTICE,
)

#: The default stack, in precedence order. Displaced only by an explicit override.
DEFAULT_LAYER_FILES: list[tuple[str, str]] = [
    ("identity", "prompts/00-identity.md"),
    ("system-rules", "prompts/10-system-rules.md"),
    ("brand-voice", "prompts/20-brand-voice.md"),
    ("workflow", "prompts/30-workflow.md"),
]

GOVERNANCE_FILE = "CLAUDE.md"
ENTRYPOINT_FILE = "memory/MEMORY.md"


@dataclass(frozen=True)
class Layer:
    """One section of the system prompt."""

    name: str
    text: str
    cacheable: bool
    source: str


@dataclass(frozen=True)
class RunContext:
    """Per-run facts. Volatile by definition — these live after the cache breakpoint."""

    run_id: str = "local"
    today: str = field(default_factory=lambda: _date.today().isoformat())
    window: str = "last 7 days"
    metric: str | None = None
    sources: tuple[str, ...] = ()

    def render(self) -> str:
        lines = [
            "# Run Context",
            "",
            "Facts about this run only. Nothing here is a standing rule.",
            "",
            f"- Today: {self.today}",
            f"- Run id: {self.run_id}",
            f"- Window: {self.window}",
        ]
        if self.metric:
            lines.append(f"- Primary efficiency metric: {self.metric}")
        if self.sources:
            lines.append(f"- Connected sources: {', '.join(self.sources)}")
        else:
            lines.append(
                "- Connected sources: none. You have no data tools this run — say so "
                "rather than working from memory."
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class AssembledPrompt:
    """The assembled control plane, inspectable and wire-ready."""

    layers: list[Layer]

    @property
    def stable_layers(self) -> list[Layer]:
        return [layer for layer in self.layers if layer.cacheable]

    @property
    def volatile_layers(self) -> list[Layer]:
        return [layer for layer in self.layers if not layer.cacheable]

    @property
    def stable_text(self) -> str:
        return "\n\n".join(layer.text for layer in self.stable_layers)

    @property
    def volatile_text(self) -> str:
        return "\n\n".join(layer.text for layer in self.volatile_layers)

    def blocks(self) -> list[dict]:
        """The `system` argument for messages.create — cache breakpoint included."""
        wire: list[dict] = [
            {
                "type": "text",
                "text": self.stable_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if self.volatile_text:
            wire.append({"type": "text", "text": self.volatile_text})
        return wire


def truncate_entrypoint_content(content: str) -> str:
    """Cap the memory index at whichever limit trips first, leaving a pointer behind."""
    truncated = False

    lines = content.splitlines()
    if len(lines) > MAX_ENTRYPOINT_LINES:
        lines = lines[:MAX_ENTRYPOINT_LINES]
        truncated = True
    out = "\n".join(lines)

    encoded = out.encode("utf-8")
    if len(encoded) > MAX_ENTRYPOINT_BYTES:
        out = encoded[:MAX_ENTRYPOINT_BYTES].decode("utf-8", errors="ignore")
        truncated = True

    if truncated:
        out = f"{out.rstrip()}\n\n{TRUNCATION_NOTICE}"
    return out


def _read(agent_dir: Path, relative: str) -> str:
    path = agent_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"control plane file missing: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_default_layers(agent_dir: Path = AGENT_DIR) -> list[Layer]:
    return [
        Layer(name=name, text=_read(agent_dir, rel), cacheable=True, source=rel)
        for name, rel in DEFAULT_LAYER_FILES
    ]


def build_effective_system_prompt(
    *,
    agent_dir: Path = AGENT_DIR,
    run_context: RunContext | None = None,
    override: str | None = None,
    custom: str | None = None,
    append: str | None = None,
) -> AssembledPrompt:
    """Assemble the system prompt in precedence order.

    Args:
        agent_dir: Root of the control plane.
        run_context: Per-run facts. Rendered after the cache breakpoint.
        override: Replaces the default layer stack. Governance, the memory index and
            `append` still apply — an override narrows the constitution, it does not
            abolish it.
        custom: A job description appended after the default stack. Never replaces it.
        append: Always last, always after the cache breakpoint.
    """
    run_context = run_context or RunContext()
    layers: list[Layer] = []

    # 1-4. The default stack, or an explicit override of it.
    if override is not None:
        layers.append(Layer("override", override.strip(), True, "<override>"))
    else:
        layers.extend(load_default_layers(agent_dir))

    # 5. A job description extends; it does not displace.
    if custom:
        layers.append(Layer("custom", custom.strip(), True, "<custom>"))

    # 6-7. Governance and the memory index are never displaced.
    layers.append(
        Layer("governance", _read(agent_dir, GOVERNANCE_FILE), True, GOVERNANCE_FILE)
    )
    layers.append(
        Layer(
            "memory-index",
            truncate_entrypoint_content(_read(agent_dir, ENTRYPOINT_FILE)),
            True,
            ENTRYPOINT_FILE,
        )
    )

    # ---- cache breakpoint ----

    # 8. Per-run facts.
    layers.append(Layer("run-context", run_context.render(), False, "<runtime>"))

    # 9. Always last.
    if append:
        layers.append(Layer("append", append.strip(), False, "<append>"))

    return AssembledPrompt(layers=layers)
