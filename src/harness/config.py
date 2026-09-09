"""Constants for the control plane.

The entrypoint caps come from Claude Code's memory governance: an index file is loaded on
every single run, so if it is allowed to grow it quietly drags context down forever.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: The strategist's control plane — prompt layers, governance, memory.
AGENT_DIR = ROOT / "agent"

MODEL = "claude-opus-5"

#: MEMORY.md is an index, not a store. Past these caps it gets truncated with a pointer.
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000

TRUNCATION_NOTICE = (
    "> [index truncated: it exceeded its line or byte cap] Entries were cut from the end "
    "of this index. Read the topic files in `agent/memory/` directly rather than assuming "
    "this list is complete."
)
