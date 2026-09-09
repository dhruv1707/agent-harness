"""Constants for the control plane.

The entrypoint caps come from Claude Code's memory governance: an index file is loaded on
every single run, so if it is allowed to grow it quietly drags context down forever.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so the harness finds its own credentials.

    Real environment variables always win — this only fills in what is unset.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export ") :]
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv(ROOT / ".env")

#: The strategist's control plane — prompt layers, governance, memory.
AGENT_DIR = ROOT / "agent"

MODEL = os.environ.get("HARNESS_MODEL", "gemini-2.5-pro")

#: Minimum prompt-prefix size before a model will cache it at all. Below the floor the
#: stable prefix is re-billed in full on every run and the cache breakpoint buys nothing.
#: Source: ai.google.dev/gemini-api/docs/caching (implicit caching minimums).
CACHE_FLOOR_TOKENS: dict[str, int] = {
    "gemini-2.5-flash": 2_048,
    "gemini-2.5-pro": 2_048,
    "gemini-3.5-flash": 4_096,
    "gemini-3.6-flash": 4_096,
    "gemini-3.7-flash": 4_096,
    "gemini-3.8-flash": 4_096,
    "gemini-3.1-pro-preview": 4_096,
}


def cache_floor(model: str = MODEL) -> int | None:
    """Tokens the stable prefix must exceed before `model` caches it. None if unknown."""
    if model in CACHE_FLOOR_TOKENS:
        return CACHE_FLOOR_TOKENS[model]
    # Unversioned or dated variants: fall back to the family's floor.
    for known, floor in CACHE_FLOOR_TOKENS.items():
        if model.startswith(known):
            return floor
    if model.startswith("gemini-3"):
        return 4_096
    if model.startswith("gemini-2.5"):
        return 2_048
    return None


#: MEMORY.md is an index, not a store. Past these caps it gets truncated with a pointer.
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000

TRUNCATION_NOTICE = (
    "> [index truncated: it exceeded its line or byte cap] Entries were cut from the end "
    "of this index. Read the topic files in `agent/memory/` directly rather than assuming "
    "this list is complete."
)
