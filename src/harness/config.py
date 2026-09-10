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

MODEL = os.environ.get("HARNESS_MODEL", "gemini-3.8-flash")

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

#: Hard rail on the query loop. A loop with no ceiling is a billing incident.
MAX_TURNS = int(os.environ.get("HARNESS_MAX_TURNS", "20"))

#: How many concurrency-safe tools may run at once. Unsafe tools always run alone.
MAX_PARALLEL_TOOLS = int(os.environ.get("HARNESS_MAX_PARALLEL_TOOLS", "8"))

#: Per-tool wall clock. A hung tool must not hang the ledger.
TOOL_TIMEOUT_SECONDS = float(os.environ.get("HARNESS_TOOL_TIMEOUT", "120"))

#: Session transcripts, one JSONL file per session.
RUNS_DIR = ROOT / "runs"


# ---- context governance ------------------------------------------------------
#
# The model's window is 1,048,576 tokens and a full daily run peaks around 82k — 8%.
# Computing the threshold from the hardware limit would put it near 1,015,000 and it
# would never fire. So the budget is a *policy* number: every input token is billed on
# every turn and long context degrades attention. Capacity does not justify consumption.

CONTEXT_BUDGET_TOKENS = int(os.environ.get("HARNESS_CONTEXT_BUDGET", "120000"))

#: Reserved so the summarization call itself has room to answer.
MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000

#: Early-warning headroom below the budget — warnings, errors, a manual compact.
AUTOCOMPACT_BUFFER_TOKENS = 13_000

#: Circuit breaker. "You may fail, but you may not fail infinitely without memory."
MAX_CONSECUTIVE_COMPACT_FAILURES = 3

#: Share of the most recent steps kept verbatim through a compaction.
KEEP_RECENT_SHARE = 0.20


def compact_threshold(budget: int | None = None) -> int:
    """Context size at which compaction should run, in tokens.

    Normally budget minus the two reserves. A small budget — the ones used to force
    compaction in testing — would otherwise go negative, since the reserves total 33,000,
    and a negative threshold fires on turn one before there is anything to compact. Below
    that point the threshold falls back to half the budget, which keeps the reserves
    proportional instead of nonsensical.
    """
    budget = CONTEXT_BUDGET_TOKENS if budget is None else budget
    return max(budget - MAX_OUTPUT_TOKENS_FOR_SUMMARY - AUTOCOMPACT_BUFFER_TOKENS, budget // 2)


# ---- session memory ----------------------------------------------------------
#
# Note the two 12,000s below are unrelated. The first is *when* the brief is first
# written; the second is *how large* it may grow. Same number, different meaning.

#: No brief at all below this — there is nothing worth compressing yet.
SESSION_MEMORY_FIRST_WRITE_TOKENS = 12_000

#: Context growth since the last write before an update is considered.
SESSION_MEMORY_UPDATE_INTERVAL = 5_000

#: Tool calls since the last write that count as "real work has happened".
SESSION_MEMORY_MIN_TOOL_CALLS = 3

#: Per-section and whole-brief size caps.
MAX_SECTION_TOKENS = 2_000
MAX_SESSION_MEMORY_TOKENS = 12_000


def approx_tokens(text: str) -> int:
    """Rough token count at 4 chars/token.

    Deliberately an estimate: budgets are enforced per section on every write, and a
    `count_tokens` round trip per section would cost more than the precision is worth.
    """
    return len(text) // 4

TRUNCATION_NOTICE = (
    "> [index truncated: it exceeded its line or byte cap] Entries were cut from the end "
    "of this index. Read the topic files in `agent/memory/` directly rather than assuming "
    "this list is complete."
)
