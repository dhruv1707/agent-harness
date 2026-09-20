"""Constants for the control plane.

The entrypoint caps come from Claude Code's memory governance: an index file is loaded on
every single run, so if it is allowed to grow it quietly drags context down forever.
"""

import json
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

#: How long to wait for a person to answer a permission prompt. Generous, because someone
#: is reading a plan or a diff — but not unbounded: the gate serializes prompts behind a
#: lock, so one unanswered question parks every sibling waiting to ask, and `drain()` with
#: it. An unattended terminal denies immediately and never reaches this.
APPROVAL_TIMEOUT_SECONDS = float(os.environ.get("HARNESS_APPROVAL_TIMEOUT", "600"))

#: How long an `interrupt_behavior="block"` tool may take to finish once an interrupt has
#: arrived. Letting a half-written file complete is the point; letting a stuck writer make
#: the interrupt unkillable is not.
INTERRUPT_DRAIN_SECONDS = float(os.environ.get("HARNESS_INTERRUPT_DRAIN", "30"))

# ---- child agents -------------------------------------------------------------

#: How long a spawned child may run. Far longer than a tool call, because a child *is* a
#: run: a real research pass against the ad account takes over two minutes. The per-tool
#: timeout would kill one at 120s and report it as a hung tool.
AGENT_TIMEOUT_SECONDS = float(os.environ.get("HARNESS_AGENT_TIMEOUT", "600"))

#: How many children one run may spawn in total. A coordinator that misjudges a one-line
#: question should cost a bounded amount, and a cap is a control a human will actually get
#: — unlike a permission prompt per child, which nobody reads by the fourth one.
MAX_CHILDREN_PER_RUN = int(os.environ.get("HARNESS_MAX_CHILDREN", "6"))

#: Children may not spawn children. One level of delegation is the whole design; a second
#: makes cost and cancellation unbounded for no benefit anyone has asked for.
MAX_AGENT_DEPTH = 1

#: How long a lifecycle hook may take. Short, because a hook is a notifier or a check, not
#: a job — and because the run waits for it. Past this it is abandoned and the run carries
#: on, since a broken hook must not cost a research run.
HOOK_TIMEOUT_SECONDS = float(os.environ.get("HARNESS_HOOK_TIMEOUT", "30"))

#: How many times a hook may bounce a child back for another attempt. One, matching the
#: verifier's revision cap and plan mode's: a gate that can bounce forever is a gate that
#: will, and the second refusal is rarely more informative than the first.
MAX_HOOK_BOUNCES = int(os.environ.get("HARNESS_MAX_HOOK_BOUNCES", "1"))


#: Cap on a tool's *error* text. A tool that raises with a megabyte message would otherwise
#: write a megabyte into the context. Successful results are bounded separately, by the
#: size gate below — this one stays because an error is never worth persisting.
MAX_TOOL_ERROR_BYTES = 2_000


# ---- tool result size gate ---------------------------------------------------
#
# Compaction and microcompaction both act on results *already* in the context. This is
# the gate in front of it. The corpus says the gate is the part that was missing: across
# runs/, list_ad_account_creative_tags medians 284 KB and the largest single result on
# record is 292 KB — a quarter of the whole budget arriving in one step.

#: Ceiling for one result. Past this the full text goes to a file and the model is sent a
#: preview and the path. A tool may declare a lower limit, or opt out entirely.
DEFAULT_MAX_RESULT_SIZE_CHARS = int(os.environ.get("HARNESS_MAX_RESULT_CHARS", "50000"))

#: Ceiling for all results returned by one round of parallel calls. Four tools can each
#: hit the per-result ceiling and still fit — 200k/50k = 4 — so this exists for the fifth.
#: Groups are judged independently: 150k in one round and 150k in the next are both fine.
MAX_TOOL_RESULTS_PER_MESSAGE_CHARS = int(os.environ.get("HARNESS_MAX_GROUP_CHARS", "200000"))

#: How much of a persisted result the model still gets to see inline.
PREVIEW_BYTES = 2_000

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

#: Below this there is nothing worth a summarization call. Without it a small budget
#: compacts a one-step history, spending a model call to reclaim almost nothing.
MIN_STEPS_TO_COMPACT = 8


# ---- microcompaction ---------------------------------------------------------
#
# Compaction is all-or-nothing: a model call, a new root, the old history gone. Between
# those events nothing reclaims anything, and tool results are ~92% of the byte mass.
# Microcompaction is the cheap incremental version — no model call, no summary, just
# replacing stale tool results with a placeholder.

MICROCOMPACT_ENABLED = os.environ.get("HARNESS_MICROCOMPACT", "1") != "0"

#: Gap since the last model output, in minutes, after which the prompt cache is assumed
#: dead and clearing is therefore free. Google documents no implicit-cache TTL at all —
#: the published numbers are the token minimums in CACHE_FLOOR_TOKENS above — so this is
#: reasoned from guaranteed expiry rather than read off a stated lifetime: at an hour, any
#: implicit cache is long gone, so we never force a miss that would not have happened.
MICROCOMPACT_GAP_MINUTES = float(os.environ.get("HARNESS_MICROCOMPACT_GAP_MINUTES", "60"))

#: Tool results kept verbatim. Everything older is replaced. Five is enough working
#: context to carry on from and few enough that what we keep is smaller than what we
#: remove — which is the condition that makes a warm-cache clearing pay for itself.
MICROCOMPACT_KEEP_RECENT = int(os.environ.get("HARNESS_MICROCOMPACT_KEEP", "5"))

#: Tool results in context before the count trigger fires. Unlike the time-based trigger
#: this one fires while the cache is warm and costs one uncached turn.
MICROCOMPACT_TOOL_THRESHOLD = int(os.environ.get("HARNESS_MICROCOMPACT_THRESHOLD", "12"))

#: Don't break a warm cache for less than this. A cache miss costs you whatever you are
#: sending *after* the clearing, and saves you what you removed on every turn after — so
#: the trade is good only when we remove more than we keep. Clearing 5k while keeping 75k
#: is worse than doing nothing. The time-based path ignores this: it has no miss to
#: justify, because the cache had already expired.
MICROCOMPACT_MIN_RECLAIM_BYTES = int(os.environ.get("HARNESS_MICROCOMPACT_MIN_RECLAIM", "20000"))


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


# ---- attachments -------------------------------------------------------------
#
# Attachments are re-rendered into every rebuilt context — each compaction boundary and
# each resume — so unlike a tool result they are paid for repeatedly. Unlike the brief, a
# shed attachment is recoverable: the agent can read the file again. That asymmetry is why
# this budget is tight where the brief's is generous.

#: Whole-set cap for everything re-attached to a rebuilt context.
MAX_ATTACHMENT_TOKENS = 8_000

#: Per attached file. A topic file may reach MAX_MEMORY_FILE_BYTES (20,000 B, roughly
#: 5,000 tokens) and three of those would be the entire budget, so no single file takes
#: more than this share of it.
MAX_ATTACHED_FILE_TOKENS = 2_500

#: A plan refused twice is not going to be approved on the third try. Same shape as
#: MAX_CONSECUTIVE_COMPACT_FAILURES: you may fail, but not indefinitely.
MAX_PLAN_ATTEMPTS = 2


def bytes_per_token_for(text: str) -> int:
    """Chars per token for this content: 2 for JSON, 4 for everything else.

    Dense JSON is mostly single-character tokens — `{`, `}`, `:`, `,`, `"` — so it runs
    near two chars per token where prose runs near four. The distinction is load-bearing
    rather than cosmetic: at 4, a 100 KB payload estimates to 25k tokens when it is really
    closer to 50k, and the results that most need catching by a size gate are exactly the
    ones that would be waved through.

    `raw_decode` rather than `json.loads` because the MCP layer appends a derived-totals
    block after the JSON (`mcp.py`), so a strict parse would answer "not JSON" for our
    largest results — the precise case this exists to get right.
    """
    head = text.lstrip()
    if not head.startswith(("{", "[")):
        return 4
    try:
        json.JSONDecoder().raw_decode(head)
    except ValueError:
        return 4
    return 2


def approx_tokens(text: str, bytes_per_token: int = 4) -> int:
    """Rough token count, 4 chars/token by default.

    Deliberately an estimate: budgets are enforced per section on every write, and a
    `count_tokens` round trip per section would cost more than the precision is worth.

    Pass `bytes_per_token_for(text)` when the content type is unknown and the answer
    drives a threshold. The default stays 4 so callers measuring prose are unaffected.
    """
    return len(text) // max(1, bytes_per_token)

TRUNCATION_NOTICE = (
    "> [index truncated: it exceeded its line or byte cap] Entries were cut from the end "
    "of this index. Read the topic files in `agent/memory/` directly rather than assuming "
    "this list is complete."
)
