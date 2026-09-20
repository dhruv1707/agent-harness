# Agent Harness

An LLM agent that does creative strategy for paid social advertising, and the Python runtime
that runs it.

You give it a task in plain English. It pulls performance data from an ad account, reads the
ads' transcripts, works out what the best-performing ads have in common, and writes the
deliverable — a performance review, creator scripts, landing-page copy. Before it returns,
every number in its answer is checked against the data it actually retrieved.

```bash
harness run "which ads performed best last week, and what were their opening lines?"
harness team "review last week's top ads and write 3 scripts iterating on the best hook"
```

**Stack:** Python 3.10+, Google Gemini (Interactions API), and MCP for connecting data
sources. No agent framework — the runtime is about 5,800 lines, and [why that is a deliberate
choice](#why-not-langgraph) is explained below.

---

## Who it's for

Teams who write briefs for content creators based on ad performance, and do it by hand today.

- **Creative strategists at DTC brands** — the weekly loop of pulling the top ads, reading
  their hooks, spotting the pattern, and writing the next round of scripts.
- **Performance creative and UGC teams** — iterating on proven hooks without reusing lines
  from ads that already ran, which makes a variation test nothing new.
- **Agencies running paid social** across several brands, each with its own voice, its own
  approved product claims, and its own list of things that must not be said on camera.
- **Growth teams** who want a read on which creative is actually carrying spend, and which is
  quietly losing money.

It is scoped to creative. When the data points to budget, targeting or a landing page as the
real problem, it says so and stops there.

## In use at Plufl

Agent Harness is currently used by creative strategists at **Plufl**, the DTC brand that
appeared on *Shark Tank*.

The `agent/` directory in this repository is configured for Plufl's account. To use it for
another brand, replace that directory — see [Configuring it for a brand](#configuring-it-for-a-brand).

New to paid social? There is a short [glossary](#glossary) at the end.

---

## Quick start

```bash
git clone <this repo> && cd agent-harness
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

echo "GEMINI_API_KEY=..." > .env      # gitignored
harness mcp atria                      # one-time sign-in to the ad data source
harness run "what won last week?"
pytest                                 # 289 tests
```

---

## How a run works

The model does not drive the program. The harness calls the model in a loop and decides what
it sees, which tools it may use, and what happens to their results.

```text
task
 │
 ├─ 1. build the prompt      layered markdown files from agent/, plus the task
 │
 ├─ 2. call the model        stream its response
 ├─ 3. run tool calls        each starts as soon as the model finishes writing it;
 │                           concurrency-safe calls run in parallel;
 │                           every call is permission-checked
 ├─ 4. collect results       in the order the model asked for them, not the order they finished
 ├─ 5. record everything     appended to the session's transcript on disk
 │     └─ repeat 2–5 until the model stops calling tools
 │
 └─ 6. check the answer      every figure compared against what the tools returned
```

In more detail:

**1. Build the prompt.** The system prompt is assembled from markdown files in `agent/`:
general rules, brand governance, a memory index, and guidance published by each data source.
It is split in two. The *stable* part is byte-identical on every run, so the model provider
can cache it; the *per-run* part — today's date, the turn budget, the task — goes in the
first user message. `harness prompt` prints the assembled result.

**2–3. Call the model, run its tools.** The response streams in. When a tool call finishes
streaming, it is dispatched immediately, while the model is still producing the rest of its
response. Tools flagged as safe to run concurrently run up to eight at a time; the rest run
one at a time. Every call passes a permission check first, and every call has a timeout.

**4. Collect results.** Each tool call gets exactly one result — success, error, timeout,
denial, or cancellation — and results are handed back in the order the model requested them.
If a call ever went missing, the model would see a question with no answer and invent one,
so the harness raises an error instead.

**5. Record everything.** Every step is appended to a transcript file under `runs/`. The
conversation sent to the model is rebuilt from that file, which is why a session can be
resumed or branched after the process exits.

**6. Check the answer.** Every figure in the output is matched against the tool results for
the specific ad it is written under. A figure that contradicts the data fails the run.

Two more things happen when the conversation gets long:

- **Session memory.** Once the conversation passes about 12,000 tokens, the harness
  periodically writes a structured summary of what has been learned so far.
- **Compaction.** Near the context limit, older history is replaced by that summary. The
  original task, any approved plan, and the files the agent had open are re-attached in full,
  and the metrics in the summary are written by the harness from the tool results rather than
  recalled by the model.

---

## Repository layout

```text
src/harness/
  cli.py             command-line entry point: run, team, prompt, transcript, memory, mcp
  session.py         opens or resumes a session and runs one task through the loop
  loop.py            the main loop: call the model, dispatch tools, record, repeat
  events.py          turns the model's streaming output into typed events
  executor.py        runs tool calls: concurrency, timeouts, permission checks, results
  permissions.py     the allow / ask / confirm / deny policy, and plan mode
  tools.py           built-in tools and the tool registry
  mcp.py             connects MCP servers and exposes their tools as ordinary local tools
  prompt.py          assembles the system prompt
  transcript.py      the session log: append-only JSONL, stored as a tree
  session_memory.py  the running summary, and the rules for when to write it
  compaction.py      shrinks the conversation when it gets too long
  attachments.py     what is carried across a compaction
  verify.py          checks figures in the output against the tool results
  agents.py          child agents, for team mode
  hooks.py           runs your own scripts when a child agent starts or stops
  config.py          every tunable constant, each with the reason for its value
agent/               configuration — prompts, brand material, permissions, data sources
scripts/             the two hook scripts that ship enabled
tests/               the test suite
```

---

## Core concepts

**Transcript.** Each session is one JSONL file under `runs/`, holding every step: user
messages, model output, tool calls and tool results. Steps are stored as a tree, so a session
can branch from any earlier point. It is the only source of truth — the in-memory
conversation is rebuilt from it at three points: opening a session, branching, and
compacting. The in-memory copy can say *less* than the transcript does, once
microcompaction has cleared stale tool results from it, but never more.

**Permission policy.** Every tool call is checked against `agent/permissions.toml` before it
runs. There are four outcomes: *allow*, *ask* (prompt a person), *confirm* (prompt a person,
and `--yes` cannot answer on their behalf), and *deny*. When several rules match, the most
restrictive wins.

**Plan mode.** With `--plan`, any tool that writes or spends money is refused. The agent
researches, proposes a plan, and a person approves or rejects it. On approval the same
session continues with everything it already learned.

**Data sources over MCP.** External data — here, Meta ad data via Atria — is connected over
the Model Context Protocol. The harness is the MCP client and registers each remote tool as a
local one, so remote calls go through the same permission check as everything else.

**Team mode.** `harness team` runs a *coordinator* agent that starts *child* agents in
parallel: researchers that may only read, an implementer that writes the deliverable, and a
verifier that checks it against the evidence without seeing how it was made. The coordinator
combines the researchers' findings into a single brief for the implementer, and if the
verifier finds problems, sends the work back to the implementer once. Children cannot start
children of their own.

**Tool result size gate.** Ad-platform listings are large — a single creative-tag listing
runs to 284 KB, a quarter of the context budget in one step. Any result over ~50k characters
is written to a file under `runs/` and the model is sent a short preview plus a handle it can
read back with `read_tool_result`. A round of parallel calls that together exceed ~200k is
trimmed the same way, largest first. A tool can declare a lower limit or opt out. Once the
model has been shown a result, what it was shown never changes — altering it afterwards would
invalidate the provider's cache for everything following it.

**Microcompaction.** Tool results are about 92% of the bytes in a long run, and most go
stale quickly — a listing pulled twenty turns ago is rarely read again. Microcompaction
replaces old tool results with a short placeholder, keeping the five most recent. It runs
in two cases: when more than an hour has passed since the model last spoke, which means the
provider's prompt cache has expired and shrinking the context costs nothing; and when tool
results pile past a threshold during an active session. The transcript keeps every byte, so
this only narrows what the model re-reads, never what the harness checks figures against.

**Hooks.** `agent/hooks.toml` names scripts to run when a child agent starts or stops. They
receive the event as JSON on stdin. Exit code 2 sends the script's stderr back to the child
as a correction; any other failure is logged and ignored. Two ship enabled: a cost logger,
and a check that rejects unapproved product claims in script copy.

---

## Commands

| command | what it does |
|---|---|
| `harness run "<task>"` | run one agent on one task |
| `harness run --plan "<task>"` | research first; nothing changes until a person approves a plan |
| `harness team "<task>"` | run a coordinator with researchers, an implementer and a verifier |
| `harness run --resume <id> "<msg>"` | continue a session; add `--from <node>` to branch it |
| `harness prompt` | print the assembled system prompt, and whether it is cacheable |
| `harness transcript <id>` | show a session's steps as a tree |
| `harness memory <id>` | show a session's running summary |
| `harness mcp <server>` | connect a data source and list its tools |

`--yes` approves ordinary permission prompts for unattended runs, though never a plan.
`--no-mcp` runs without data sources. `--max-turns` and `--budget` limit the run.

---

## Configuring it for a brand

Everything brand-specific lives in `agent/`, and none of it is code.

| file | holds |
|---|---|
| `CLAUDE.md` | governance: the brand, the ad account, metric thresholds, house conventions |
| `memory/brand-voice.md` | product facts, approved phrasing, claims that need legal sign-off |
| `memory/brief-samples.md` | approved scripts, used as a reference for tone |
| `memory/hook-patterns.md` | a taxonomy of hook types, which the agent adds to as it finds new ones |
| `memory/script-craft.md` | how a script is structured, and how a review report is laid out |
| `prompts/*.md` | the agent's standing rules, general to creative work |
| `roles/*.md` | instructions for each team-mode role |
| `permissions*.toml` | what each role may read, write, or must ask before doing |
| `mcp.toml` | which data sources to connect, and which of their tools to load |
| `hooks.toml` | scripts to run when a child agent starts or stops |

---

## Why not LangGraph

LangGraph is a good framework, and for many agents it is the right choice. It represents an
agent as a graph of steps over shared state, and provides checkpointing, persistence and
human-in-the-loop pauses. If you know the steps in advance, it gets you there with very
little code.

We built our own runtime for three reasons.

**1. The steps aren't known in advance.** An early version of this agent ran a fixed
sequence written into its prompt: rank the ads, read the transcripts, classify the hooks,
write the scripts. It couldn't do anything else well — a request for a landing-page headline
still carried nearly ten kilobytes of ad-review instructions. The sequence was removed
([`dc2004b`](https://github.com/dhruv1707/agent-harness/commit/dc2004b)), and the agent now
chooses its own tools at each turn. A graph fixes the path ahead of time; this agent needs a
loop.

**2. The behaviour that matters lives below a framework's abstractions.** Each of these
depends on controlling exactly what goes into a request, or exactly when code runs:

- *Prompt caching.* Child agents reuse their parent's cached prompt. That only works if the
  cached portion is byte-identical, so each child's role is placed in the user message rather
  than the system prompt, and children keep the full tool list — including tools their
  permissions forbid — because removing one would change the cached bytes.
- *Starting tools mid-stream.* A tool starts while the model is still generating. This
  depends on yielding to the event loop at exactly the right point
  ([`6e6c4cf`](https://github.com/dhruv1707/agent-harness/commit/6e6c4cf)).
- *One permission check for every call.* Data sources are connected as local tools rather
  than handed to the model provider, so remote calls cannot bypass the policy
  ([`96e7a4b`](https://github.com/dhruv1707/agent-harness/commit/96e7a4b)).
- *What survives a compaction.* The summary's figures are copied from tool results by the
  harness, not recalled by the model
  ([`1fdc475`](https://github.com/dhruv1707/agent-harness/commit/1fdc475)).
- *Provider requirements.* Gemini signs its reasoning steps and rejects a request unless each
  signature is sent back exactly as received. Switching from Anthropic's API to Gemini's was
  a single commit ([`524ad1c`](https://github.com/dhruv1707/agent-harness/commit/524ad1c)).

Each of these is a few lines in code you own, and a workaround in code you don't.

**3. Every failure should be readable.** The main loop is under 500 lines. Every serious bug
found during development — a cleanup step cancelling tasks on an event loop that had already
closed, compaction cutting off the summary it had just written, the model's own summary being
treated as evidence for its own figures — was found by reading that code and fixed in one
file.

**When LangGraph is the better choice.** If your agent follows a path you can draw before it
runs — an approval pipeline, a fixed multi-step process, a support flow with known branches —
LangGraph gives you that structure, with checkpointing and persistence, for much less code.
If the value of your agent is in how it manages context, cost, permissions and failure, you
end up owning those parts either way.

---

## Design principles

Each of these came from a run that went wrong.

- **If it isn't in the transcript, it doesn't survive.** Anything held only in memory is lost
  on restart or compaction.
- **The model's own writing is not evidence.** Verification uses what tools returned and what
  a person asked for, never a summary the model wrote.
- **Prevent a failure rather than report it.** Wait for child agents instead of discarding
  their work; refuse a child whose cached prompt has drifted instead of paying full price;
  put a timeout on every wait.
- **Retry, but not forever.** Plan revisions, hook objections and failed compactions are all
  capped in code.
- **Asking the model to be careful is not a safeguard.** Anything that can be checked in code
  is checked in code.

---

## Limitations

- Built for Meta ad accounts through Atria. Another data source needs an MCP server and an
  entry in `mcp.toml`, but the working rules assume Meta's metrics.
- Atria requires a person to sign in again roughly every twelve hours, so fully unattended
  daily runs are not yet practical.
- A plan awaiting approval does not survive a restart.
- Hooks run local scripts named in a config file, so `agent/hooks.toml` should be reviewed
  like code.

---

## Glossary

For engineers new to paid social.

| term | meaning |
|---|---|
| **ROAS** | return on ad spend — revenue divided by spend |
| **CPA** | cost per acquisition — spend divided by purchases |
| **Hook** | the first few seconds of a video ad; the line meant to stop someone scrolling |
| **Thumbstop ratio** | the share of impressions that kept watching past the first three seconds — a measure of the hook |
| **UGC** | ads in the style of user-generated content, filmed by creators rather than a studio |
| **Whitelisting** | running ads through a creator's own account with their permission; "creator accounts" |
| **Brief** | the instructions and scripts a creator receives before filming |
| **DTC** | direct-to-consumer — a brand selling online, rather than through retailers |
| **Atria** | the ad analytics platform this agent reads Meta ad data from |
| **MCP** | Model Context Protocol — a standard way to connect an agent to external tools and data |
