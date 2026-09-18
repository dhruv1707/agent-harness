# Agent Harness

**An AI creative strategist for paid social.** It reads what is actually winning in your ad
account, works out *why*, and turns that into briefs and scripts your creators can shoot —
then checks its own numbers against the data before you ever see them.

```bash
harness team "review last week's top ads and write 3 iteration scripts on the best hook"
```

It is built as a *harness* rather than a prompt: a runtime that decides what the model sees,
what it may touch, when it has to stop and ask, and what counts as true. Most of the code
exists to stop a capable model from being confidently wrong.

---

## Who it's for

Teams who brief creators off performance data, and are tired of doing it by hand.

- **Creative strategists at DTC brands** — the weekly loop of pulling top ads, reading the
  hooks, spotting the pattern, and writing the next round of scripts.
- **Performance creative and UGC teams** — iterating on proven hooks without copying lines
  from ads that already ran, which is how variations end up testing nothing.
- **Agencies running paid social** for several brands, where each brand has its own voice,
  its own approved claims, and its own list of things nobody may say on camera.
- **Growth teams** who want a creative read on the account — which bets are carrying spend,
  which are quietly losing money — without waiting on a strategist's calendar.

It is deliberately scoped to creative. It will tell you when the real problem is budget,
targeting or a landing page, and stop there.

## In use at Plufl

Agent Harness is currently used by creative strategists at **Plufl**, the DTC brand that
appeared on *Shark Tank*.

The `agent/` directory in this repository is configured for Plufl's account: its brand voice,
approved claims, hook taxonomy and house script format. To run it for another brand, replace
that directory — see [Configuring it for your brand](#configuring-it-for-your-brand).

---

## What it does

**Reads the account honestly.** Ranking by ROAS alone sorts by the denominator, so a $6 ad
with one order outranks everything that actually ran. The harness applies a purchase floor
and a spend floor of twice the account's own CPA, then reports in two bands — *proven at
scale* and *efficient but unproven* — plus a standing line naming where the money actually
is, whatever its ratio.

**Finds the hooks, verbatim.** Hooks come from transcripts, never reconstructed from an ad
name. A hook it could not read is marked `UNVERIFIED` with the reason, rather than guessed.
Free cached transcripts are always tried before paid transcription.

**Writes in your voice without copying it.** It borrows the register of your approved
scripts — the pacing, where the turn lands, how a claim gets deflated — but not their
sentences. The approved product boilerplate is the one thing reused word for word.

**Checks its own numbers.** Every figure in the output is matched against what the tools
actually returned, per ad. A number that contradicts the record fails the run. This exists
because a model will happily report a real ad's id next to a plausible, invented spend
figure — right about which ad won, wrong about every number beside it.

**Works as a team.** A coordinator spawns researchers in parallel, synthesizes what they
found into one creative brief, hands it to an implementer, and sends the result to an
independent verifier that never sees how it was made. The verifier gets one chance to send
work back.

**Plans before it acts, when you want it to.** In plan mode it researches freely, changes
nothing, and proposes a plan for a person to approve — the same session then carries on with
everything it already learned.

---

## Quick start

```bash
git clone <this repo> && cd agent-harness
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

echo "GEMINI_API_KEY=..." > .env           # gitignored
harness mcp atria                           # one-time sign-in to your ad data source
harness run "what won last week?"
```

Requires Python 3.10+. Runs on Gemini via the Interactions API.

## Commands

| command | what it does |
|---|---|
| `harness run "<task>"` | one agent, one task — the everyday path |
| `harness run --plan "<task>"` | research first; nothing changes until a person approves the plan |
| `harness team "<task>"` | coordinator, parallel researchers, implementer, independent verifier |
| `harness run --resume <id> "<msg>"` | continue a session — or `--from <node>` to branch it |
| `harness prompt` | assemble and inspect the exact system prompt, with a cache verdict |
| `harness transcript <id>` | render a session's history as a tree |
| `harness memory <id>` | show a session's continuation brief |
| `harness mcp <server>` | connect a data source and list what it exposes |

Useful flags: `--yes` answers ordinary permission prompts for unattended runs (it will never
approve a plan), `--no-mcp` runs without data sources, `--max-turns` and `--budget` bound the
run.

---

## How it works

```text
harness run / team
 └─ control plane ─── layered prompt: identity → rules → plan mode → working rules
 │                    → governance → memory index → data-source guidance
 │                    ─ cache breakpoint ─ → run context → your task
 └─ query loop ────── stream the model; dispatch each tool call the moment it completes
     ├─ executor ──── safe calls in parallel, writes one at a time, every call gated
     ├─ permissions ─ allow / ask / confirm / deny, plan mode layered on top
     ├─ ledger ────── exactly one result per call, in the order asked, whatever happened
     ├─ memory ────── a continuation brief written at coherent moments, not every turn
     ├─ compaction ── history replaced by the brief, with the task, plan and files re-attached
     └─ agent pool ── child agents sharing the parent's cached prefix byte for byte
```

A few pieces worth knowing about:

- **The transcript is the only thing that is true.** Every session is an append-only tree
  on disk. The in-memory conversation is a projection of it, rebuilt at exactly three
  points: opening a session, branching, and compacting. Resuming is reopening a file.
- **Streaming dispatch.** A tool starts the moment its call finishes streaming, while the
  model is still writing the rest of its turn. Read-only calls run up to eight at a time.
- **The runtime authorizes; the model only proposes.** Data sources are bridged over MCP as
  ordinary local tools rather than handed to the provider, so every call passes the same
  permission gate. Policies live in `agent/permissions*.toml`.
- **Numbers are transcribed, not recalled.** When the history is compacted, the metrics in
  the brief are written by the harness from the tool results — not by the model, which by
  then can no longer see them.
- **Children share the cache.** Every child agent gets the parent's prompt and full tool
  list, including tools its policy forbids, because removing a declaration would change the
  cached prefix. The role rides in the user turn instead. A child whose prefix drifts from
  its parent's is refused rather than silently billed at full price.
- **Hooks.** `agent/hooks.toml` runs your own commands when a child agent starts or stops.
  Exit 2 sends the hook's objection back to the child. Two ship armed: a cost logger, and a
  check that refuses unapproved claims in spoken script copy.

## Why we built our own harness instead of using LangGraph

LangGraph is a good framework, and for plenty of agents it is the right call. It models an
agent as a graph — nodes, edges, shared state — and gives you checkpointing, persistence and
human-in-the-loop interrupts out of the box. If you can draw your workflow before it runs,
that is most of what you need.

We couldn't, and that turned out to be the whole reason.

**The agent isn't a graph.** The first version of this harness worked like one: a fixed
sequence written into the prompt — rank the ads, read the transcripts, classify the hooks,
write the scripts. Nearly ten kilobytes of that procedure rode in every prompt, including one
asking it to rewrite a landing page headline. So the choreography was deleted
([`dc2004b`](https://github.com/dhruv1707/agent-harness/commit/dc2004b)), and the agent now chooses its own path from the tools
and rules it is given. A graph encodes the path up front. What we needed was a loop, and
control over everything that happens inside each turn of it.

**The parts that matter are the parts a framework owns.** Nearly every hard problem here
lived below the level a framework exposes:

- **The prompt cache is decided byte by byte.** Child agents share their parent's cached
  prefix exactly. The role rides in the user turn rather than the system prompt, and children
  keep tool declarations they are not allowed to call, because removing one would change the
  prefix. A child whose prefix drifts is refused rather than billed at full price. None of
  that works unless you own exactly what goes into each request.
- **Tools start mid-stream.** A tool begins the moment its call finishes streaming, while the
  model is still writing the rest of its turn. Getting that right came down to a single
  `await asyncio.sleep(0)` in the event loop ([`6e6c4cf`](https://github.com/dhruv1707/agent-harness/commit/6e6c4cf)).
- **One permission gate for everything.** Data sources are bridged over MCP as ordinary local
  tools instead of being handed to the model provider, so every call — local or remote —
  passes the same allow / ask / confirm / deny policy ([`96e7a4b`](https://github.com/dhruv1707/agent-harness/commit/96e7a4b)).
- **Context is governed, not just stored.** Compaction replaces history with a brief,
  re-attaches the task and plan, and has the harness — not the model — write the numbers
  into it ([`1fdc475`](https://github.com/dhruv1707/agent-harness/commit/1fdc475)).

Each of those is a few lines in a loop you own, and a fight with an abstraction in one you
don't.

**Provider details are ours to get right.** The harness moved from Anthropic's API to
Gemini's Interactions API in one commit ([`524ad1c`](https://github.com/dhruv1707/agent-harness/commit/524ad1c)). Gemini signs its
thought steps and rejects the next request unless each signature is replayed verbatim with
the history. That kind of detail is simple when you build the request yourself.

**We needed to read every failure.** The runtime is about 5,800 lines, and the loop at its
centre is under 500. Every serious bug found while building it — a teardown cancelling tasks
on an event loop that had already closed, compaction truncating the brief it had just
written, the model's own summary being counted as evidence for its own numbers — was found by
reading that code and fixed in a single file.

### The same reasoning as Claude Code

This harness was built chapter by chapter alongside *[Harness Engineering: A Design Guide to
Claude Code](https://harness-books.agentway.dev/en/book1-claude-code/)*, which states the
premise plainly:

> *"The center of gravity is not model capability, but how the harness organizes constraints
> and execution."*

The book treats the control plane, the main loop, tool permissions, context governance,
recovery paths and multi-agent verification as *"one coherent skeleton"* — not features
bolted onto a model, but the structure that decides whether a capable model stays on course
once it is plugged into a real system.

Claude Code is built that way: its own query loop, its own streaming tool executor, its own
permission system, its own context compaction and sub-agents. We made the same bet for a
creative agent instead of a coding one. The model is an input; the harness is what we are
actually building.

### When LangGraph is the better choice

If your agent follows a path you can draw before it runs — an approval pipeline, a fixed
multi-step process, a support flow with known branches — LangGraph gives you that shape,
along with checkpointing and persistence, for very little code. If the value of your agent
lives in how it manages context, cost, permissions and failure, you will end up owning those
parts either way. We chose to own them from the start.

## Configuring it for your brand

Everything brand-specific lives in `agent/`, and none of it is code.

| file | holds |
|---|---|
| `CLAUDE.md` | governance — the brand, the account, the metric floors, house conventions |
| `memory/brand-voice.md` | product facts, approved phrasing, claims that need sign-off |
| `memory/brief-samples.md` | approved scripts, verbatim — the register to learn from |
| `memory/hook-patterns.md` | the hook taxonomy, which the agent extends as it finds new ones |
| `memory/script-craft.md` | how a script is built and how a review report is laid out |
| `prompts/*.md` | the agent's standing rules — general to creative work, not to one brand |
| `roles/*.md` | what the coordinator, researcher, implementer and verifier are each for |
| `permissions*.toml` | what each role may read, write, or must ask about |
| `mcp.toml` | which data sources to connect, and which of their tools to load |
| `hooks.toml` | your own commands to run around child agents |

## Principles

A few rules the code keeps returning to, each learned from a run that went wrong:

- **If it isn't in the transcript, it doesn't survive.** Anything that lives only in memory
  is gone after a restart or a compaction.
- **A figure the model wrote is not evidence.** Verification reads what tools returned and
  what a human asked. The model's own summaries never count as sources, however plausible.
- **Prevent the failure rather than reporting it.** Wait for children instead of evicting
  their work; refuse a drifted fork instead of paying for it; bound every wait instead of
  hoping nothing hangs.
- **You may fail, but not indefinitely.** Every retry is capped — a plan revision, a
  verifier bounce, a hook's objection, a failed compaction.
- **Asking a model to be careful is not a control.** Where something can be checked
  deterministically, it is.

## Limitations

- Built around Meta ad accounts through Atria. Other data sources need an MCP server and a
  line in `mcp.toml`, but the working rules assume Meta's metrics.
- Atria's OAuth needs a person to sign in again roughly every twelve hours, so a fully
  unattended daily schedule is not yet practical.
- A pending plan does not survive a restart; plan mode is per process.
- Hooks run local commands named in a config file. That is the feature, and it means
  `agent/hooks.toml` should be treated like code.

## Background

Built step by step alongside *[Harness Engineering: A Design Guide to Claude
Code](https://harness-books.agentway.dev/en/book1-claude-code/)* — the prompt as control
plane, the query loop, tools and permissions, context governance, and multi-agent
verification — adapted from a coding agent to a creative one.

```bash
pytest   # 289 tests
```
