"""Command line: inspect the control plane, run the loop, read the transcript."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .config import CONTEXT_BUDGET_TOKENS, MAX_TURNS, MODEL, cache_floor
from .events import ToolCallReady, ToolCallStarted
from .prompt import AssembledPrompt, RunContext, build_effective_system_prompt
from .agents import AgentPool, _fingerprint, load_role
from .hooks import load_hooks
from .mcp import MCPBridge, load_servers
from .verify import sources_from_nodes, verify
from .session import AgentSession
from .session_memory import SessionMemory
from .tools import default_registry
from .transcript import Transcript

RULE = "=" * 78


def _explain(exc: Exception) -> str:
    """Say what actually went wrong, not just the exception class."""
    text = str(exc)
    if "api key" in text.lower() or "api_key" in text.lower():
        return "no GEMINI_API_KEY — put it in .env or export it"
    for attr in ("message", "details"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value:
            return value
    return f"{type(exc).__name__}: {text[:160]}"


# ---- harness prompt ----------------------------------------------------------


class _Counter:
    """Real token counts when the API is reachable, byte counts when it isn't."""

    def __init__(self, enabled: bool = True, model: str = MODEL):
        self.client = None
        self.model = model
        self.note = ""
        if not enabled:
            self.note = "token counting disabled (--no-tokens); showing bytes"
            return
        try:
            from google import genai

            self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        except KeyError:
            self.note = "no GEMINI_API_KEY — put it in .env or export it; showing bytes"
        except Exception as exc:
            self.note = f"{_explain(exc)}; showing bytes"

    def tokens(self, text: str) -> int | None:
        if self.client is None or not text:
            return None
        try:
            return self.client.models.count_tokens(
                model=self.model, contents=text
            ).total_tokens
        except Exception as exc:
            self.client = None
            self.note = f"{_explain(exc)}; showing bytes"
            return None

    def measure(self, text: str) -> str:
        count = self.tokens(text)
        if count is None:
            return f"{len(text.encode('utf-8')):,} B"
        return f"{count:,} tok"


def _render_prompt(prompt: AssembledPrompt, counter: _Counter) -> None:
    print(RULE)
    print(f" CACHEABLE PREFIX — system_instruction, stable across runs  [{counter.model}]")
    print(RULE)

    marked = False
    for index, layer in enumerate(prompt.layers, start=1):
        if not layer.cacheable and not marked:
            print()
            print("-" * 78)
            print(" ^^^ CACHE BREAKPOINT — below here rides in the user turn, never cached ^^^")
            print("-" * 78)
            marked = True
        print()
        print(f"=== [{index}] {layer.name}  ({layer.source})  {counter.measure(layer.text)} ===")
        print()
        print(layer.text)

    print()
    print(RULE)
    stable_tokens = counter.tokens(prompt.stable_text)
    print(
        f" TOTAL  cacheable: {counter.measure(prompt.stable_text)}"
        f"   volatile: {counter.measure(prompt.volatile_text)}"
    )

    # The cached prefix is system_instruction *plus* the tool declarations, and the tools
    # are the larger half — measuring the layers alone reported MISS on a prefix nearly
    # four times the floor.
    local_tools = json.dumps(default_registry().declarations())
    tool_tokens = counter.tokens(local_tools)
    if stable_tokens is not None and tool_tokens is not None:
        print(
            f" TOOLS  + {tool_tokens:,} tok for {len(default_registry())} local tools"
        )
        stable_tokens += tool_tokens

    # This command does not connect to MCP servers — that would need a live network and,
    # for an OAuth server, a human. So the count above is the prefix a run would have with
    # every server down. Naming the shortfall matters: MCP declarations are the larger half
    # of a real prefix, and judging the floor without them reports MISS on a prefix that
    # clears it four times over.
    uncounted = [s for s in load_servers() if s.enabled]

    floor = cache_floor(counter.model)
    if floor is None:
        print(f" CACHE  floor unknown for {counter.model} — cannot verify")
    elif stable_tokens is None:
        print(f" CACHE  floor is {floor:,} tok for {counter.model} — count unavailable")
    elif stable_tokens >= floor:
        print(f" CACHE  OK — prefix {stable_tokens:,} tok clears the {floor:,} tok floor")
    elif uncounted:
        names = ", ".join(s.name for s in uncounted)
        print(
            f" CACHE  UNVERIFIED — prefix without MCP tools is {stable_tokens:,} tok, "
            f"{floor - stable_tokens:,} tok short of the {floor:,} tok floor."
        )
        print(
            f"        {names} declare{'' if len(uncounted) > 1 else 's'} tools at connect, "
            f"uncounted here. "
            f"Run `harness run` and read total_cached_tokens for the real verdict."
        )
    else:
        print(
            f" CACHE  MISS — prefix {stable_tokens:,} tok is BELOW the {floor:,} tok "
            f"floor for {counter.model}."
        )
        print(
            f"        It will be re-billed in full every run. Either grow the prefix by "
            f"{floor - stable_tokens:,} tok or use a model with a lower floor."
        )

    if counter.note:
        print(f" note: {counter.note}")
    print(RULE)


def _cmd_prompt(args) -> int:
    override = None
    if args.system_prompt_file:
        override = args.system_prompt_file.read_text(encoding="utf-8")

    prompt = build_effective_system_prompt(
        run_context=RunContext(
            run_id=args.run_id,
            window=args.window,
            metric=args.metric,
            sources=tuple(args.source),
        ),
        override=override,
        agent=args.agent_system_prompt,
        append=args.append_system_prompt,
    )

    if args.raw:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "system_instruction": prompt.system_instruction,
                    "input": [prompt.initial_input(args.message)],
                    "store": False,
                    "stream": True,
                },
                indent=2,
            )
        )
    else:
        _render_prompt(prompt, _Counter(enabled=not args.no_tokens, model=args.model))
    return 0


# ---- harness run -------------------------------------------------------------


def _resume_hint(session_id: str) -> str:
    """No brief is written on an interrupt — that would cost a summarization round trip
    nobody asked for. The transcript carries everything a resume needs, so say where."""
    return (
        f"[interrupted] the turn is on the record: "
        f'harness run --resume {session_id} "..."'
    )


def _report_children(session) -> None:
    """What the team cost, after the pool has been drained inside the event loop.

    The cached share is the evidence that the fork kept the prefix intact — a child whose
    prompt had diverged from its parent's would show none of it.
    """
    pool = getattr(session, "pool", None)
    if pool is None or not pool.spawned:
        return
    outcomes = pool.ledger()
    failed = [o for o in outcomes if o.is_error]
    print(
        f"[team] {len(outcomes)} child agent(s)"
        + (f", {len(failed)} did not finish" if failed else ""),
        file=sys.stderr,
    )
    for outcome in outcomes:
        used = outcome.usage or {}
        total, cached = used.get("total_input_tokens"), used.get("total_cached_tokens")
        if total:
            share = f", {cached / total:.0%} cached" if cached else ", none cached"
            print(
                f"[team]   {outcome.child_id.split('.')[-1]:<15}{total:>8,} in{share}",
                file=sys.stderr,
            )
        elif outcome.is_error:
            print(
                f"[team]   {outcome.child_id.split('.')[-1]:<15}{outcome.reason}",
                file=sys.stderr,
            )


def _attach_pool(session, role_name: str | None) -> None:
    """Only a coordinator may delegate, and only through the pool it is handed here.

    A session given no pool cannot spawn at all — `spawn_agent` says so and the run
    carries on alone — which keeps delegation a capability the runtime grants rather than
    one the model can reach for.
    """
    if role_name != "coordinator":
        return
    session.pool = AgentPool(
        parent_id=session.session_id,
        registry=session.registry,
        client=session.client,
        agent_dir=session.agent_dir,
        runs_dir=session.transcript.path.parent,
        model=session.model,
        budget=session.budget,
        max_turns=session.max_turns,
        mcp_instructions=dict(session.mcp_instructions),
        asker=session.gate.asker,
        # Captured now, after MCP registration, so a child built later can be checked
        # against what the parent is actually caching.
        parent_prefix=_fingerprint(session.build_prompt().system_instruction),
        hooks=load_hooks(),
    )


def _cmd_team(args) -> int:
    """A coordinator that delegates, synthesizes, and hands work to a verifier.

    The same machinery as `run` — one session, one loop — with two additions: the
    coordinator carries a role, and it is given a pool it can spawn children into. Children
    share its cached prefix byte for byte and run under their own narrower policies.
    """
    args.resume = None
    args.from_node = None
    args.plan = False
    return _cmd_run(args, role_name="coordinator")


def _cmd_run(args, *, role_name: str | None = None) -> int:
    common = {
        "model": args.model,
        "max_turns": args.max_turns,
        "policy_path": args.permissions,
        "auto_approve": args.yes,
        "plan": args.plan,
        "budget": args.budget,
    }
    if role_name:
        common["role"] = load_role(role_name)
    try:
        if args.resume:
            session = AgentSession.resume(
                args.resume, from_node=args.from_node, **common
            )
        else:
            session = AgentSession.create(**common)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    where = f" from {args.from_node}" if args.from_node else ""
    print(
        f"[session {session.session_id}]{where}  model={session.model}",
        file=sys.stderr,
        flush=True,
    )

    def on_text(chunk: str) -> None:
        print(chunk, end="", flush=True)

    def on_event(event) -> None:
        if isinstance(event, ToolCallStarted):
            print(f"\n  · {event.name} …", file=sys.stderr, flush=True)
        elif isinstance(event, ToolCallReady):
            args_preview = json.dumps(event.arguments)[:120]
            print(f"  → {event.name}({args_preview})", file=sys.stderr, flush=True)

    async def go():
        try:
            return await _drive()
        finally:
            # Parent dies, children die — and this has to happen *inside* the loop that
            # created them. Cancelling from a second `asyncio.run` cancels tasks belonging
            # to a loop that is already closed, which only ever worked because spawning
            # used to block. A `finally` also covers the interrupt and error paths, which
            # returned before the old teardown and would now leave orphans.
            pool = getattr(session, "pool", None)
            if pool is not None and pool.spawned:
                await pool.cancel()

    async def _drive():
        servers = [s for s in load_servers(args.mcp_config) if s.enabled]
        if args.no_mcp or not servers:
            if servers and args.no_mcp:
                print("[mcp] disabled by --no-mcp", file=sys.stderr)
            _attach_pool(session, role_name)
            return await session.submit(
                args.task, on_text=None if args.quiet else on_text, on_event=on_event
            )

        # Bridged, not declared natively: these tools must pass the permission gate.
        async with MCPBridge(servers=servers) as bridge:
            discovered = await bridge.discover()
            for entry in discovered:
                session.registry.register(entry)
            session.mcp_instructions = dict(bridge.instructions)
            _attach_pool(session, role_name)
            names = ", ".join(sorted(bridge.clients)) or "none"
            print(f"[mcp] {len(discovered)} tools from {names}", file=sys.stderr)
            for failed, why in bridge.failures.items():
                print(f"[mcp] {failed} UNAVAILABLE — {why[:120]}", file=sys.stderr)
            return await session.submit(
                args.task, on_text=None if args.quiet else on_text, on_event=on_event
            )

    try:
        result = asyncio.run(go())
    except KeyboardInterrupt:
        # No brief was written — an interrupt should not cost a summarization round trip.
        # The transcript carries everything a resume needs, so point at it.
        print("\n[interrupted]", file=sys.stderr)
        _report_children(session)
        print(_resume_hint(session.session_id), file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[error] {_explain(exc)}", file=sys.stderr)
        _report_children(session)
        return 1

    _report_children(session)

    print()
    verdict = None
    usage = f" usage={result.usage}" if result.usage else ""
    print(f"[{result.stop_reason}] turns={result.turns}{usage}", file=sys.stderr)
    gate = session.size_gate
    if gate.persisted:
        print(
            f"[size-gate] {gate.persisted} oversized result(s) written to disk, "
            f"{gate.reclaimed_chars:,} chars kept out of context",
            file=sys.stderr,
        )
    micro = session.microcompaction
    if micro.clearings:
        print(
            f"[microcompact] {micro.clearings} clearing(s), "
            f"{micro.reclaimed_bytes:,} bytes of tool results dropped from context "
            f"({len(micro.cleared)} results)",
            file=sys.stderr,
        )
    if result.error:
        print(f"[error] {result.error}", file=sys.stderr)
    # Governance, not advice: the transcript holds every byte every tool returned, so
    # the figures in the report are checked against it rather than trusted.
    if result.text:
        verdict = verify(result.text, sources_from_nodes(session.transcript.all_nodes()))
        print(verdict.render(), file=sys.stderr)

    print(
        f"[transcript] harness transcript {session.session_id}",
        file=sys.stderr,
    )
    # A planning run with nobody to approve did exactly what it was asked to do. The plan
    # is in the transcript; failing the run would be reporting success as an error.
    # Ctrl-C has two landing sites: caught inside the loop, which returns an "interrupted"
    # result, or escaping to the handler above. Live testing found the same keystroke
    # exiting 1 with no resume hint down one path and 130 with one down the other.
    if result.stop_reason == "interrupted":
        print(_resume_hint(session.session_id), file=sys.stderr)
        return 130

    if result.stop_reason == "plan_pending":
        print(
            "[plan] proposed, and nobody was available to approve it. Review it above, "
            f"then: harness run --resume {session.session_id} \"go ahead\"",
            file=sys.stderr,
        )
        return 0
    if result.stop_reason != "end_turn":
        return 1
    # A report whose numbers contradict the tools is a failed run, not a caveat.
    return 2 if verdict and verdict.contradicted else 0


# ---- harness mcp -------------------------------------------------------------


def _cmd_mcp(args) -> int:
    """Connect (running OAuth if needed) and print what each server offers.

    Run this once per server to sign in, and to learn the real tool names before writing
    permission rules for them.
    """
    configured = load_servers(args.mcp_config)
    if args.server:
        # Naming one explicitly is consent to connect it, enabled flag or not — this is
        # how you authorize a server before turning it on for runs.
        servers = [s for s in configured if s.name == args.server]
    else:
        servers = [s for s in configured if s.enabled]
    if not servers:
        print("[error] no matching servers in agent/mcp.toml", file=sys.stderr)
        return 1

    async def go() -> int:
        async with MCPBridge(servers=servers) as bridge:
            tools = await bridge.discover()
            for failed, why in bridge.failures.items():
                print(f"\n{failed}  UNAVAILABLE\n  {why[:200]}", file=sys.stderr)
            for name in sorted(bridge.clients):
                owned = [t for t in tools if t.name.startswith(f"mcp__{name}__")]
                print(f"\n{name}  ({len(owned)} tools)")
                for entry in sorted(owned, key=lambda t: t.name):
                    safe = "parallel" if entry.concurrency_safe else "serial  "
                    first_line = entry.description.splitlines()[0] if entry.description else ""
                    print(f"  {safe}  {entry.name}")
                    if first_line:
                        print(f"            {first_line[:88]}")
            print(
                "\nTools with no read-only hint run serially. Add rules for these names "
                "to agent/permissions.toml — unmatched tools fall to the policy default.",
                file=sys.stderr,
            )
        return 0

    try:
        return asyncio.run(go())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[error] {_explain(exc)}", file=sys.stderr)
        return 1


# ---- harness memory ----------------------------------------------------------


def _cmd_memory(args) -> int:
    """Show a session's continuation brief."""
    brief = SessionMemory.load(args.session_id)
    if brief is None:
        print(
            f"[none] session {args.session_id} has no continuation brief — it never grew "
            "large enough to need one",
            file=sys.stderr,
        )
        return 1
    print(brief.render())
    print(f"[{brief.tokens():,} tokens estimated]", file=sys.stderr)
    return 0


# ---- harness transcript ------------------------------------------------------


def _cmd_transcript(args) -> int:
    try:
        transcript = Transcript.load(args.session_id)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(transcript.render())
    return 0


# ---- entry point -------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("prompt", help="Assemble and inspect the system prompt.")
    show.add_argument("--raw", action="store_true", help="Print the request JSON only.")
    show.add_argument("--system-prompt-file", type=Path, help="Override the default stack.")
    show.add_argument("--append-system-prompt", help="Appended last, after the breakpoint.")
    show.add_argument("--agent-system-prompt", help="A role description; extends the stack.")
    show.add_argument("--model", default=MODEL, help=f"Default: {MODEL}")
    show.add_argument("--window", default="last 7 days")
    show.add_argument("--metric", default=None)
    show.add_argument("--run-id", default="inspect")
    show.add_argument("--source", action="append", default=[], help="Repeatable.")
    show.add_argument("--message", default="Write today's scripts.", help="Sample user turn.")
    show.add_argument("--no-tokens", action="store_true", help="Skip the count_tokens calls.")
    show.set_defaults(fn=_cmd_prompt)

    run = sub.add_parser("run", help="Run the query loop against a task.")
    run.add_argument("task", help="What the agent should do.")
    run.add_argument("--resume", metavar="SESSION", help="Continue an existing session.")
    run.add_argument(
        "--from",
        dest="from_node",
        metavar="NODE",
        help="Branch from this node instead of the head. Requires --resume.",
    )
    run.add_argument("--model", default=MODEL, help=f"Default: {MODEL}")
    run.add_argument("--max-turns", type=int, default=MAX_TURNS)
    run.add_argument(
        "--budget",
        type=int,
        default=None,
        help=f"Context budget in tokens. Default: {CONTEXT_BUDGET_TOKENS:,}",
    )
    run.add_argument(
        "--permissions",
        type=Path,
        default=None,
        help="Policy file. Default: agent/permissions.toml",
    )
    run.add_argument(
        "--plan",
        action="store_true",
        help="Start in plan mode: read anything, change nothing, until a plan is approved.",
    )
    run.add_argument(
        "--yes",
        action="store_true",
        help="Approve every 'ask' without prompting. For unattended runs; think first.",
    )
    run.add_argument(
        "--mcp-config", type=Path, default=None, help="Default: agent/mcp.toml"
    )
    run.add_argument(
        "--no-mcp", action="store_true", help="Skip MCP servers for this run."
    )
    run.add_argument("-q", "--quiet", action="store_true", help="Suppress streamed text.")
    run.set_defaults(fn=_cmd_run)

    team = sub.add_parser(
        "team",
        help="Run a coordinator that delegates to researchers, an implementer and a verifier.",
    )
    team.add_argument("task", help="What the team should do.")
    team.add_argument("--model", default=MODEL, help=f"Default: {MODEL}")
    team.add_argument("--max-turns", type=int, default=MAX_TURNS)
    team.add_argument("--budget", type=int, default=None, help="Context budget in tokens.")
    team.add_argument("--permissions", type=Path, default=None, help="Coordinator policy.")
    team.add_argument("--yes", action="store_true", help="Approve every 'ask' without asking.")
    team.add_argument("--mcp-config", type=Path, default=None)
    team.add_argument("--no-mcp", action="store_true", help="Skip MCP servers for this run.")
    team.add_argument("-q", "--quiet", action="store_true", help="Suppress streamed text.")
    team.set_defaults(fn=_cmd_team)

    mcp_cmd = sub.add_parser("mcp", help="Connect to MCP servers and list their tools.")
    mcp_cmd.add_argument(
        "server", nargs="?", help="Only this server. Default: every enabled one."
    )
    mcp_cmd.add_argument("--mcp-config", type=Path, default=None)
    mcp_cmd.set_defaults(fn=_cmd_mcp)

    mem = sub.add_parser("memory", help="Show a session's continuation brief.")
    mem.add_argument("session_id")
    mem.set_defaults(fn=_cmd_memory)

    tree = sub.add_parser("transcript", help="Render a session's node tree.")
    tree.add_argument("session_id")
    tree.set_defaults(fn=_cmd_transcript)

    args = parser.parse_args()
    if getattr(args, "from_node", None) and not getattr(args, "resume", None):
        parser.error("--from requires --resume")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
