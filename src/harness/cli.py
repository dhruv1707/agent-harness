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
from .mcp import MCPBridge, load_servers
from .session import AgentSession
from .session_memory import SessionMemory
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

    floor = cache_floor(counter.model)
    if floor is None:
        print(f" CACHE  floor unknown for {counter.model} — cannot verify")
    elif stable_tokens is None:
        print(f" CACHE  floor is {floor:,} tok for {counter.model} — count unavailable")
    elif stable_tokens >= floor:
        print(f" CACHE  OK — prefix {stable_tokens:,} tok clears the {floor:,} tok floor")
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


def _cmd_run(args) -> int:
    common = {
        "model": args.model,
        "max_turns": args.max_turns,
        "policy_path": args.permissions,
        "auto_approve": args.yes,
        "budget": args.budget,
    }
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
        servers = [s for s in load_servers(args.mcp_config) if s.enabled]
        if args.no_mcp or not servers:
            if servers and args.no_mcp:
                print("[mcp] disabled by --no-mcp", file=sys.stderr)
            return await session.submit(
                args.task, on_text=None if args.quiet else on_text, on_event=on_event
            )

        # Bridged, not declared natively: these tools must pass the permission gate.
        async with MCPBridge(servers=servers) as bridge:
            discovered = await bridge.discover()
            for entry in discovered:
                session.registry.register(entry)
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
        print("\n[interrupted]", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[error] {_explain(exc)}", file=sys.stderr)
        return 1

    print()
    usage = f" usage={result.usage}" if result.usage else ""
    print(f"[{result.stop_reason}] turns={result.turns}{usage}", file=sys.stderr)
    if result.error:
        print(f"[error] {result.error}", file=sys.stderr)
    print(
        f"[transcript] harness transcript {session.session_id}",
        file=sys.stderr,
    )
    return 0 if result.stop_reason == "end_turn" else 1


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
