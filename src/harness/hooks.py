"""Lifecycle hooks: external commands that observe a child agent, and may stop it.

**A hook runs an arbitrary local command named in `agent/hooks.toml`.** That is the
feature, not an oversight — it is how you wire this harness to something it does not know
about: a notifier, an audit log, a house style checker. It is also the only place the
harness executes anything outside its own process, so the rules are set here rather than
inherited from somewhere.

A hook can do more than watch. Exit 2 sends its stderr back to the child as a new turn, so
a shell script that greps a deliverable for unapproved claims can bounce it without a model
being involved at all — deterministic where the verifier is judgement. The bounce is capped
at one, for the same reason the verifier's revision is: a gate that can bounce forever is a
gate that will.

Everything else fails open. A hook that errors, times out, or cannot be found is logged and
the run carries on, because a broken notifier must not cost a research run. Only exit 2 is
load-bearing, and it has to be asked for explicitly.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .config import AGENT_DIR, HOOK_TIMEOUT_SECONDS, ROOT

#: The events a hook can subscribe to. Deliberately only two: a child starting and a child
#: stopping. Tool-level and turn-level hooks are a far larger surface and nothing has asked.
EVENTS: tuple[str, ...] = ("subagent_start", "subagent_stop")

#: The exit code that means "I object, and here is why". Anything else non-zero is a broken
#: hook rather than a considered refusal, and is treated as such.
BLOCK_EXIT_CODE = 2


@dataclass(frozen=True)
class HookConfig:
    event: str
    command: str
    timeout: float = HOOK_TIMEOUT_SECONDS
    enabled: bool = True


@dataclass(frozen=True)
class HookResult:
    """What a hook said. `blocked` is the only field that changes what the harness does."""

    event: str
    command: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    #: Exit 2, and only exit 2.
    blocked: bool = False
    #: Set when the hook could not be run or would not finish.
    error: str | None = None


def load_hooks(path: Path | None = None) -> list[HookConfig]:
    """Read `agent/hooks.toml`. A missing file simply means no hooks."""
    path = path or (AGENT_DIR / "hooks.toml")
    if not path.is_file():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    hooks: list[HookConfig] = []
    for event in EVENTS:
        for entry in data.get(event, ()) or ():
            hooks.append(
                HookConfig(
                    event=event,
                    command=entry["command"],
                    timeout=float(entry.get("timeout", HOOK_TIMEOUT_SECONDS)),
                    enabled=entry.get("enabled", True),
                )
            )
    return hooks


async def run_hook(hook: HookConfig, payload: dict) -> HookResult:
    """Run one hook, handing it the payload as JSON on stdin.

    No `shell=True`: the command is split with `shlex`, so a hook cannot accidentally
    inherit shell expansion over a value the model chose. Output is captured rather than
    inherited so a chatty hook cannot corrupt the streamed answer.
    """
    try:
        argv = shlex.split(hook.command)
    except ValueError as exc:
        return HookResult(hook.event, hook.command, error=f"unparseable command: {exc}")
    if not argv:
        return HookResult(hook.event, hook.command, error="empty command")

    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=ROOT,
        )
    except (OSError, ValueError) as exc:
        return HookResult(hook.event, hook.command, error=f"{type(exc).__name__}: {exc}")

    try:
        out, err = await asyncio.wait_for(
            process.communicate(json.dumps(payload).encode("utf-8")), hook.timeout
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return HookResult(
            hook.event, hook.command, error=f"no answer within {hook.timeout:g}s"
        )

    code = process.returncode
    return HookResult(
        event=hook.event,
        command=hook.command,
        exit_code=code,
        stdout=out.decode("utf-8", "replace").strip(),
        stderr=err.decode("utf-8", "replace").strip(),
        blocked=code == BLOCK_EXIT_CODE,
    )


async def fire(event: str, payload: dict, hooks: list[HookConfig] | None) -> list[HookResult]:
    """Run every hook for an event, in file order, and report what they said.

    Serial rather than concurrent: hooks are few, and one that writes a log should see the
    events in the order they happened.
    """
    results: list[HookResult] = []
    for hook in hooks or []:
        if hook.event != event or not hook.enabled:
            continue
        result = await run_hook(hook, payload)
        if result.error:
            # Fails open. A broken notifier must not cost a research run.
            print(f"[hook] {event}: {result.error}", file=sys.stderr)
        elif result.exit_code not in (0, BLOCK_EXIT_CODE):
            print(
                f"[hook] {event}: exit {result.exit_code} — {result.stderr[:120]}",
                file=sys.stderr,
            )
        results.append(result)
    return results


def objection(results: list[HookResult]) -> str | None:
    """The first blocking hook's stderr, which becomes the child's next instruction."""
    for result in results:
        if result.blocked:
            return result.stderr or "a hook objected but said nothing about why"
    return None
