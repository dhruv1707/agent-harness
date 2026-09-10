"""Permission: a runtime object, not a boolean.

Chapter 4's constraint is that three answers are needed, not two — *"if an agent cannot
distinguish 'I can do this,' 'I cannot do this,' and 'I must ask,' it should not touch a
terminal."* `ask` is what makes an interactive harness possible at all, and it is the state
you cannot retrofit onto a bool.

The model proposes; the runtime authorizes.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Decision = Literal["allow", "deny", "ask"]

#: Higher is more restrictive. When several rules match, the most restrictive wins.
_RANK: dict[str, int] = {"allow": 0, "ask": 1, "deny": 2}


@dataclass(frozen=True)
class PermissionResult:
    """One authorization decision, with the rule that produced it.

    `rule` is kept so a decision is auditable after the fact — "why was this allowed?"
    should never require re-deriving the policy.
    """

    decision: Decision
    reason: str = ""
    rule: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    @property
    def denied(self) -> bool:
        return self.decision == "deny"

    def __str__(self) -> str:
        via = f" [{self.rule}]" if self.rule else ""
        return f"{self.decision}{via}: {self.reason}" if self.reason else f"{self.decision}{via}"


def matches_pattern(pattern: str, name: str) -> bool:
    """Exact match, or a trailing `*` prefix match (`mcp__triplewhale__*`)."""
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return pattern == name


@dataclass(frozen=True)
class PermissionPolicy:
    """The rules. Editable at `agent/permissions.toml`, alongside the control plane."""

    allow: tuple[str, ...] = ()
    ask: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    default: Decision = "ask"

    def evaluate(self, tool_name: str) -> PermissionResult:
        """Most restrictive matching rule wins: deny > ask > allow.

        A narrow `allow` cannot punch a hole in a broad `deny` — the same shape as the
        prompt precedence in step 1, where a job description extends the constitution but
        cannot wipe it.
        """
        matched: list[tuple[Decision, str]] = []
        for decision, patterns in (
            ("deny", self.deny),
            ("ask", self.ask),
            ("allow", self.allow),
        ):
            for pattern in patterns:
                if matches_pattern(pattern, tool_name):
                    matched.append((decision, pattern))  # type: ignore[arg-type]

        if not matched:
            return PermissionResult(
                self.default,
                f"no rule matched {tool_name}; policy default is {self.default}",
            )

        decision, pattern = max(matched, key=lambda entry: _RANK[entry[0]])
        losers = [p for d, p in matched if p != pattern]
        reason = f"matched {pattern}"
        if losers:
            reason += f" (most restrictive of {len(matched)} matching rules)"
        return PermissionResult(decision, reason, rule=pattern)

    @classmethod
    def load(cls, path: Path) -> PermissionPolicy:
        """Read a policy file. A missing file means the safe default: ask about everything."""
        if not path.is_file():
            return cls()
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return cls(
            allow=tuple(data.get("allow", ())),
            ask=tuple(data.get("ask", ())),
            deny=tuple(data.get("deny", ())),
            default=data.get("default", "ask"),
        )


# ---- asking a human ----------------------------------------------------------

#: Returns "y" (allow once), "n" (deny), or "a" (allow this tool for the session).
Asker = Callable[[str, dict], str]


def terminal_asker(tool_name: str, arguments: dict) -> str:
    """Prompt on the terminal. Runs in a worker thread — never on the event loop."""
    rendered = json.dumps(arguments, indent=2) if arguments else "{}"
    print(f"\n  permission required: {tool_name}", file=sys.stderr)
    for line in rendered.splitlines():
        print(f"      {line}", file=sys.stderr)
    try:
        answer = input("  allow? [y]es / [n]o / [a]lways: ").strip().lower()
    except EOFError:
        return "n"
    return answer[:1] if answer else "n"


def default_asker() -> Asker | None:
    """A terminal asker only when someone is actually there to answer.

    Without a TTY there is no one to ask, so `ask` must resolve to deny — an unattended
    daily job should never hang on a prompt, and should never quietly gain more authority
    than an attended one.
    """
    return terminal_asker if sys.stdin.isatty() else None


class PermissionGate:
    """Evaluates policy, asks when needed, and remembers what it decided.

    Decisions are memoized per `call_id`, which is the chapter's invariant: *"deny is
    sticky for this tool_use_id — no silent retry to allow."*
    """

    def __init__(
        self,
        policy: PermissionPolicy,
        *,
        asker: Asker | None = None,
        auto_approve: bool = False,
    ):
        self.policy = policy
        self.asker = asker
        self.auto_approve = auto_approve
        self._decisions: dict[str, PermissionResult] = {}
        self._session_allow: set[str] = set()
        self._lock = asyncio.Lock()

    def decided(self, call_id: str) -> PermissionResult | None:
        return self._decisions.get(call_id)

    async def check(self, call_id: str, tool_name: str, arguments: dict) -> PermissionResult:
        remembered = self._decisions.get(call_id)
        if remembered is not None:
            return remembered

        result = await self._decide(tool_name, arguments)
        self._decisions[call_id] = result
        return result

    async def _decide(self, tool_name: str, arguments: dict) -> PermissionResult:
        result = self.policy.evaluate(tool_name)
        if result.decision != "ask":
            return result

        # --yes answers "ask". It is not a licence to run what policy forbids, so it is
        # applied after evaluation, never before it.
        if self.auto_approve:
            return PermissionResult("allow", "--yes was passed", rule=result.rule or "--yes")

        if tool_name in self._session_allow:
            return PermissionResult("allow", "approved earlier this session", rule=result.rule)

        if self.asker is None:
            return PermissionResult(
                "deny",
                "approval required but no one is available to ask "
                "(not an interactive terminal); pass --yes to run unattended",
                rule=result.rule,
            )

        # One prompt at a time, or two concurrent asks garble the terminal.
        async with self._lock:
            if tool_name in self._session_allow:  # granted while we waited for the lock
                return PermissionResult("allow", "approved earlier this session", rule=result.rule)
            answer = await asyncio.to_thread(self.asker, tool_name, arguments)

        if answer == "a":
            self._session_allow.add(tool_name)
            return PermissionResult("allow", "approved for the session", rule=result.rule)
        if answer == "y":
            return PermissionResult("allow", "approved by the operator", rule=result.rule)
        return PermissionResult("deny", "declined by the operator", rule=result.rule)
