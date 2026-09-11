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

Decision = Literal["allow", "deny", "ask", "confirm"]

#: Higher is more restrictive. When several rules match, the most restrictive wins.
#:
#: `confirm` is `ask` that `--yes` cannot answer: a decision reserved to a person. It sits
#: between ask and deny so the precedence law is unchanged — a narrow `allow` still cannot
#: punch through, `deny` still beats everything — and it never escapes this module, since
#: `_decide` resolves it to allow or deny before returning.
_RANK: dict[str, int] = {"allow": 0, "ask": 1, "confirm": 2, "deny": 3}


@dataclass(frozen=True)
class PermissionResult:
    """One authorization decision, with the rule that produced it.

    `rule` is kept so a decision is auditable after the fact — "why was this allowed?"
    should never require re-deriving the policy.
    """

    decision: Decision
    reason: str = ""
    rule: str | None = None
    #: True when the denial means "there was nobody to ask", not "a person said no". Only
    #: the second is a judgement worth trying to address; arguing with the first wastes a
    #: turn, and counting it as a refusal spends an attempt the agent never had.
    unattended: bool = False

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
    #: Asked like `ask`, but `--yes` may not answer it. For decisions that belong to a
    #: person even when nobody is watching.
    confirm: tuple[str, ...] = ()
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
            ("confirm", self.confirm),
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
            confirm=tuple(data.get("confirm", ())),
            deny=tuple(data.get("deny", ())),
            default=data.get("default", "ask"),
        )


@dataclass
class PlanState:
    """Whether this run may act yet, and what it has proposed.

    Mutable on purpose: the approval that lifts it arrives mid-turn, from a tool the model
    itself called. `ToolContext` is frozen and holds the reference, not the value, so one
    instance is shared by the gate that enforces the mode and the tool that ends it.
    """

    active: bool = False
    #: Plans a person has refused. Not incremented when there was nobody to ask — that is
    #: a fact about the environment, and spending an attempt on it would be spending one
    #: the agent never had.
    attempts: int = 0
    #: A plan was proposed with nobody available to approve it. The run has done what a
    #: planning run is for and should stop, rather than re-proposing into an empty room.
    pending: bool = False


# ---- asking a human ----------------------------------------------------------


@dataclass(frozen=True)
class PermissionRequest:
    """What the operator is being asked to authorize.

    Wider than a name and an arguments dict, because two things need more: a plan is prose
    and must not be shown as escaped JSON, and a `confirm` has no "always" — approving one
    plan is not approving the next one, and offering the option would be a lie.
    """

    tool_name: str
    arguments: dict
    kind: Decision = "ask"
    rule: str | None = None

    @property
    def one_shot(self) -> bool:
        return self.kind == "confirm"


#: Returns "y" (allow once), "n" (deny), or "a" (allow this tool for the session).
Asker = Callable[["PermissionRequest"], str]


def _render_arguments(arguments: dict) -> list[str]:
    """Readable, not round-trippable. Multi-line values print as themselves.

    `json.dumps` turns a plan into one long line of `\n` escapes, which is unreadable at
    exactly the moment someone is being asked to approve it.
    """
    if not arguments:
        return ["{}"]
    lines: list[str] = []
    for key, value in arguments.items():
        if isinstance(value, str) and "\n" in value:
            lines.append(f"{key}:")
            lines += [f"  {line}" for line in value.splitlines()]
        else:
            lines.append(f"{key}: {json.dumps(value, default=str)}")
    return lines


def terminal_asker(request: PermissionRequest) -> str:
    """Prompt on the terminal. Runs in a worker thread — never on the event loop."""
    label = "approval required" if request.one_shot else "permission required"
    print(f"\n  {label}: {request.tool_name}", file=sys.stderr)
    for line in _render_arguments(request.arguments):
        print(f"      {line}", file=sys.stderr)
    choices = "[y]es / [n]o" if request.one_shot else "[y]es / [n]o / [a]lways"
    try:
        answer = input(f"  allow? {choices}: ").strip().lower()
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
        plan: PlanState | None = None,
    ):
        self.policy = policy
        self.asker = asker
        self.auto_approve = auto_approve
        #: Shared with the tool that ends plan mode. None means plan mode is unavailable.
        self.plan = plan
        self._decisions: dict[str, PermissionResult] = {}
        self._session_allow: set[str] = set()
        self._lock = asyncio.Lock()

    def decided(self, call_id: str) -> PermissionResult | None:
        return self._decisions.get(call_id)

    async def check(
        self,
        call_id: str,
        tool_name: str,
        arguments: dict,
        *,
        read_only: bool = False,
    ) -> PermissionResult:
        """`read_only` defaults False because that is the fail-closed direction: a caller
        that forgets to say gets "treat it as a write", which plan mode refuses."""
        remembered = self._decisions.get(call_id)
        if remembered is not None:
            return remembered

        result = await self._decide(tool_name, arguments, read_only=read_only)
        self._decisions[call_id] = result
        return result

    async def _decide(
        self, tool_name: str, arguments: dict, *, read_only: bool
    ) -> PermissionResult:
        result = self.policy.evaluate(tool_name)
        if result.decision == "deny":
            return result  # an explicit deny wins, keeping its own rule and reason

        # Plan mode forbids acting alone. A `confirm` is by definition not alone — a person
        # reads it before it runs — so it is the one thing that may still be proposed. That
        # is what makes the plan submittable without the gate special-casing a tool name,
        # in a design whose whole premise is that policy decides.
        if self.plan is not None and self.plan.active and not read_only:
            if result.decision != "confirm":
                return PermissionResult(
                    "deny",
                    "plan mode: only read-only tools run until a plan is approved",
                    rule=result.rule,
                )

        if result.decision not in ("ask", "confirm"):
            return result
        one_shot = result.decision == "confirm"

        # --yes answers "ask". It is not a licence to run what policy forbids, so it is
        # applied after evaluation, never before it — and it does not answer a `confirm`,
        # which is reserved to a person by definition.
        if self.auto_approve and not one_shot:
            return PermissionResult("allow", "--yes was passed", rule=result.rule or "--yes")

        # A confirm is never remembered: approving one plan is not approving the next.
        if not one_shot and tool_name in self._session_allow:
            return PermissionResult("allow", "approved earlier this session", rule=result.rule)

        if self.asker is None:
            if one_shot and self.plan is not None:
                self.plan.pending = True
            return PermissionResult(
                "deny",
                "approval required but no one is available to ask "
                "(not an interactive terminal); pass --yes to run unattended",
                rule=result.rule,
                unattended=True,
            )

        request = PermissionRequest(
            tool_name=tool_name, arguments=arguments, kind=result.decision, rule=result.rule
        )
        # One prompt at a time, or two concurrent asks garble the terminal.
        async with self._lock:
            if not one_shot and tool_name in self._session_allow:  # granted while waiting
                return PermissionResult("allow", "approved earlier this session", rule=result.rule)
            answer = await asyncio.to_thread(self.asker, request)

        if answer == "a" and not one_shot:
            self._session_allow.add(tool_name)
            return PermissionResult("allow", "approved for the session", rule=result.rule)
        if answer in ("y", "a"):  # "a" on a confirm means yes, once
            return PermissionResult("allow", "approved by the operator", rule=result.rule)
        if one_shot and self.plan is not None:
            self.plan.attempts += 1
        return PermissionResult("deny", "declined by the operator", rule=result.rule)
