"""Child agents: one role, one session, one transcript, one outcome.

A single agent that researches, decides, writes and then checks its own work is the
arrangement the chapter warns about — *"'I changed code' and 'the change is correct' are
separated by a wide river, and models are good at building paper bridges over it."* We have
watched it here: a brief invented eight of nine hooks and the run reported success.

So the work splits into roles, each its own `AgentSession` over its own `Transcript`. Two
things follow, and only the first is obvious.

**Failures localise.** A dead ad library no longer poisons the brand research; the
coordinator sees one worker come back empty and says so in the brief.

**The cached prefix is shared, byte for byte.** A child gets the parent's prompt layers,
governance, MCP guidance and the *whole* tool list — including tools it is not allowed to
call. Removing a declaration would change the prefix, and declarations are roughly
three-quarters of it. Scoping is the permission gate's job, which is what actually stops a
call; an absent declaration never did. The role rides in the opening user turn instead of
the system prompt for the same reason.

The ledger discipline is lifted wholesale from `StreamingToolExecutor`: every spawned child
produces exactly one outcome, in spawn order, whether it finished, failed, timed out or was
cancelled. Parent dies, children die.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    AGENT_TIMEOUT_SECONDS,
    INTERRUPT_DRAIN_SECONDS,
    MAX_CHILDREN_PER_RUN,
    AGENT_DIR,
)
from .permissions import Asker, PermissionPolicy
from .session import AgentSession
from .tools import ToolRegistry

#: Where role descriptions live, one file per role.
ROLES_DIR = "roles"


def load_role(name: str, agent_dir: Path | None = None) -> str:
    """Read a role description. A missing one is an error, not an empty string —
    a worker with no job is a worker that will invent one."""
    path = (agent_dir or AGENT_DIR) / ROLES_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no role description: {path}")
    return path.read_text(encoding="utf-8")


def role_policy(name: str, agent_dir: Path | None = None) -> PermissionPolicy:
    """A role's policy, falling back to the session default when it has no file of its own."""
    directory = agent_dir or AGENT_DIR
    specific = directory / f"permissions.{name}.toml"
    return PermissionPolicy.load(specific if specific.is_file() else directory / "permissions.toml")


@dataclass(frozen=True)
class ChildOutcome:
    """One closed ledger entry. The shape a coordinator reads."""

    child_id: str
    role: str
    task: str
    #: What the child actually said. The coordinator's raw material — deliberately not a
    #: summary, because a worker that pre-compresses leaves the coordinator nothing to
    #: recompress, and recompression is the entire point of having one.
    text: str = ""
    stop_reason: str = ""
    is_error: bool = False
    #: failed | timeout | cancelled | not_started | spawn_error
    reason: str | None = None
    usage: dict = field(default_factory=dict)

    def render(self) -> str:
        """How this child appears to the coordinator."""
        head = f"### {self.role} — {self.child_id}\nTask: {self.task}"
        if self.is_error:
            return f"{head}\nFAILED ({self.reason}): {self.text or 'no output'}"
        return f"{head}\n\n{self.text}"


@dataclass
class AgentPool:
    """Runs children concurrently and guarantees each one closes.

    Holds no model state of its own. Everything a child needs — the client, the registry,
    the shared prompt inputs — is handed in, so two pools in one process share nothing by
    accident.
    """

    parent_id: str
    registry: ToolRegistry
    client: Any = None
    agent_dir: Path = AGENT_DIR
    runs_dir: Path | None = None
    model: str | None = None
    budget: int | None = None
    max_turns: int | None = None
    mcp_instructions: dict[str, str] = field(default_factory=dict)
    #: Shared so two children prompting at once cannot garble one terminal.
    asker: Asker | None = None
    timeout: float = AGENT_TIMEOUT_SECONDS
    drain_timeout: float = INTERRUPT_DRAIN_SECONDS
    max_children: int = MAX_CHILDREN_PER_RUN

    _running: dict[str, asyncio.Task] = field(default_factory=dict, init=False)
    _outcomes: dict[str, ChildOutcome] = field(default_factory=dict, init=False)
    _issued: list[str] = field(default_factory=list, init=False)
    _meta: dict[str, tuple[str, str]] = field(default_factory=dict, init=False)

    @property
    def spawned(self) -> int:
        return len(self._issued)

    def child_id(self, role: str) -> str:
        """Named for its parent, so the family is visible in `runs/` without an index."""
        nth = sum(1 for r, _ in self._meta.values() if r == role) + 1
        return f"{self.parent_id}.{role}-{nth}"

    # ---- spawning ------------------------------------------------------------

    def spawn(self, role: str, task: str) -> str:
        """Start a child and return immediately. Its outcome lands in the ledger."""
        if self.spawned >= self.max_children:
            raise ValueError(
                f"this run has already spawned {self.spawned} children "
                f"(cap is {self.max_children})"
            )
        child_id = self.child_id(role)
        self._issued.append(child_id)
        self._meta[child_id] = (role, task)
        self._running[child_id] = asyncio.create_task(self._run(child_id, role, task))
        return child_id

    async def _run(self, child_id: str, role: str, task: str) -> None:
        try:
            session = AgentSession.create(
                child_id,
                runs_dir=self.runs_dir,
                registry=self.registry,
                role=load_role(role, self.agent_dir),
                policy=role_policy(role, self.agent_dir),
                client=self.client,
                agent_dir=self.agent_dir,
                mcp_instructions=dict(self.mcp_instructions),
                asker=self.asker,
                budget=self.budget,
                root_meta={"parent_session": self.parent_id, "role": role},
                **({"model": self.model} if self.model else {}),
                **({"max_turns": self.max_turns} if self.max_turns else {}),
            )
            result = await asyncio.wait_for(
                session.submit(task, on_text=None, on_event=None), self.timeout
            )
            self._close(
                ChildOutcome(
                    child_id=child_id,
                    role=role,
                    task=task,
                    text=result.text,
                    stop_reason=result.stop_reason,
                    is_error=result.stop_reason != "end_turn",
                    reason=None if result.stop_reason == "end_turn" else result.stop_reason,
                    usage=result.usage or {},
                )
            )
        except asyncio.TimeoutError:
            self._close(self._failed(child_id, role, task, "timeout",
                                     f"no answer within {self.timeout:g}s"))
        except asyncio.CancelledError:
            # Not re-raised: cancellation still owes the ledger an entry.
            self._close(self._failed(child_id, role, task, "cancelled",
                                     "cancelled before it finished"))
        except Exception as exc:  # noqa: BLE001 - a child's failure is data, not a crash
            self._close(self._failed(child_id, role, task, "spawn_error",
                                     f"{type(exc).__name__}: {exc}"))
        finally:
            self._running.pop(child_id, None)

    @staticmethod
    def _failed(child_id: str, role: str, task: str, reason: str, text: str) -> ChildOutcome:
        return ChildOutcome(
            child_id=child_id, role=role, task=task, text=text,
            stop_reason=reason, is_error=True, reason=reason,
        )

    def _close(self, outcome: ChildOutcome) -> None:
        """First writer wins, so a cancel sweep cannot clobber a real result."""
        self._outcomes.setdefault(outcome.child_id, outcome)

    # ---- completion ----------------------------------------------------------

    async def wait_for(self, child_id: str) -> ChildOutcome:
        """Await one child. Concurrency comes from the caller issuing several spawns in
        one turn — the executor runs those tool calls in parallel, so the children do too."""
        task = self._running.get(child_id)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        outcome = self._outcomes.get(child_id)
        if outcome is None:  # a task cancelled before its body ran owes an entry anyway
            role, requested = self._meta.get(child_id, ("?", "?"))
            outcome = self._failed(child_id, role, requested, "cancelled", "never started")
            self._close(outcome)
        return outcome

    async def drain(self) -> list[ChildOutcome]:
        """Wait for every spawned child, then return outcomes in spawn order."""
        if self._running:
            await asyncio.gather(*list(self._running.values()), return_exceptions=True)
        return self.ledger()

    def ledger(self) -> list[ChildOutcome]:
        missing = [cid for cid in self._issued if cid not in self._outcomes]
        if missing:
            raise RuntimeError(f"ledger not closed for children: {missing}")
        return [self._outcomes[cid] for cid in self._issued]

    async def cancel(self) -> list[ChildOutcome]:
        """Parent dies, children die — and the ledger still closes.

        Bounded, for the same reason the executor's is: a child that will not wind down
        must not make the interrupt unkillable.
        """
        for task in list(self._running.values()):
            task.cancel()
        if self._running:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*list(self._running.values()), return_exceptions=True),
                    self.drain_timeout,
                )
            except asyncio.TimeoutError:
                pass
        self._running.clear()
        return self.close_all()

    def close_all(self, reason: str = "cancelled") -> list[ChildOutcome]:
        """Close whatever is still open, awaiting nothing. Cannot hang, cannot raise."""
        for child_id in self._issued:
            role, task = self._meta.get(child_id, ("?", "?"))
            self._close(
                self._failed(child_id, role, task, reason, "did not finish before the interrupt")
            )
        return [self._outcomes[cid] for cid in self._issued]
