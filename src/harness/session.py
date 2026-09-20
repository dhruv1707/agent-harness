"""The agent session: what survives a turn.

The boundary matters. The session owns durable things — the transcript tree, the tool
registry, the model choice, the client. The loop owns what happens *within* a turn and
stays a plain function over explicit state, `query_loop(state, ...)`, never a method here.
That is what stops this class from quietly becoming the runtime.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compaction import summarize
from .config import AGENT_DIR, MAX_TURNS, MODEL, RUNS_DIR
from .loop import LoopResult, LoopState, query_loop
from .microcompact import MicrocompactState
from .permissions import Asker, PermissionGate, PermissionPolicy, PlanState, default_asker
from .session_memory import SessionMemory
from .prompt import AssembledPrompt, RunContext, build_effective_system_prompt
from .tools import ToolContext, ToolRegistry, default_registry
from .transcript import Transcript


def make_client(api_key: str | None = None) -> Any:
    """Build a Gemini client. Credentials come from .env via config import."""
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("no GEMINI_API_KEY — put it in .env or export it")
    return genai.Client(api_key=key)


@dataclass
class AgentSession:
    transcript: Transcript
    registry: ToolRegistry = field(default_factory=default_registry)
    model: str = MODEL
    agent_dir: Path = AGENT_DIR
    client: Any = None
    max_turns: int = MAX_TURNS
    policy_path: Path | None = None
    auto_approve: bool = False
    #: Context budget in tokens. None uses the configured default.
    budget: int | None = None
    #: Server-published guidance, keyed by server name. Set by whoever opened the bridge.
    mcp_instructions: dict[str, str] = field(default_factory=dict)
    #: None means "pick a terminal asker if someone is there to answer".
    asker: Asker | None = None
    #: Start in plan mode: research freely, change nothing, until a plan is approved.
    plan: bool = False
    #: This session's job, when it is a worker rather than the whole run.
    #:
    #: Deliberately *not* the cacheable `agent` prompt layer. That layer sits above
    #: governance and the MCP guidance, so using it would push both below the role and
    #: re-bill them once per role. Riding in the opening user turn instead leaves every
    #: child's cached prefix byte-identical to its parent's, which is the fork's first
    #: invariant. A role description is a few hundred tokens; the prefix is fifteen
    #: thousand.
    role: str | None = None
    #: Built policy, for a worker that must run narrower than the file on disk. Takes
    #: precedence over `policy_path`.
    policy: PermissionPolicy | None = None
    #: Stamped on this session's opening node. A child records its parent here, so the
    #: family is readable from the tree itself rather than from a second index that can
    #: fall out of step — the same mechanism compaction uses for `compacted_from`.
    root_meta: dict = field(default_factory=dict)
    #: Set when this run may delegate. `spawn_agent` reaches it through ToolContext and
    #: nowhere else, so a session that was not given one simply cannot spawn.
    pool: Any = None

    #: The gate enforces plan mode and `submit_plan` ends it, so both must hold the same
    #: object. Built here, once, for exactly that reason.
    _plan_state: PlanState = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    #: Held here rather than on the LoopState for the same reason: a fresh LoopState is
    #: built per submission, so a cleared set living there would be forgotten between
    #: submissions and the next turn would restore every result the last one cleared.
    _micro: MicrocompactState = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._plan_state = PlanState(active=self.plan)
        self._micro = MicrocompactState()

    _gate: PermissionGate | None = field(default=None, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)

    @property
    def session_id(self) -> str:
        return self.transcript.session_id

    @property
    def microcompaction(self) -> MicrocompactState:
        """What microcompaction has cleared this session, for reporting."""
        return self._micro

    @property
    def gate(self) -> PermissionGate:
        """Built once per session, so an 'always allow' answer survives later turns."""
        if self._gate is None:
            policy = self.policy
            if policy is None:
                path = self.policy_path or (self.agent_dir / "permissions.toml")
                policy = PermissionPolicy.load(path)
            self._gate = PermissionGate(
                policy,
                asker=self.asker if self.asker is not None else default_asker(),
                auto_approve=self.auto_approve,
                plan=self._plan_state,
            )
        return self._gate

    def build_context(self) -> ToolContext:
        """The ambient state tools run against. The loop stamps the turn number on it."""
        return ToolContext(
            session_id=self.session_id,
            agent_dir=self.agent_dir,
            # The transcript's own directory, not the global one. A session pointed at a
            # different runs dir had its tools writing to the default anyway, so per-child
            # isolation was not actually isolating.
            runs_dir=self.transcript.path.parent,
            plan=self._plan_state,
            pool=self.pool,
        )

    # ---- lifecycle -----------------------------------------------------------

    @classmethod
    def create(
        cls,
        session_id: str | None = None,
        *,
        runs_dir: Path | None = None,
        **kwargs,
    ) -> AgentSession:
        return cls(transcript=Transcript.create(session_id, runs_dir=runs_dir), **kwargs)

    @classmethod
    def resume(
        cls,
        session_id: str,
        *,
        from_node: str | None = None,
        runs_dir: Path | None = None,
        **kwargs,
    ) -> AgentSession:
        """Reopen a session. With `from_node`, the next turn branches from that node."""
        transcript = Transcript.load(session_id, runs_dir=runs_dir or RUNS_DIR)
        if from_node is not None:
            transcript.branch_from(from_node)
        return cls(transcript=transcript, **kwargs)

    # ---- one turn ------------------------------------------------------------

    def build_prompt(self, run_context: RunContext | None = None) -> AssembledPrompt:
        """Rebuild the control plane. Run context is fresh every turn, by design."""
        return build_effective_system_prompt(
            agent_dir=self.agent_dir,
            mcp_instructions=self.mcp_instructions,
            run_context=run_context
            or RunContext(
                run_id=self.session_id,
                sources=tuple(sorted(t.name for t in self.registry)),
                max_turns=self.max_turns,
                plan_mode=self._plan_state.active,
            ),
        )

    async def submit(
        self,
        message: str,
        *,
        run_context: RunContext | None = None,
        on_text: Callable[[str], None] | None = None,
        on_event: Callable[[Any], None] | None = None,
    ) -> LoopResult:
        """Open a turn: record the user step, then run the loop until it terminates.

        One turn at a time. New input cannot interleave with a turn already in flight —
        that is the session half of chapter 4's interrupt semantics. The executor half
        already holds: `cancel()` awaits any `interrupt_behavior="block"` tool before the
        loop returns, so nothing is still writing when this method exits.
        """
        if self._active:
            raise RuntimeError(
                f"session {self.session_id} is already running a turn; "
                "wait for it to finish before submitting another"
            )

        prompt = self.build_prompt(run_context)
        opening = prompt.initial_input(message)

        if self.role:
            # Ahead of the run context and the task, because it frames how both are read.
            opening["content"].insert(
                0, {"type": "text", "text": "Your job on this run:\n\n" + self.role.strip()}
            )

        carried = SessionMemory.load(self.session_id)
        if carried is not None:
            # Resuming: lead with where things stand, so the model does not have to
            # re-derive it from a long history it may no longer fully hold.
            opening["content"].insert(
                0,
                {
                    "type": "text",
                    "text": "Continuation brief from earlier in this session:\n\n"
                    + carried.render(),
                },
            )
        self.transcript.append(opening, turn=0, meta=self.root_meta or None)

        state = LoopState(transcript=self.transcript, micro=self._micro)
        # Opening a session — or resuming, or branching — is a rebuild point: walk the
        # tree once here, then append for the rest of the run.
        state.project()
        if self.client is None:
            self.client = make_client()

        # A resumed session picks its brief back up, so the gate updates it rather than
        # starting over.
        if carried is not None:
            state.session_memory = carried
            state.memory_gate.exists = True

        # The brief writer and the compaction summarizer produce the same artifact from
        # the same input, so they are the same function.
        async def writer(steps, previous):
            return await summarize(self.client, self.model, steps, previous)

        self._active = True
        try:
            return await query_loop(
                state,
                client=self.client,
                prompt=prompt,
                registry=self.registry,
                model=self.model,
                max_turns=self.max_turns,
                ctx=self.build_context(),
                gate=self.gate,
                writer=writer,
                budget=self.budget,
                memory_dir=self.agent_dir / "memory",
                on_text=on_text,
                on_event=on_event,
            )
        finally:
            self._active = False
