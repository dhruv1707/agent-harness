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
from .permissions import Asker, PermissionGate, PermissionPolicy, default_asker
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
    #: None means "pick a terminal asker if someone is there to answer".
    asker: Asker | None = None

    _gate: PermissionGate | None = field(default=None, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)

    @property
    def session_id(self) -> str:
        return self.transcript.session_id

    @property
    def gate(self) -> PermissionGate:
        """Built once per session, so an 'always allow' answer survives later turns."""
        if self._gate is None:
            path = self.policy_path or (self.agent_dir / "permissions.toml")
            self._gate = PermissionGate(
                PermissionPolicy.load(path),
                asker=self.asker if self.asker is not None else default_asker(),
                auto_approve=self.auto_approve,
            )
        return self._gate

    def build_context(self) -> ToolContext:
        """The ambient state tools run against. The loop stamps the turn number on it."""
        return ToolContext(
            session_id=self.session_id, agent_dir=self.agent_dir, runs_dir=RUNS_DIR
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
            run_context=run_context
            or RunContext(
                run_id=self.session_id,
                sources=tuple(sorted(t.name for t in self.registry)),
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
        self.transcript.append(opening, turn=0)

        state = LoopState(transcript=self.transcript)
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
                on_text=on_text,
                on_event=on_event,
            )
        finally:
            self._active = False
