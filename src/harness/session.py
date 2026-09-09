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

from .config import AGENT_DIR, MAX_TURNS, MODEL, RUNS_DIR
from .loop import LoopResult, LoopState, query_loop
from .prompt import AssembledPrompt, RunContext, build_effective_system_prompt
from .tools import ToolRegistry, default_registry
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

    @property
    def session_id(self) -> str:
        return self.transcript.session_id

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
        """Open a turn: record the user step, then run the loop until it terminates."""
        prompt = self.build_prompt(run_context)
        self.transcript.append(prompt.initial_input(message), turn=0)

        state = LoopState(transcript=self.transcript)
        if self.client is None:
            self.client = make_client()

        return await query_loop(
            state,
            client=self.client,
            prompt=prompt,
            registry=self.registry,
            model=self.model,
            max_turns=self.max_turns,
            on_text=on_text,
            on_event=on_event,
        )
