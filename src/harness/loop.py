"""The query loop — the heartbeat.

Every iteration rebuilds the entire runtime context from state: system instruction, tool
declarations, and the conversation derived by walking the transcript from its head back to
the root. Nothing is carried implicitly between turns and nothing is stored server-side, so
the loop can be pointed at any node in the tree and will simply build a different request.

Termination is deliberate and enumerated. There is no retry or recovery policy yet, so the
loop fails loudly rather than guessing.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from .compaction import is_prompt_too_long, plan_cut, summarize
from .config import (
    MAX_CONSECUTIVE_COMPACT_FAILURES,
    MIN_STEPS_TO_COMPACT,
    MAX_PLAN_ATTEMPTS,
    MAX_TURNS,
    MODEL,
    RUNS_DIR,
    approx_tokens,
    bytes_per_token_for,
    compact_threshold,
)
from .events import (
    StreamDone,
    StreamError,
    TextDelta,
    ThoughtDelta,
    ThoughtReady,
    ToolCallReady,
    ToolCallStarted,
    normalize,
)
from .budget import ContentReplacementState
from .budget import apply as apply_budget
from .budget import reapply as reapply_budget
from .executor import StreamingToolExecutor
from .microcompact import MicrocompactState, maybe_microcompact, reapply
from .permissions import PermissionGate
from .prompt import AssembledPrompt
from .session_memory import MemoryGate, SessionMemory, Writer
from .attachments import from_lineage, render_boundary
from .tools import ToolContext, ToolRegistry
from .transcript import Transcript


def estimate_step(step: dict) -> int:
    """Rough token cost of one step, by content type.

    Tool results are usually JSON and bill at about two chars per token, where prose runs
    near four; measuring them at four halves the answer for exactly the steps that weigh
    most. `thought` steps are counted by their signature, which is a large base64 blob
    that is replayed verbatim and is not free.
    """
    kind = step.get("type")
    if kind == "function_result":
        body = " ".join(
            block.get("text", "")
            for block in step.get("result") or []
            if isinstance(block, dict)
        )
        return approx_tokens(body, bytes_per_token_for(body))
    if kind in ("user_input", "model_output"):
        body = " ".join(
            block.get("text", "")
            for block in step.get("content") or []
            if isinstance(block, dict)
        )
        return approx_tokens(body)
    if kind == "thought":
        return approx_tokens(str(step.get("signature") or ""))
    if kind == "function_call":
        return approx_tokens(json.dumps(step.get("arguments") or {}), 2)
    return 0


@dataclass(frozen=True)
class RuntimeContext:
    """Everything one model call needs. Rebuilt from scratch each iteration."""

    model: str
    system_instruction: str
    input: list[dict]
    tools: list[dict]
    store: bool = False

    def request(self) -> dict:
        return {
            "model": self.model,
            "system_instruction": self.system_instruction,
            "input": self.input,
            "tools": self.tools,
            "store": self.store,
            "stream": True,
        }


@dataclass
class LoopState:
    """What changes across turns. The transcript is the history; nothing shadows it."""

    transcript: Transcript

    #: The conversation the model actually sees, held in memory.
    #:
    #: The tree is the durable record; this is its projection. Re-deriving it by walking
    #: the tree on every turn would work — it is what we did first — but it leaves the
    #: question "when is the context rebuilt?" with the answer "constantly, implicitly".
    #: Projecting once and appending makes the rebuild points explicit, and there are
    #: exactly four: loading a session, branching, compacting, and microcompacting.
    #:
    #: Microcompaction is the odd one out: the other three re-derive the projection *from*
    #: the tree, while it rewrites the projection so it deliberately says less than the
    #: tree does. That is the point — "what the model actually sees" is what this field
    #: means, and after stale results are cleared the model genuinely sees less. The tree
    #: still holds all of it, which is why `verify` reads the tree and not this.
    #:
    #: Every append goes through `record()` so the two cannot drift.
    messages: list[dict] = field(default_factory=list)

    turn: int = 0
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    text: str = ""

    #: Current context size, read from the last response rather than counted — free
    #: and exact, where a count_tokens round trip would be neither.
    context_tokens: int = 0
    #: (turn, input tokens, cached tokens) per turn. The provider already reports both and
    #: nothing read `total_cached_tokens` before this, which left the cache unobservable
    #: from inside a run. Microcompaction's whole cost argument is a claim about this
    #: series — cached tokens collapse on the turn a clearing fires and recover on the
    #: next — so it needs to be visible rather than argued about.
    cache_history: list[tuple[int, int, int]] = field(default_factory=list)
    #: Consecutive compaction failures. Three and we stop trying.
    compact_failures: int = 0
    compactions: int = 0
    #: Context size at the last compaction. Compacting again before the context has grown
    #: past it cannot help — if the prefix alone exceeds the budget, no amount of
    #: summarizing gets under it, and retrying every turn is a runaway.
    last_compact_tokens: int = 0
    memory_gate: MemoryGate = field(default_factory=MemoryGate)
    session_memory: SessionMemory | None = None

    #: Which tool results microcompaction has cleared. Shared by reference with the
    #: `Session` that owns it, the way `PlanState` is shared with the permission gate — a
    #: fresh `LoopState` is built per submission, so a set held here alone would forget
    #: everything between submissions and restore results the last one cleared.
    micro: MicrocompactState = field(default_factory=MicrocompactState)

    #: Which oversized results have been written to disk and what the model was shown in
    #: their place. Shared with the `Session` for the same reason as `micro`: once the
    #: model has seen a result, that decision must outlive the submission that made it.
    budget: ContentReplacementState = field(default_factory=ContentReplacementState)

    #: How many steps went into the last request. Everything after this index was added
    #: since, so its size has to be estimated rather than read off the response — see
    #: `note_usage`.
    sent_through: int = 0


    # ---- the projection ------------------------------------------------------

    def project(self) -> None:
        """Rebuild the working context from the tree.

        Called at the three points where the tree's shape changes under us: opening a
        session, branching to a different node, and compacting. Microcompaction is the
        fourth rebuild point but does not come through here — it rewrites the projection
        rather than re-deriving it.

        The tree still holds every tool result in full, so a plain walk would undo both
        the size gate and any clearing microcompaction has done. Re-applying keeps the
        rebuild faithful to what the model was last shown.
        """
        self.messages = reapply(reapply_budget(self.transcript.steps(), self.budget), self.micro)

    def note_usage(self) -> None:
        """Work out how full the context actually is, after a response.

        An exact baseline plus an estimate of the rest. `total_input_tokens` is exact but
        describes the request we *sent*; everything recorded since — the model's own
        output, and above all this turn's tool results — is already in the context and
        will be in the next request, so it has to be counted too.

        Reading the response alone was the bug this replaces. `note_usage` runs after the
        turn's results are recorded, so the count excluded the single largest thing in the
        context and every threshold built on it fired late.

        On Gemini `total_cached_tokens` is the cached *part* of `total_input_tokens`, not
        a separate bucket, so the two are never added — that would double-count the cache.
        """
        sent = self.usage.get("total_input_tokens", self.context_tokens)
        pending = self.messages[self.sent_through :]
        self.context_tokens = sent + sum(estimate_step(step) for step in pending)
        self.cache_history.append(
            (
                self.turn,
                self.usage.get("total_input_tokens", 0),
                self.usage.get("total_cached_tokens", 0),
            )
        )

    def record(self, step: dict, *, turn: int = 0, meta: dict | None = None):
        """Append to the durable tree and the in-memory projection together.

        The single write path. Anything appending to one and not the other is a bug that
        surfaces as the model seeing a different conversation than the transcript records.
        """
        node = self.transcript.append(step, turn=turn, meta=meta)
        self.messages.append(step)
        return node


@dataclass(frozen=True)
class LoopResult:
    stop_reason: str
    turns: int
    text: str
    usage: dict
    error: str | None = None


def build_runtime(
    state: LoopState,
    prompt: AssembledPrompt,
    registry: ToolRegistry,
    model: str = MODEL,
) -> RuntimeContext:
    """Assemble the full runtime context for one turn."""
    return RuntimeContext(
        model=model,
        system_instruction=prompt.system_instruction,
        input=list(state.messages),
        tools=registry.declarations(),
        store=False,
    )


async def query_loop(
    state: LoopState,
    *,
    client: Any,
    prompt: AssembledPrompt,
    registry: ToolRegistry,
    model: str = MODEL,
    max_turns: int = MAX_TURNS,
    ctx: ToolContext | None = None,
    gate: PermissionGate | None = None,
    writer: Writer | None = None,
    budget: int | None = None,
    on_text: Callable[[str], None] | None = None,
    memory_dir: Path | None = None,
    on_event: Callable[[Any], None] | None = None,
) -> LoopResult:
    """Run until the model stops asking for tools, or a termination condition fires.

    `client` needs one thing: an awaitable `aio.interactions.create(**request)` returning an
    async iterator of SSE events. Tests supply a fake.
    """
    last_text = ""
    ptl_recovered = False  # one compaction-and-retry per run, never a loop

    while True:
        if state.turn >= max_turns:
            state.stop_reason = "max_turns"
            return LoopResult("max_turns", state.turn, last_text, state.usage)

        state.turn += 1
        # Both of these run before the request is assembled — the whole point is to shrink
        # what this turn sends — and both rewrite only the projection, never the tree.
        #
        # The size gate goes first: it decides what a *new* result may occupy, while
        # microcompaction clears results that have gone stale, which means long since
        # sent. They cannot fight over the same step, because a result the gate may still
        # act on is by definition one the model has never seen.
        nodes = state.transcript.path_to_root()
        state.messages = apply_budget(
            state.messages,
            nodes,
            state.budget,
            registry=registry,
            runs_dir=state.transcript.path.parent
            if state.transcript.path is not None
            else RUNS_DIR,
            session_id=state.transcript.session_id,
        )
        # Re-assert what microcompaction has already cleared. The size gate re-applies its
        # replacements verbatim every turn, which would otherwise put a preview back over a
        # result that was cleared for age — undoing the clearing, and re-counting the
        # preview's bytes as reclaimed on every turn after. Stale beats persisted.
        state.messages = reapply(state.messages, state.micro)

        cleared = maybe_microcompact(state.messages, nodes, state.micro)
        if cleared is not None:
            state.messages = cleared

        runtime = build_runtime(state, prompt, registry, model)
        # Watermark for `note_usage`: everything appended past here arrived after the
        # request went out, so the response's token count cannot account for it.
        state.sent_through = len(runtime.input)
        # A fresh executor per turn: a turn's ledger must not leak into the next one.
        turn_ctx = replace(ctx or ToolContext(session_id="local"), turn=state.turn)
        executor = StreamingToolExecutor(registry, ctx=turn_ctx, gate=gate)

        text_buffer: list[str] = []
        stream_error: str | None = None

        def flush_text() -> None:
            """Write buffered assistant text into the tree before any call that follows."""
            if not text_buffer:
                return
            joined = "".join(text_buffer)
            state.record(
                {"type": "model_output", "content": [{"type": "text", "text": joined}]},
                turn=state.turn,
            )
            text_buffer.clear()

        try:
            stream = await client.aio.interactions.create(**runtime.request())

            async for event in normalize(stream):
                if on_event is not None:
                    on_event(event)

                if isinstance(event, TextDelta):
                    text_buffer.append(event.text)
                    last_text += event.text
                    if on_text is not None:
                        on_text(event.text)

                elif isinstance(event, ThoughtDelta):
                    pass  # display only; the replayable form is ThoughtReady

                elif isinstance(event, ThoughtReady):
                    # Gemini 3 signs its thought steps and rejects the next request if the
                    # signature is not replayed with the history. We carry it verbatim.
                    flush_text()
                    thought_step: dict = {"type": "thought"}
                    if event.signature:
                        thought_step["signature"] = event.signature
                    if event.summary:
                        thought_step["summary"] = event.summary
                    state.record(thought_step, turn=state.turn)

                elif isinstance(event, ToolCallStarted):
                    pass  # arguments still streaming — nothing to do yet

                elif isinstance(event, ToolCallReady):
                    # Dispatch mid-stream. This is the whole point of the executor.
                    flush_text()
                    state.record(
                        {
                            "type": "function_call",
                            "id": event.call_id,
                            "name": event.name,
                            "arguments": event.arguments,
                        },
                        turn=state.turn,
                    )
                    executor.submit(event)
                    # Yield to the event loop so the task submit() just created can
                    # actually begin. Without this, buffered SSE events are iterated
                    # without ever suspending, and no tool body starts until the stream
                    # ends — which would make mid-stream dispatch purely notional.
                    await asyncio.sleep(0)

                elif isinstance(event, StreamError):
                    stream_error = event.message

                elif isinstance(event, StreamDone):
                    if event.usage:
                        state.usage = event.usage

        except (asyncio.CancelledError, KeyboardInterrupt):
            await _close_ledger(state, executor)
            state.stop_reason = "interrupted"
            return LoopResult("interrupted", state.turn, last_text, state.usage)

        except Exception as exc:  # transport, auth, malformed request
            await _close_ledger(state, executor)

            # The request outgrew the window. Compaction is the recovery, not a retry
            # loop: one attempt, then the turn fails honestly.
            if is_prompt_too_long(exc) and not ptl_recovered:
                if await maybe_compact(
                    state, client, model, budget, forced=True, memory_dir=memory_dir
                ):
                    ptl_recovered = True
                    state.turn -= 1  # the turn never ran; do not spend it
                    continue

            await _brief_before_leaving(state, client, model, writer)
            state.stop_reason = "api_error"
            return LoopResult(
                "api_error", state.turn, last_text, state.usage, f"{type(exc).__name__}: {exc}"
            )

        flush_text()

        if stream_error is not None:
            # API errors return directly. No retry policy exists yet, and inventing one
            # silently is worse than stopping where the failure happened.
            await _close_ledger(state, executor)
            await _brief_before_leaving(state, client, model, writer)
            state.stop_reason = "api_error"
            return LoopResult("api_error", state.turn, last_text, state.usage, stream_error)

        if executor.issued == 0:
            # A turn may not end while children are still working. The alternative is to
            # evict them, but waiting is the better trade: the work is already paid for,
            # and a coordinator that forgets to collect would otherwise silently throw
            # away three researchers.
            #
            # This is the one place the loop records a `user_input` step mid-run — every
            # other `record` writes model output, a thought, or a tool call. It still goes
            # through `record`, so the tree and the projection cannot drift.
            harvest = await _collect_children(ctx)
            if harvest is not None:
                state.record(harvest, turn=state.turn)
                last_text = ""
                continue

            state.note_usage()
            await maybe_write_memory(state, client, model, writer, at_stopping_point=True)
            state.stop_reason = "end_turn"
            return LoopResult("end_turn", state.turn, last_text, state.usage)

        try:
            outcomes = await executor.drain()
        except (asyncio.CancelledError, KeyboardInterrupt):
            await _close_ledger(state, executor)
            state.stop_reason = "interrupted"
            return LoopResult("interrupted", state.turn, last_text, state.usage)

        for outcome in outcomes:
            state.record(outcome.to_step(), turn=state.turn)

        # Plan mode has two endings of its own, both deliberate and both enumerated here
        # with the rest. Neither is a failure of the loop: one is a plan waiting for a
        # person, the other a plan a person has now refused twice.
        plan = getattr(gate, "plan", None)
        if plan is not None:
            if plan.pending:
                await _brief_before_leaving(state, client, model, writer)
                state.stop_reason = "plan_pending"
                return LoopResult("plan_pending", state.turn, last_text, state.usage)
            if plan.attempts >= MAX_PLAN_ATTEMPTS:
                state.stop_reason = "plan_refused"
                return LoopResult("plan_refused", state.turn, last_text, state.usage)

        state.note_usage()
        state.memory_gate.observe_tool_calls(
            len(outcomes), sum(1 for o in outcomes if o.is_error)
        )
        # Mid-tool-chain is not a coherent moment to write notes, so the gate may defer.
        await maybe_write_memory(state, client, model, writer, at_stopping_point=False)
        await maybe_compact(state, client, model, budget, memory_dir=memory_dir)


async def maybe_write_memory(
    state: LoopState,
    client: Any,
    model: str,
    writer: Writer | None,
    *,
    at_stopping_point: bool,
) -> bool:
    """Write the continuation brief if the gate says this is the moment."""
    if writer is None:
        return False
    decision = state.memory_gate.decide(state.context_tokens, at_stopping_point)
    if decision is None:
        return False
    try:
        state.session_memory = await writer(list(state.messages), state.session_memory)
    except Exception:  # noqa: BLE001 - a brief we could not write must not end the run
        return False
    state.session_memory.save(state.transcript.session_id)
    state.memory_gate.record_write(state.context_tokens)
    return True


async def maybe_compact(
    state: LoopState,
    client: Any,
    model: str,
    budget: int | None = None,
    *,
    forced: bool = False,
    memory_dir: Path | None = None,
) -> bool:
    """Summarize the old history and rebuild the working context from it.

    Returns True if a boundary was written. The circuit breaker exists because failing is
    survivable but failing forever without memory is not.
    """
    if state.compact_failures >= MAX_CONSECUTIVE_COMPACT_FAILURES:
        return False
    if not forced and state.context_tokens < compact_threshold(budget):
        return False
    if not forced and state.compactions and state.context_tokens <= state.last_compact_tokens:
        # Still over the threshold but no larger than last time: the irreducible part of
        # the context is simply bigger than the budget. Compacting again cannot fix that.
        return False

    nodes = state.transcript.path_to_root()
    if len(nodes) < MIN_STEPS_TO_COMPACT:
        return False  # not enough history to be worth a summarization call
    cut = plan_cut(nodes)
    if cut <= 0:
        return False  # nothing old enough to reclaim

    try:
        brief = await summarize(
            client, model, [n.step for n in nodes[:cut]], state.session_memory
        )
    except Exception:  # noqa: BLE001 - counted, not raised; three strikes stops it
        state.compact_failures += 1
        return False

    state.compact_failures = 0
    state.session_memory = brief
    brief.save(state.transcript.session_id)
    # Attachments are derived from the tree rather than held on the side, so this reads
    # the lineage that is about to be replaced — including whatever earlier boundaries
    # already replaced — and re-attaches it alongside the brief.
    lineage = state.transcript.lineage()
    state.transcript.compact_boundary(
        render_boundary(lineage, brief.render(), memory_dir=memory_dir),
        nodes[cut:],
        meta={
            "pre_compact_tokens": state.context_tokens,
            "steps_compacted": cut,
            "attachments": [i.name for i in from_lineage(lineage).items],
        },
    )
    state.project()  # the tree's shape changed under us
    state.compactions += 1
    state.last_compact_tokens = state.context_tokens
    state.memory_gate.record_write(state.context_tokens)
    return True


async def _brief_before_leaving(state, client, model, writer) -> None:
    """Write the continuation brief on an exit the user did not ask for.

    `maybe_write_memory` sits at the tail of a tool-using turn, so every early return used
    to skip it and a run that died at turn 15 resumed from a brief many turns stale.

    Deliberately *not* called on `interrupted`. Someone pressed Ctrl-C; making them wait
    for a summarization round trip is the opposite of what they asked for, and the
    transcript already carries everything a resume needs — attachments are derived from it.
    """
    try:
        await maybe_write_memory(state, client, model, writer, at_stopping_point=True)
    except Exception:  # noqa: BLE001 - a brief is a convenience; it must not mask the exit
        pass


async def _collect_children(ctx: ToolContext | None) -> dict | None:
    """Wait for outstanding children and turn their answers into the next user turn.

    Returns None when there is nothing to wait for, which is every run that never
    delegated — so the ordinary path costs one attribute lookup.
    """
    pool = getattr(ctx, "pool", None)
    if pool is None or not pool.spawned:
        return None
    # "Anything undelivered?", not "anything still running?". Once the parent can outlive
    # the spawn, children finish while it is busy elsewhere — and a run that asked only
    # whether tasks were live silently skipped the results, so the coordinator sat there
    # announcing it was waiting for reports it already had.
    if all(child_id in pool.reported for child_id in pool.issued):
        return None
    outcomes = await pool.drain()
    unreported = [o for o in outcomes if o.child_id not in pool.reported]
    pool.reported.update(o.child_id for o in unreported)
    if not unreported:
        return None
    body = "\n\n---\n\n".join(o.render() for o in unreported)
    return {
        "type": "user_input",
        "content": [
            {
                "type": "text",
                "text": (
                    f"{len(unreported)} agent(s) you started have finished. Their findings "
                    "follow in full. Work out what matters across them and carry on.\n\n"
                    + body
                ),
            }
        ],
    }


async def _close_ledger(state: LoopState, executor: StreamingToolExecutor) -> None:
    """Write a result for every issued call, however the turn ended.

    `BaseException`, not `Exception`, because `CancelledError` is not an `Exception` in
    3.8+ — a second interrupt arriving *during* closure used to abandon the ledger
    half-written and escape the loop entirely. And a failure here used to discard every
    outcome including the ones that had closed cleanly, so the fallback keeps those: a
    second Ctrl-C should cost the remainder, not the lot.
    """
    if executor.issued == 0:
        return
    try:
        outcomes = await executor.cancel()
    except BaseException:  # noqa: BLE001 - a failed cancel must not mask the original exit
        outcomes = executor.close_all()
    for outcome in outcomes:
        state.record(outcome.to_step(), turn=state.turn)
