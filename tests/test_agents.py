"""Child agents: isolation, the shared prefix, and a ledger that always closes.

The invariants, as assertions: a fork does not break the prompt cache, sharing is
explicit, parent dies child dies, and every spawn eventually accounts for itself.
"""

import asyncio
import json

import pytest

from harness.agents import AgentPool, ChildOutcome, load_role, role_policy
from harness.config import AGENT_DIR, MAX_AGENT_DEPTH
from harness.permissions import PermissionPolicy
from harness.prompt import build_effective_system_prompt
from harness.session import AgentSession
from harness.tools import default_registry

ROLES = ("coordinator", "researcher", "implementer", "verifier")


def run(coro):
    return asyncio.run(coro)


# ---- "fork does not break prompt cache" ---------------------------------------


def test_a_role_does_not_change_the_cached_prefix():
    """The first invariant. A role rides in the opening user turn precisely so
    the cacheable bytes stay identical — putting it in the `agent` layer would push
    governance and the MCP guidance below it and re-bill both once per role."""
    parent = build_effective_system_prompt()
    child = build_effective_system_prompt()

    assert child.system_instruction == parent.system_instruction

    session = AgentSession.create("cache-parent", role=load_role("researcher"))
    assert session.build_prompt().system_instruction == parent.system_instruction


def test_every_role_shares_one_prefix_and_one_tool_list():
    """N roles must cost one cache entry, not N. Declarations are most of the prefix, so
    this also means a child keeps tools it is not allowed to call."""
    prefixes = set()
    declarations = set()
    for role in ROLES:
        session = AgentSession.create(f"cache-{role}", role=load_role(role))
        prefixes.add(session.build_prompt().system_instruction)
        declarations.add(json.dumps(session.registry.declarations(), sort_keys=True))

    assert len(prefixes) == 1
    assert len(declarations) == 1


def test_the_role_reaches_the_model_in_the_volatile_turn():
    role = "You gather. You do not conclude."
    session = AgentSession.create("role-lands", role=role)
    try:
        run(_open_turn(session, "find the winners"))
        opening = session.transcript.all_nodes()[0].step
        texts = [b["text"] for b in opening["content"]]

        assert any(role in t for t in texts)
        assert role not in session.build_prompt().system_instruction
    finally:
        session.transcript.path.unlink(missing_ok=True)


async def _open_turn(session, message):
    """Record the opening turn the way `submit` does, without calling a model."""
    prompt = session.build_prompt()
    opening = prompt.initial_input(message)
    if session.role:
        opening["content"].insert(
            0, {"type": "text", "text": "Your job on this run:\n\n" + session.role.strip()}
        )
    session.transcript.append(opening, turn=0, meta=session.root_meta or None)


# ---- "sharing must be explicit" -----------------------------------------------


def test_a_child_cannot_spawn_a_child():
    """Depth is capped at one, enforced by policy rather than by hiding the tool — hiding
    it would change the declarations and break the shared prefix."""
    for role in ("researcher", "implementer", "verifier"):
        assert role_policy(role).evaluate("spawn_agent").decision == "deny"
    assert role_policy("coordinator").evaluate("spawn_agent").decision == "allow"


def test_a_researcher_may_read_and_may_not_write():
    policy = role_policy("researcher")

    assert policy.evaluate("mcp__atria__get_ad_account_ad").decision == "allow"
    assert policy.evaluate("read_memory").decision == "allow"
    for forbidden in ("append_memory", "mcp__atria__transcribe_ad_account_video",
                      "mcp__atria__follow_advertiser", "submit_plan"):
        assert policy.evaluate(forbidden).decision == "deny", forbidden


def test_an_implementer_has_no_research_to_do():
    """The brief carries what it needs; going back to the account invites it to re-derive
    findings it was handed, and to disagree with them."""
    policy = role_policy("implementer")

    assert policy.evaluate("mcp__atria__list_ad_account_ads").decision == "deny"
    assert policy.evaluate("read_memory").decision == "allow"


def test_a_session_without_a_pool_simply_cannot_delegate():
    session = AgentSession.create("no-pool")
    try:
        assert session.build_context().pool is None
    finally:
        session.transcript.path.unlink(missing_ok=True)


def test_two_sessions_share_nothing_by_accident():
    first = AgentSession.create("iso-a")
    second = AgentSession.create("iso-b")
    try:
        assert first.registry is not second.registry
        assert first._plan_state is not second._plan_state
        assert first.gate is not second.gate
        assert first.transcript.path != second.transcript.path
    finally:
        first.transcript.path.unlink(missing_ok=True)
        second.transcript.path.unlink(missing_ok=True)


# ---- "lifecycle closes" -------------------------------------------------------


def pool(tmp_path, **kwargs):
    return AgentPool(
        parent_id="s-parent", registry=default_registry(), agent_dir=AGENT_DIR,
        runs_dir=tmp_path, **kwargs,
    )


def test_a_child_is_named_for_its_parent(tmp_path):
    p = pool(tmp_path)

    assert p.child_id("researcher") == "s-parent.researcher-1"
    p._meta["s-parent.researcher-1"] = ("researcher", "x")
    assert p.child_id("researcher") == "s-parent.researcher-2"


def test_the_fan_out_cap_is_enforced(tmp_path):
    async def scenario():
        p = pool(tmp_path, max_children=2)
        p.spawn("researcher", "one")
        p.spawn("researcher", "two")
        with pytest.raises(ValueError, match="cap is 2"):
            p.spawn("researcher", "three")
        return await p.cancel()

    assert len(run(scenario())) == 2


def test_every_spawned_child_closes_even_when_cancelled(tmp_path):
    """The executor's invariant, one level up: a spawn that never reported would be a
    silent hole in what the coordinator believes it asked for."""
    async def scenario():
        p = pool(tmp_path)
        for i in range(3):
            p.spawn("researcher", f"task {i}")
        outcomes = await p.cancel()
        return p, outcomes

    p, outcomes = run(scenario())

    assert [o.child_id for o in outcomes] == [
        f"s-parent.researcher-{n}" for n in (1, 2, 3)
    ]
    assert all(o.is_error for o in outcomes)
    assert p.ledger() == outcomes, "the ledger is stable once closed"


def test_close_all_never_raises_on_an_open_ledger(tmp_path):
    async def scenario():
        p = pool(tmp_path)
        p.spawn("researcher", "x")
        with pytest.raises(RuntimeError):
            p.ledger()
        return p.close_all()

    assert len(run(scenario())) == 1


def test_a_missing_role_fails_loudly(tmp_path):
    """A worker with no job is a worker that will invent one, so an unknown role is an
    error rather than an empty prompt."""
    with pytest.raises(FileNotFoundError):
        load_role("nonexistent-role")

    async def scenario():
        p = pool(tmp_path)
        p.spawn("nonexistent-role", "x")
        return (await p.drain())[0]

    outcome = run(scenario())
    assert outcome.is_error and outcome.reason == "spawn_error"


# ---- what the coordinator reads -----------------------------------------------


def test_a_failed_child_reports_why_rather_than_vanishing():
    """The coordinator is the only role that can see a worker came back empty, and the
    brief has to say so or the implementer fills the hole with something plausible."""
    outcome = ChildOutcome(
        child_id="s-p.researcher-1", role="researcher", task="read the library",
        text="no answer within 600s", is_error=True, reason="timeout",
    )
    rendered = outcome.render()

    assert "FAILED (timeout)" in rendered
    assert "read the library" in rendered


def test_a_childs_answer_arrives_whole():
    """Workers return findings, not summaries. A worker that pre-compresses leaves the
    coordinator nothing to recompress, which is the one thing only it can do."""
    body = "### `6897420294631`\n- `spend`: $29.02\n- Hook: \"I thought the only way...\""
    outcome = ChildOutcome(
        child_id="s-p.researcher-1", role="researcher", task="account", text=body
    )

    assert body in outcome.render()


def test_every_shipped_role_has_a_description_and_a_policy():
    for role in ROLES:
        assert load_role(role).strip()
        assert isinstance(role_policy(role), PermissionPolicy)


# ---- the subagent failure matrix ----------------------------------------------
#
# The subagent failure modes, as assertions. Three were already true; three were not.


def test_spawning_does_not_block_the_parent(tmp_path):
    """`registerAsyncAgent` semantics: the parent carries on. It used to await the child,
    so across two live runs nothing ever actually ran concurrently."""
    async def scenario():
        p = pool(tmp_path)
        p.spawn("researcher", "one")
        p.spawn("researcher", "two")
        still_running = p.pending()          # observed before anything is awaited
        await p.cancel()
        return still_running

    assert len(run(scenario())) == 2


def test_pending_and_ready_describe_different_things(tmp_path):
    async def scenario():
        p = pool(tmp_path)
        p.spawn("researcher", "x")
        before = (p.pending(), p.ready())
        await p.cancel()
        return before, (p.pending(), p.ready())

    (pending_before, ready_before), (pending_after, ready_after) = run(scenario())

    assert pending_before and not ready_before
    assert not pending_after and len(ready_after) == 1


def test_cache_drift_refuses_the_fork(tmp_path):
    """The cache-safe parameters must align or the fork is refused.
    Nothing enforced it: a control-plane file edited mid-run would silently drop
    every child to paying full price, and the only symptom is a usage line."""
    async def scenario():
        p = pool(tmp_path, parent_prefix="a-prefix-that-no-longer-matches")
        p.spawn("researcher", "x")
        return (await p.drain())[0]

    outcome = run(scenario())

    assert outcome.is_error and outcome.reason == "prefix_divergence"
    assert "refusing the fork" in outcome.text


def test_a_matching_prefix_is_not_refused(tmp_path):
    """The check must not fire on the ordinary case, or it would refuse every fork.

    Exercises `_drift` directly rather than spawning: a real child would call a real
    model, and this is asserting a comparison, not a run."""
    from harness.agents import _fingerprint
    from harness.session import AgentSession

    child = AgentSession.create("prefix-probe", runs_dir=tmp_path)
    p = pool(tmp_path, parent_prefix=_fingerprint(child.build_prompt().system_instruction))
    p.registry = child.registry  # parent and child share one registry, as they do live

    assert p._drift(child) is None


def test_a_child_built_on_a_different_registry_is_refused(tmp_path):
    """Declarations are most of the prefix, so a second registry is a second cache entry
    even when every tool in it happens to match."""
    from harness.session import AgentSession

    child = AgentSession.create("registry-probe", runs_dir=tmp_path)
    p = pool(tmp_path, parent_prefix="anything")

    assert "different tool registry" in (p._drift(child) or "")


def test_a_child_pool_cannot_spawn(tmp_path):
    """Depth capped in code, not only in the role policies — those are a file someone
    can edit."""
    async def scenario():
        p = pool(tmp_path, depth=MAX_AGENT_DEPTH)
        with pytest.raises(ValueError, match="depth cap"):
            p.spawn("researcher", "x")
        return True

    assert run(scenario())


def test_a_hook_can_send_a_child_back(tmp_path, monkeypatch):
    """Exit 2 is a gate a shell script can hold: deterministic where the verifier is
    judgement. Capped at one bounce, like every other retry in this harness."""
    from harness import agents

    submissions = []

    class FakeResult:
        stop_reason, text, usage = "end_turn", "with NASA-grade in it", {}

    class FakeSession:
        transcript = type("T", (), {"path": tmp_path / "child.jsonl"})()

        async def submit(self, message, **_kwargs):
            submissions.append(message)
            return FakeResult()

    objections = iter([("fix the claim"), None])
    monkeypatch.setattr(agents, "fire", lambda *a, **k: _async(None))
    monkeypatch.setattr(agents, "objection", lambda _r: next(objections, None))

    p = pool(tmp_path)
    result = run(p._let_hooks_object(FakeSession(), "s-p.implementer-1", "implementer", FakeResult()))

    assert len(submissions) == 1, "bounced exactly once"
    assert "fix the claim" in submissions[0]
    assert result.stop_reason == "end_turn"


async def _async(value):
    return value


def test_spawn_stays_on_the_event_loop(tmp_path):
    """`spawn_agent` must be async even though it awaits nothing. A sync tool runs in a
    worker thread, where `create_task` raises "no running event loop" — which put child
    ids in the ledger with no task behind them, so nothing ever waited for them and the
    teardown reported two perfectly good researchers as cancelled."""
    import inspect

    from harness.tools import spawn_agent

    assert inspect.iscoroutinefunction(spawn_agent.fn)


def test_a_child_never_reaches_the_ledger_without_a_task(tmp_path):
    """The ordering that made the bug above so hard to see: the id was recorded first, so
    a failed `create_task` left an entry nothing was running."""
    async def scenario():
        p = pool(tmp_path)
        p.spawn("researcher", "x")
        assert set(p.pending()) == set(p._issued), "every issued child has a live task"
        await p.cancel()

    run(scenario())
