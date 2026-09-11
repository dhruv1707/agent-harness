"""Plan mode: research freely, change nothing, until a person approves.

The property under test is that plan mode is a boundary rather than a convention — it
fails closed, `--yes` cannot talk its way past it, and a refusal is not an argument.
"""

import asyncio
import json

import pytest

from harness.attachments import from_lineage, render_boundary
from harness.config import MAX_PLAN_ATTEMPTS
from harness.permissions import (
    PermissionGate,
    PermissionPolicy,
    PermissionRequest,
    PlanState,
    terminal_asker,
)
from harness.tools import ToolContext, submit_plan
from harness.transcript import Transcript


def run(coro):
    return asyncio.run(coro)


def gate(*, active=True, asker=None, auto_approve=False, plan=None, **policy):
    state = plan or PlanState(active=active)
    return (
        PermissionGate(
            PermissionPolicy(confirm=("submit_plan",), **policy),
            asker=asker,
            auto_approve=auto_approve,
            plan=state,
        ),
        state,
    )


# ---- the boundary itself ------------------------------------------------------


def test_a_write_is_refused_while_a_plan_is_pending():
    g, _ = gate(allow=("append_memory",))
    verdict = run(g.check("c1", "append_memory", {}, read_only=False))

    assert verdict.denied
    assert "plan mode" in verdict.reason


def test_a_read_still_runs():
    g, _ = gate(allow=("read_memory",))
    assert run(g.check("c1", "read_memory", {}, read_only=True)).allowed


def test_a_tool_of_unknown_kind_is_refused():
    """`read_only` defaults False, so a caller that never says fails closed. This is what
    makes an unannotated MCP tool safe in plan mode rather than merely undescribed."""
    g, _ = gate(allow=("mystery_tool",))
    assert run(g.check("c1", "mystery_tool", {})).denied


def test_lifting_the_mode_restores_the_refused_tools():
    g, state = gate(allow=("append_memory",))
    assert run(g.check("c1", "append_memory", {})).denied
    state.active = False
    assert run(g.check("c2", "append_memory", {})).allowed


def test_an_explicit_deny_keeps_its_own_reason_in_plan_mode():
    """Plan mode must never soften anything. A denied tool stays denied by policy, with
    the rule that denied it, so the audit trail does not change shape under plan mode."""
    g, _ = gate(deny=("unfollow_advertiser",))
    verdict = run(g.check("c1", "unfollow_advertiser", {}))

    assert verdict.denied and verdict.rule == "unfollow_advertiser"
    assert "plan mode" not in verdict.reason


# ---- who may approve ----------------------------------------------------------


def test_yes_cannot_approve_a_plan():
    """The decision that makes a run stop being read-only is not one a flag makes."""
    g, _ = gate(auto_approve=True, asker=None)
    verdict = run(g.check("c1", "submit_plan", {"plan": "do the thing"}))

    assert verdict.denied
    assert verdict.unattended


def test_yes_still_answers_an_ordinary_ask():
    g, _ = gate(ask=("append_memory",), auto_approve=True)
    assert run(g.check("c1", "append_memory", {}, read_only=False)).denied, "plan mode first"
    g2, state = gate(ask=("append_memory",), auto_approve=True, active=False)
    assert run(g2.check("c1", "append_memory", {}, read_only=False)).allowed


def test_nobody_to_ask_is_pending_not_a_refusal():
    """'There was no human' and 'a human said no' are different facts. Counting the first
    as a refusal spends an attempt the agent never had."""
    g, state = gate(asker=None)
    run(g.check("c1", "submit_plan", {"plan": "x"}))

    assert state.pending and state.attempts == 0


def test_a_refusal_counts_and_a_second_one_stops_it():
    g, state = gate(asker=lambda request: "n")
    run(g.check("c1", "submit_plan", {"plan": "first"}))
    run(g.check("c2", "submit_plan", {"plan": "second"}))

    assert state.attempts == MAX_PLAN_ATTEMPTS and not state.pending


def test_approving_one_plan_does_not_approve_the_next():
    """'always' is not on offer for a confirm, and answering it anyway grants once."""
    answers = iter(["a", "n"])
    g, state = gate(asker=lambda request: next(answers))

    assert run(g.check("c1", "submit_plan", {"plan": "first"})).allowed
    assert run(g.check("c2", "submit_plan", {"plan": "second"})).denied
    assert state.attempts == 1


def test_the_operator_is_told_which_kind_of_decision_it_is():
    seen: list[PermissionRequest] = []

    def asker(request):
        seen.append(request)
        return "y"

    g, _ = gate(asker=asker)
    run(g.check("c1", "submit_plan", {"plan": "x"}))

    assert seen[0].kind == "confirm" and seen[0].one_shot


def test_a_multi_line_argument_prints_as_itself(capsys, monkeypatch):
    """A plan shown as escaped JSON is unreadable at the moment it must be read."""
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    terminal_asker(
        PermissionRequest("submit_plan", {"plan": "1. first\n2. second"}, kind="confirm")
    )
    err = capsys.readouterr().err

    assert "1. first" in err and "\\n" not in err
    assert "always" not in err, "there is no 'always' for a one-shot approval"


# ---- the tool -----------------------------------------------------------------


def test_submitting_lifts_the_mode():
    state = PlanState(active=True)
    result = submit_plan.invoke(ToolContext(session_id="s", plan=state), plan="do it")

    assert not state.active
    assert "approved" in result.lower()


def test_submitting_outside_plan_mode_says_so_rather_than_flipping_anything():
    state = PlanState(active=False)
    result = submit_plan.invoke(ToolContext(session_id="s", plan=state), plan="do it")

    assert "not in plan mode" in result


def test_the_plan_tool_is_a_barrier():
    """It must run alone: a sibling call could otherwise race its own permission check
    against the approval that would have allowed it."""
    assert submit_plan.concurrency_safe is False
    assert submit_plan.read_only is False


# ---- the plan survives whatever the verdict -----------------------------------


def test_the_plan_is_recorded_before_anyone_approves_it():
    """The call step is written before the gate runs, so a refused plan is still readable
    afterwards — which is what makes an unattended planning run useful."""
    transcript = Transcript.create("plan-record")
    try:
        transcript.append(
            {"type": "user_input", "content": [{"type": "text", "text": "do a thing"}]}
        )
        transcript.append(
            {
                "type": "function_call",
                "id": "p1",
                "name": "submit_plan",
                "arguments": {"plan": "1. rank\n2. report"},
            }
        )
        transcript.append(
            {
                "type": "function_result",
                "call_id": "p1",
                "name": "submit_plan",
                "result": [{"type": "text", "text": "permission denied: declined"}],
            }
        )
        attached = from_lineage(transcript.lineage())

        assert [i.kind for i in attached.items] == ["task"], "a refused plan is not attached"
        raw = json.dumps([n.step for n in transcript.lineage()])
        assert "1. rank" in raw, "but it is still on the record"
    finally:
        transcript.path.unlink(missing_ok=True)


def test_an_approved_plan_is_attached():
    transcript = Transcript.create("plan-attach")
    try:
        transcript.append(
            {"type": "user_input", "content": [{"type": "text", "text": "do a thing"}]}
        )
        transcript.append(
            {
                "type": "function_call",
                "id": "p1",
                "name": "submit_plan",
                "arguments": {"plan": "1. rank\n2. report"},
            }
        )
        transcript.append(
            {
                "type": "function_result",
                "call_id": "p1",
                "name": "submit_plan",
                "result": [{"type": "text", "text": "Plan approved."}],
            }
        )
        boundary = render_boundary(transcript.lineage(), "# Session Memory\n## Next\ngo")

        assert "1. rank" in boundary and "do a thing" in boundary
        assert boundary.index("do a thing") < boundary.index("## Next"), (
            "the task outranks the brief: it is the one thing nothing can regenerate"
        )
    finally:
        transcript.path.unlink(missing_ok=True)
