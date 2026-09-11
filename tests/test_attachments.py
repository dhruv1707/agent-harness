"""What the agent was working from, carried across a compaction.

Compaction replaces the history with a brief written from a truncated view of it. Numbers
were fixed by transcribing them; everything else — the request, the plan, the open files —
is re-attached instead, and derived from the tree so there is no second copy to drift.
"""

import json
from pathlib import Path

from harness.attachments import from_lineage, render_boundary
from harness.config import MAX_ATTACHED_FILE_TOKENS, MAX_ATTACHMENT_TOKENS, approx_tokens
from harness.session_memory import MAX_SESSION_MEMORY_TOKENS
from harness.compaction import strip_for_summary
from harness.config import CONTEXT_BUDGET_TOKENS, compact_threshold
from harness.transcript import Transcript


def build(session_id, steps):
    transcript = Transcript.create(session_id)
    for step in steps:
        transcript.append(step)
    return transcript


def call(tool, call_id, **arguments):
    return {"type": "function_call", "id": call_id, "name": tool, "arguments": arguments}


def result(tool, call_id, text):
    return {
        "type": "function_result",
        "call_id": call_id,
        "name": tool,
        "result": [{"type": "text", "text": text}],
    }


def opening(text):
    return {
        "type": "user_input",
        "content": [{"type": "text", "text": "# Run Context"}, {"type": "text", "text": text}],
    }


# ---- derivation ---------------------------------------------------------------


def test_the_task_is_the_user_s_own_words_not_the_run_context():
    transcript = build("attach-task", [opening("find the best hooks")])
    try:
        items = from_lineage(transcript.lineage()).items
        assert [(i.kind, i.body) for i in items] == [("task", "find the best hooks")]
    finally:
        transcript.path.unlink(missing_ok=True)


def test_a_failed_read_attaches_nothing():
    transcript = build(
        "attach-failed",
        [
            opening("go"),
            call("read_memory", "c1", name="nope"),
            result("read_memory", "c1", "tool error: no such memory file: nope"),
        ],
    )
    try:
        assert not from_lineage(transcript.lineage()).files
    finally:
        transcript.path.unlink(missing_ok=True)


def test_a_file_read_twice_attaches_once():
    transcript = build(
        "attach-dup",
        [
            opening("go"),
            call("read_memory", "c1", name="hook-patterns"),
            result("read_memory", "c1", "# Hook Patterns"),
            call("read_memory", "c2", name="hook-patterns.md"),
            result("read_memory", "c2", "# Hook Patterns"),
        ],
    )
    try:
        assert [f.name for f in from_lineage(transcript.lineage()).files] == [
            "hook-patterns.md"
        ]
    finally:
        transcript.path.unlink(missing_ok=True)


def test_derivation_crosses_a_compaction_boundary():
    """The live path forgets what a boundary replaced, and the full node set remembers
    branches this run abandoned. Lineage is the only source that answers the question."""
    transcript = build(
        "attach-lineage",
        [opening("find the best hooks"), call("read_memory", "c1", name="hook-patterns")],
    )
    try:
        transcript.append(result("read_memory", "c1", "# Hook Patterns"))
        transcript.compact_boundary("# Session Memory", retained=[])

        assert not from_lineage([n for n in transcript.path_to_root()]).items
        kinds = {i.kind for i in from_lineage(transcript.lineage()).items}
        assert kinds == {"task", "file"}
    finally:
        transcript.path.unlink(missing_ok=True)


# ---- rendering and budget -----------------------------------------------------


def test_a_file_comes_back_as_it_is_now_not_as_it_was_read(tmp_path):
    """Held by name for exactly this reason: the agent may have appended to it since."""
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "hook-patterns.md").write_text("# Hook Patterns\noriginal", encoding="utf-8")
    transcript = build(
        "attach-fresh",
        [
            opening("go"),
            call("read_memory", "c1", name="hook-patterns"),
            result("read_memory", "c1", "# Hook Patterns\noriginal"),
        ],
    )
    try:
        (memory / "hook-patterns.md").write_text("# Hook Patterns\nappended", encoding="utf-8")
        rendered = from_lineage(transcript.lineage()).render(memory_dir=memory)
        assert "appended" in rendered and "original" not in rendered
    finally:
        transcript.path.unlink(missing_ok=True)


def test_a_deleted_file_says_so_rather_than_vanishing():
    transcript = build(
        "attach-gone",
        [
            opening("go"),
            call("read_memory", "c1", name="ghost"),
            result("read_memory", "c1", "# Ghost"),
        ],
    )
    try:
        rendered = from_lineage(transcript.lineage()).render(
            memory_dir=Path("/nonexistent-memory-dir")
        )
        assert "no longer on disk" in rendered
    finally:
        transcript.path.unlink(missing_ok=True)


def test_nothing_attached_renders_to_nothing():
    """So a boundary with no attachments is byte-identical to one written before this
    existed, and the compaction tests that predate it keep meaning what they meant."""
    transcript = build("attach-empty", [])
    try:
        assert from_lineage(transcript.lineage()).render() == ""
        assert render_boundary(transcript.lineage(), "BRIEF") == "BRIEF"
    finally:
        transcript.path.unlink(missing_ok=True)


def test_the_task_and_plan_are_never_shed(tmp_path):
    """Enough files to exceed the whole-set budget even after each is capped. The task
    and the plan are the two things nothing can regenerate, so the budget bends for them
    and the files go."""
    memory = tmp_path / "memory"
    memory.mkdir()
    steps = [opening("the irreducible request")]
    for index in range(10):
        (memory / f"f{index}.md").write_text("line\n" * 4_000, encoding="utf-8")
        steps.append(call("read_memory", f"c{index}", name=f"f{index}"))
        steps.append(result("read_memory", f"c{index}", "..."))
    steps.append(call("submit_plan", "p1", plan="the irreducible plan"))
    steps.append(result("submit_plan", "p1", "Plan approved."))

    transcript = build("attach-shed", steps)
    try:
        rendered = from_lineage(transcript.lineage()).render(memory_dir=memory)
        assert "the irreducible request" in rendered
        assert "the irreducible plan" in rendered
        assert "over budget" in rendered, "and the drop is named, so it can be re-read"
    finally:
        transcript.path.unlink(missing_ok=True)


def test_one_file_cannot_eat_the_whole_budget(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "big.md").write_text("# Top\n" + "line\n" * 20_000, encoding="utf-8")
    transcript = build(
        "attach-trim",
        [opening("go"), call("read_memory", "c1", name="big"), result("read_memory", "c1", "x")],
    )
    try:
        rendered = from_lineage(transcript.lineage()).render(memory_dir=memory)
        assert approx_tokens(rendered) <= MAX_ATTACHMENT_TOKENS + 200
        assert "# Top" in rendered, "trimmed from the tail: a document's framing is at the top"
        assert "truncated" in rendered
    finally:
        transcript.path.unlink(missing_ok=True)


def test_the_boundary_fits_under_the_compaction_threshold():
    """Attachments and the brief are additive at the boundary. If together they exceeded
    the threshold, the turn after a compaction would be instantly over it again."""
    assert MAX_ATTACHMENT_TOKENS + MAX_SESSION_MEMORY_TOKENS < compact_threshold(
        CONTEXT_BUDGET_TOKENS
    )


def test_a_single_file_is_bounded_well_under_the_whole_set():
    assert MAX_ATTACHED_FILE_TOKENS * 3 <= MAX_ATTACHMENT_TOKENS


# ---- the two bugs this change had to fix --------------------------------------


def test_the_root_of_the_walk_is_never_elided():
    """After the first compaction the root *is* the previous boundary — task, plan and
    brief. Eliding it at 1,500 bytes discarded everything a 12,000-token brief had just
    been written to preserve, `## Next` included, on the second compaction of any long
    run."""
    body = "# Session Memory\n## Findings\n" + ("- a finding\n" * 400) + "## Next\ngo"
    root = {"type": "user_input", "content": [{"type": "text", "text": body}]}
    later = {"type": "user_input", "content": [{"type": "text", "text": "y" * 4_000}]}

    cleaned = strip_for_summary([root, later])
    kept_root = "".join(b.get("text", "") for b in cleaned[0]["content"])
    kept_later = "".join(b.get("text", "") for b in cleaned[1]["content"])

    assert "## Next" in kept_root and len(kept_root) == len(body)
    assert len(kept_later) < 4_000, "later turns still elide"


def test_the_model_s_own_brief_is_not_evidence():
    """A boundary is a user_input, so the flat view counted it as a source: a figure
    invented in the brief then validated itself in the report, and the checker reported
    'all figures agree' for a number in no tool output anywhere."""
    from harness.verify import sources_from_nodes, verify

    ads = json.dumps(
        {"body": json.dumps({"data": {"items": [{"platform_ad_id": "6897420294631",
                                                 "metrics": {"spend": 29.02}}]}})}
    )
    transcript = build(
        "attach-launder",
        [{"type": "function_result", "name": "ads", "result": [{"type": "text", "text": ads}]}],
    )
    try:
        transcript.compact_boundary(
            "# Session Memory\n## Findings\n- Account spend was $88,412.10.", retained=[]
        )
        verdict = verify(
            "Total account spend was $88,412.10.", sources_from_nodes(transcript.all_nodes())
        )
        assert verdict.unsupported, "an invented figure cannot launder itself through a brief"
    finally:
        transcript.path.unlink(missing_ok=True)
