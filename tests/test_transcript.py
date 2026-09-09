"""The transcript tree: append-only, walkable, branchable, reloadable."""

from harness.transcript import Transcript


def user(text: str) -> dict:
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def model(text: str) -> dict:
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def test_append_builds_a_chain_and_steps_walk_it_in_order():
    t = Transcript("s-test")
    t.append(user("one"))
    t.append(model("two"))
    t.append(user("three"))

    texts = [s["content"][0]["text"] for s in t.steps()]
    assert texts == ["one", "two", "three"]


def test_head_advances_with_each_append():
    t = Transcript("s-test")
    first = t.append(user("one"))
    assert t.head == first.id
    second = t.append(model("two"))
    assert t.head == second.id
    assert second.parent_id == first.id


def test_branch_from_creates_a_sibling_not_a_continuation():
    t = Transcript("s-test")
    root = t.append(user("shared question"))
    t.append(model("first answer"))

    # Rewind to the shared node and take a different path.
    t.branch_from(root.id)
    t.append(model("second answer"))

    children = t.children(root.id)
    assert len(children) == 2
    assert t.is_branch_point(root.id)

    # Two branches, two different walks, both containing the shared prefix.
    walk_a = [s["content"][0]["text"] for s in t.steps(children[0].id)]
    walk_b = [s["content"][0]["text"] for s in t.steps(children[1].id)]
    assert walk_a == ["shared question", "first answer"]
    assert walk_b == ["shared question", "second answer"]


def test_branching_never_rewrites_earlier_nodes():
    t = Transcript("s-test")
    root = t.append(user("shared"))
    original = t.append(model("first"))
    t.branch_from(root.id)
    t.append(model("second"))

    # The first branch is untouched and still reachable.
    assert t.get(original.id).step["content"][0]["text"] == "first"
    assert len(t) == 3


def test_reloaded_transcript_reproduces_the_same_walk(tmp_path):
    t = Transcript.create("s-persist", runs_dir=tmp_path)
    t.append(user("one"))
    t.append(model("two"))
    branch_point = t.append(user("three"))
    t.branch_from(branch_point.id)
    t.append(model("branch"))

    before = t.steps()

    reloaded = Transcript.load("s-persist", runs_dir=tmp_path)
    assert len(reloaded) == len(t)
    assert reloaded.steps() == before
    assert reloaded.head == t.head  # head lands on the last node written


def test_persistence_is_append_only_jsonl(tmp_path):
    t = Transcript.create("s-lines", runs_dir=tmp_path)
    t.append(user("one"))
    t.append(model("two"))

    lines = (tmp_path / "s-lines.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2

    t.append(user("three"))
    lines_after = (tmp_path / "s-lines.jsonl").read_text().strip().splitlines()
    assert len(lines_after) == 3
    assert lines_after[:2] == lines, "existing lines must never be rewritten"


def test_render_marks_head_and_branch_points():
    t = Transcript("s-render")
    root = t.append(user("q"))
    t.append(model("a"))
    t.branch_from(root.id)
    t.append(model("b"))

    output = t.render()
    assert "<- HEAD" in output
    assert "*branch*" in output
    assert output.count("\n") >= 3
