"""The session transcript: an append-only tree of interaction steps.

History is a tree, not a list. Each node wraps exactly one Interactions step and points at
its parent; the conversation sent to the model is a walk from the current head back to a
root, reversed. Two children of one parent *are* two branches — there is no branch object.

Append-only is the point. Branching never rewrites, a crash leaves a readable partial
transcript, and every run is auditable after the fact.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import RUNS_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class Node:
    """One step in the tree."""

    id: str
    parent_id: str | None
    step: dict
    turn: int
    ts: str
    meta: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return str(self.step.get("type", "unknown"))

    def summary(self, width: int = 60) -> str:
        """A one-line description for tree rendering."""
        step = self.step
        kind = self.kind
        if kind in ("user_input", "model_output"):
            text = " ".join(
                part.get("text", "")
                for block in step.get("content", []) or []
                for part in (block.get("parts") or [block])
                if isinstance(part, dict)
            ).strip()
            text = " ".join(text.split())
        elif kind == "function_call":
            text = f"{step.get('name')}({json.dumps(step.get('arguments', {}))})"
        elif kind == "function_result":
            text = f"-> {step.get('name') or step.get('call_id')}"
            if step.get("is_error"):
                text = f"{text} [error]"
        else:
            text = json.dumps(step)[:width]
        return text[:width] + ("…" if len(text) > width else "")


class Transcript:
    """An append-only node tree with a movable head."""

    def __init__(
        self,
        session_id: str,
        path: Path | None = None,
        nodes: dict[str, Node] | None = None,
        head: str | None = None,
    ):
        self.session_id = session_id
        self.path = path
        self._nodes: dict[str, Node] = nodes or {}
        self._order: list[str] = list(self._nodes)
        self.head: str | None = head

    # ---- reading -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, node_id: str) -> Node:
        if node_id not in self._nodes:
            raise KeyError(f"no such node: {node_id}")
        return self._nodes[node_id]

    def path_to_root(self, node_id: str | None = None) -> list[Node]:
        """Walk head (or `node_id`) back to a root, returned root-first."""
        cursor = node_id if node_id is not None else self.head
        chain: list[Node] = []
        seen: set[str] = set()
        while cursor is not None:
            if cursor in seen:  # defensive: a cycle would hang the loop
                raise ValueError(f"cycle in transcript at {cursor}")
            seen.add(cursor)
            node = self.get(cursor)
            chain.append(node)
            cursor = node.parent_id
        chain.reverse()
        return chain

    def steps(self, node_id: str | None = None) -> list[dict]:
        """The linear history to send as `input`. This is the runtime context."""
        return [node.step for node in self.path_to_root(node_id)]

    def all_steps(self) -> list[dict]:
        """Every step ever recorded, in write order — across branches and compactions.

        Not the runtime context: `steps()` is what the model can see now, which after a
        compaction excludes the history the new root replaced. This is what the session
        was ever told, which is the right question for auditing a figure in the output.
        """
        return [self._nodes[node_id].step for node_id in self._order]

    def children(self, node_id: str | None) -> list[Node]:
        return [
            self._nodes[nid] for nid in self._order if self._nodes[nid].parent_id == node_id
        ]

    def roots(self) -> list[Node]:
        return self.children(None)

    def is_branch_point(self, node_id: str) -> bool:
        return len(self.children(node_id)) > 1

    # ---- writing -------------------------------------------------------------

    def append(self, step: dict, *, turn: int = 0, meta: dict | None = None) -> Node:
        """Add `step` as a child of the current head and advance the head onto it."""
        node = Node(
            id=new_id(),
            parent_id=self.head,
            step=step,
            turn=turn,
            ts=_now(),
            meta=meta or {},
        )
        self._nodes[node.id] = node
        self._order.append(node.id)
        self.head = node.id
        self._persist(node)
        return node

    def compact_boundary(
        self,
        summary: str,
        retained: list[Node],
        meta: dict | None = None,
    ) -> Node:
        """Replace the walked history with a summary, without destroying anything.

        The boundary is appended as a **new root** — its parent is None — so walking from
        the new head yields the summary plus the retained steps and nothing else. The whole
        pre-compaction branch stays in the file, still renderable and still branchable from;
        `compacted_from` records the link the parent pointer no longer carries.

        This is what the tree buys us: compaction is non-destructive by construction rather
        than by keeping a separate archive.
        """
        previous_head = self.head
        self.head = None  # the next append has no parent, so it starts a new root

        boundary = self.append(
            {"type": "user_input", "content": [{"type": "text", "text": summary}]},
            turn=0,
            meta={
                "compact_boundary": True,
                "compacted_from": previous_head,
                "retained": len(retained),
                **(meta or {}),
            },
        )
        for node in retained:
            self.append(node.step, turn=node.turn, meta={"retained_from": node.id})
        return boundary

    def branch_from(self, node_id: str) -> Node:
        """Move the head back. The next append becomes a sibling, not a continuation."""
        node = self.get(node_id)
        self.head = node.id
        return node

    # ---- persistence ---------------------------------------------------------

    def _persist(self, node: Node) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(node), default=str) + "\n")

    @classmethod
    def create(cls, session_id: str | None = None, runs_dir: Path | None = None) -> Transcript:
        session_id = session_id or new_id("s-")
        directory = runs_dir if runs_dir is not None else RUNS_DIR
        return cls(session_id=session_id, path=directory / f"{session_id}.jsonl")

    @classmethod
    def load(cls, session_id: str, runs_dir: Path | None = None) -> Transcript:
        """Rebuild from disk.

        The head lands on the last node written, which is where an unbranched session left
        off. To continue from anywhere else, call `branch_from` explicitly.
        """
        directory = runs_dir if runs_dir is not None else RUNS_DIR
        path = directory / f"{session_id}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"no transcript for session {session_id}: {path}")

        nodes: dict[str, Node] = {}
        order: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            node = Node(**json.loads(line))
            nodes[node.id] = node
            order.append(node.id)

        transcript = cls(session_id=session_id, path=path, nodes=nodes)
        transcript._order = order
        transcript.head = order[-1] if order else None
        return transcript

    # ---- rendering -----------------------------------------------------------

    def annotate(self, node_id: str | None = None) -> list[tuple[str, Node]]:
        """The path, each node labelled `submission.turn`.

        `Node.turn` is the query loop's own counter and restarts at every submission, so
        on a path that crosses a resume or a branch it repeats — two turn-1s in a row,
        which reads as nonsense. The submission number is derived here by counting
        `user_input` boundaries along the path rather than stored on the node: a stored
        global counter would be wrong the moment two branches share a prefix.
        """
        labelled: list[tuple[str, Node]] = []
        submission = 0
        for node in self.path_to_root(node_id):
            if node.kind == "user_input":
                submission += 1
            labelled.append((f"{submission}.{node.turn}", node))
        return labelled

    def render(self) -> str:
        """An indented tree, marking the head and any branch points."""
        lines = [f"session {self.session_id}  ({len(self)} nodes)"]

        def walk(node: Node, depth: int, submission: int) -> None:
            if node.kind == "user_input":
                submission += 1
            marker = " <- HEAD" if node.id == self.head else ""
            fork = "  *branch*" if self.is_branch_point(node.id) else ""
            if node.meta.get("compact_boundary"):
                fork += "  *COMPACTED*"
            lines.append(
                f"{'  ' * depth}{submission}.{node.turn}  {node.id}  "
                f"[{node.kind}] {node.summary()}{fork}{marker}"
            )
            for child in self.children(node.id):
                walk(child, depth + 1, submission)

        for root in self.roots():
            walk(root, 1, 0)
        return "\n".join(lines)
