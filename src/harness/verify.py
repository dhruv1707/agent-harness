"""Check that the figures in a report match what the tools actually returned.

The agent reads metrics from Atria and then writes them into prose, and nothing about
that second step is grounded. On a run that pulled 120 rows across three overlapping
list calls, five of seven reported ads carried spend and ROAS figures found in no tool
result: `$1,470.92` for an ad the API had reported at `$29.02`. The report was right
about *which* ads won and wrong about every number beside them — worse than being wrong
about both, because nothing on the page looks off.

A prompt rule telling the model to quote rather than recall is worth having and is not a
control. This is the control. The transcript already holds every byte every tool
returned, so the arithmetic can be checked without asking anyone's opinion.

Two checks, in descending order of confidence:

**Contradiction.** The report writes ``- `spend`: $1,281.33`` under a heading naming ad
`52551364556835`, and that ad's record says `65.79`. There is no interpretation under
which that is right. This is the check that matters.

**Absence.** A figure that appears nowhere in any tool output. Weaker — with several
hundred numbers in play a fabricated one can collide with an unrelated real one — so a
clean absence result proves little, while a hit is worth reading.

It reads any tool payload shaped as *entity id plus metrics*, which is not specific to
one vendor but is an assumption: a server that reports metrics some other way is checked
only by the weaker pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

_NUMBER = re.compile(r"(?<![\w.\-])\$?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%?)")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

#: ``- `spend`: $1,281.33`` and its unquoted variants: a metric named, then its value.
_LABELLED = re.compile(
    r"[`*_]*([A-Za-z][A-Za-z0-9_ ]{2,30}?)[`*_]*\s*[:=]\s*\*{0,2}\$?"
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%?)"
)

#: An identifier long enough that a collision with a quantity is not a concern.
_ID = re.compile(r"\b\d{8,}\b")

#: Below this, a figure is structure rather than measurement — list numbering, "top 5".
MIN_INTERESTING = Decimal("10")


def _dec(whole: str, frac: str | None) -> Decimal | None:
    try:
        return Decimal(whole.replace(",", "") + ("." + frac if frac else ""))
    except InvalidOperation:
        return None


def _matches(source: Decimal, claimed: Decimal) -> bool:
    """Is `source`, rounded the way the report rounded it, the claimed figure?

    A ratio in the data (0.354) is routinely written as a percentage (35.4%) and back,
    so both scalings count as the same figure.
    """
    quantum = Decimal(1).scaleb(claimed.as_tuple().exponent)
    for candidate in (source, source * 100, source / 100):
        try:
            if candidate.quantize(quantum) == claimed:
                return True
        except InvalidOperation:
            pass
        # Binary floating point does not round-trip: an agent copying 65.79 out of a
        # payload that holds 65.78999999999999 has copied it correctly. Differences this
        # small are representation, not disagreement — a fabricated figure is never
        # within a billionth of the real one.
        if candidate and abs(candidate - source) <= abs(source) * Decimal("1e-9"):
            if abs(candidate - claimed) <= max(abs(claimed), Decimal(1)) * Decimal("1e-9"):
                return True
    return False


# ---- what the tools said ------------------------------------------------------


@dataclass
class Index:
    """Everything the run was told, addressable two ways."""

    numbers: set[Decimal] = field(default_factory=set)
    #: entity id -> metric name -> value, from any payload shaped that way.
    metrics: dict[str, dict[str, Decimal]] = field(default_factory=dict)

    def record(self, entity: str, name: str, value: Decimal) -> None:
        self.metrics.setdefault(entity, {})[name] = value


def _walk(node: object, index: Index) -> None:
    """Collect `{id, metrics: {...}}` objects wherever they sit in a payload."""
    if isinstance(node, list):
        for item in node:
            _walk(item, index)
        return
    if not isinstance(node, dict):
        return

    ids = [
        str(value)
        for key, value in node.items()
        if key.endswith("_id") and isinstance(value, (str, int)) and _ID.fullmatch(str(value))
    ]
    metrics = node.get("metrics")
    if ids and isinstance(metrics, dict):
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                for entity in ids:
                    index.record(entity, name, Decimal(str(value)))

    for value in node.values():
        _walk(value, index)


def index_sources(sources: list[str]) -> Index:
    index = Index()
    for source in sources:
        for match in _NUMBER.finditer(_ISO_DATE.sub(" ", source)):
            value = _dec(match.group(1), match.group(2))
            if value is not None:
                index.numbers.add(value)
        try:
            payload = json.loads(source)
        except (ValueError, TypeError):
            continue
        # Bridged HTTP responses nest the real payload as a JSON string.
        body = payload.get("body") if isinstance(payload, dict) else None
        if isinstance(body, str):
            try:
                payload = json.loads(body)
            except ValueError:
                pass
        _walk(payload, index)
    return index


# ---- what the report said -----------------------------------------------------


@dataclass(frozen=True)
class Figure:
    value: Decimal
    raw: str
    line: str
    entity: str | None = None
    metric: str | None = None
    actual: Decimal | None = None

    def __str__(self) -> str:
        where = f"{self.entity} " if self.entity else ""
        if self.actual is not None:
            return f"{where}{self.metric}: report says {self.raw}, tools said {self.actual}"
        return f"{self.raw} — {self.line.strip()[:80]}"


def _figures(report: str, index: Index) -> tuple[list[Figure], list[Figure]]:
    """Split the report's numbers into labelled metrics and everything else."""
    labelled: list[Figure] = []
    loose: list[Figure] = []
    entity: str | None = None

    for line in report.splitlines():
        for found in _ID.findall(line):
            if found in index.metrics:
                entity = found  # a heading names the ad the next lines describe
        cleaned = _ISO_DATE.sub(" ", line)

        claimed_spans = []
        for match in _LABELLED.finditer(cleaned):
            value = _dec(match.group(2), match.group(3))
            if value is None:
                continue
            name = match.group(1).strip().lower().replace(" ", "_")
            if entity and name in index.metrics.get(entity, {}):
                labelled.append(
                    Figure(value, match.group(0).split(":")[-1].strip(), line, entity, name,
                           index.metrics[entity][name])
                )
                claimed_spans.append(match.span(2))

        for match in _NUMBER.finditer(cleaned):
            value = _dec(match.group(1), match.group(2))
            if value is None or abs(value) < MIN_INTERESTING:
                continue
            if match.span(1) in claimed_spans:
                continue
            loose.append(Figure(value, match.group(0), line))
    return labelled, loose


@dataclass(frozen=True)
class Verdict:
    checked: int
    contradicted: list[Figure]
    unsupported: list[Figure]

    @property
    def ok(self) -> bool:
        return not self.contradicted and not self.unsupported

    def render(self) -> str:
        if not self.checked:
            return "[verify] no figures in the report to check"
        if self.ok:
            return f"[verify] all {self.checked} figures agree with the tool results"

        lines = []
        if self.contradicted:
            lines.append(
                f"[verify] {len(self.contradicted)} figure(s) CONTRADICT the tool results:"
            )
            lines += [f"         {figure}" for figure in self.contradicted]
        if self.unsupported:
            seen: set[str] = set()
            unique = [f for f in self.unsupported if not (f.raw in seen or seen.add(f.raw))]
            lines.append(
                f"[verify] {len(unique)} figure(s) appear in no tool output:"
            )
            lines += [f"         {figure}" for figure in unique]
        lines.append(f"[verify] {self.checked} figures checked")
        return "\n".join(lines)


def verify(report: str, sources: list[str]) -> Verdict:
    """Diff a report's figures against everything the run was told.

    `sources` is every tool result plus the prompt: a number handed over in governance or
    read out of a memory file is as grounded as one from an API.
    """
    index = index_sources(sources)
    labelled, loose = _figures(report, index)

    contradicted = [f for f in labelled if f.actual is None or not _matches(f.actual, f.value)]
    unsupported = [
        f for f in loose if not any(_matches(source, f.value) for source in index.numbers)
    ]
    return Verdict(
        checked=len(labelled) + len(loose),
        contradicted=contradicted,
        unsupported=unsupported,
    )


def sources_from_steps(steps: list[dict]) -> list[str]:
    """Everything the run was told, as flat text: tool results and the opening context.

    A compacted session is checked against what survives on the current path, which is
    what the agent could still see when it wrote the report.
    """
    out: list[str] = []
    for step in steps:
        kind = step.get("type")
        if kind == "function_result":
            out += [b.get("text", "") for b in (step.get("result") or [])]
        elif kind == "user_input":
            out += [b.get("text", "") for b in (step.get("content") or [])]
    return [text for text in out if text]
