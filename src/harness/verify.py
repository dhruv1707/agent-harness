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
clean absence result proves little, while a hit is worth reading. It can never catch a
*real* figure filed under the wrong ad, which is the common failure.

**The report's format is therefore load-bearing**, and the verdict says so. Only a figure
written `metric: value` under a heading naming an ad reaches the strong check. A run once
laid its metrics out as ``- **Metrics**: `roas` 8.210 | `spend` $29.02`` — no colon before
a number, so no pair — and the verdict read "all 60 figures agree" having compared none of
them to anything. `Verdict.weakly_checked_only` exists so that cannot recur silently.

It reads any tool payload shaped as *entity id plus metrics*, which is not specific to
one vendor but is an assumption: a server that reports metrics some other way is checked
only by the weaker pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation

_NUMBER = re.compile(r"(?<![\w.\-])\$?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%?)")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

#: ``- `spend`: $1,281.33`` and its unquoted variants: a metric named, then its value.
_LABELLED = re.compile(
    r"[`*_]*([A-Za-z][A-Za-z0-9_ ]{2,30}?)[`*_]*\s*[:=]\s*\*{0,2}\$?"
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%?)"
)

#: An identifier long enough that a collision with a quantity is not a concern.
_ID = re.compile(r"\b\d{8,}\b")

def _blank_dates(text: str) -> str:
    """Blank out ISO dates without changing any character offset."""
    return _ISO_DATE.sub(lambda m: " " * len(m.group(0)), text)


#: A list item whose subject is the ad — `- \`<id>\` — name | spend: $X`, or `- Ad \`<id>\`
#: (Spend: …)`. A short label may precede the id; a sentence may not. Anything after the
#: id on such a line describes it, while a mention mid-prose does not.
_ITEM_SUBJECT = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(?:[A-Za-z#]{1,12}[.:]?\s+){0,3}[`*_\[]*\d{8,}")

#: A Markdown heading closes the block above it. Attribution is positional, so without
#: this an account-level `spend:` written under "### Baseline" is read as a claim about
#: whichever ad was named last — and `repair` then rewrites it to that ad's figure.
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")

#: Below this a *whole number* is structure rather than measurement — list numbering,
#: "top 5". It must not apply to anything with a decimal part: every `roas` and
#: `thumbstop_ratio` is under 10, and skipping them left the figures that drive the whole
#: ranking checked by neither pass.
MIN_INTERESTING = Decimal("10")


def _is_structure(value: Decimal) -> bool:
    return abs(value) < MIN_INTERESTING and value == value.to_integral_value()


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
        # Both rounding modes. `Decimal` rounds half to even, so 58.445 quantizes to
        # 58.44, while anyone writing a dollar figure — and every report formatter — gives
        # 58.45. Treating that as a contradiction fails an honest run.
        for mode in (ROUND_HALF_EVEN, ROUND_HALF_UP):
            try:
                if candidate.quantize(quantum, rounding=mode) == claimed:
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
        for match in _NUMBER.finditer(_blank_dates(source)):
            value = _dec(match.group(1), match.group(2))
            if value is not None:
                index.numbers.add(value)
        # `raw_decode`, not `loads`: a tool result carries a trailing annotation — the
        # derived-totals block this module appends — so the text is a JSON value followed
        # by prose. `loads` rejects the whole thing and silently indexes nothing, which
        # took the per-ad check down to the handful of ads seen only via detail calls.
        try:
            payload, _end = json.JSONDecoder().raw_decode(source.lstrip())
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
    #: Where the figure sits, so a correction replaces that figure and nothing else. A
    #: naive str.replace once rewrote digits inside an ad id that happened to contain the
    #: same two characters as a wrong purchase count.
    line_no: int = -1
    start: int = -1
    end: int = -1

    def __str__(self) -> str:
        where = f"{self.entity} " if self.entity else ""
        if self.actual is not None:
            return f"{where}{self.metric}: report says {self.raw}, tools said {self.actual}"
        return f"{self.raw} — {self.line.strip()[:80]}"


def _resolve(label: str, metrics: dict[str, Decimal]) -> str | None:
    """Map a label as written to a metric key.

    Reports shorten names — `Thumbstop:` for `thumbstop_ratio`, `CPA:` for
    `cost_per_purchase`. An unambiguous prefix is accepted; an ambiguous one is not, since
    guessing which metric a figure was filed under is how a correction becomes a new error.
    """
    name = label.strip().strip("*`_").lower().replace(" ", "_")
    if name in metrics:
        return name
    candidates = [key for key in metrics if key.startswith(name)]
    return candidates[0] if len(candidates) == 1 else None


def _figures(report: str, index: Index) -> tuple[list[Figure], list[Figure]]:
    """Split the report's numbers into labelled metrics and everything else."""
    labelled: list[Figure] = []
    loose: list[Figure] = []
    entity: str | None = None

    for line_no, line in enumerate(report.splitlines()):
        # Attribution has to be structural. A summary paragraph naming two ads and then
        # quoting the account's blended roas is not making a claim about either of them,
        # and taking "the last id on the line" filed all three figures under the second ad
        # and failed an honest run. So only a line that names exactly one ad, as a heading
        # or as the start of its own item, says which ad the figures belong to; a line
        # naming several says nothing and clears the subject rather than guessing.
        named = [found for found in _ID.findall(line) if found in index.metrics]
        if len(named) > 1:
            entity = None
        elif len(named) == 1 and (_HEADING.match(line) or _ITEM_SUBJECT.match(line)):
            entity = named[0]
        elif _HEADING.match(line):
            entity = None  # a heading with no ad in it closes the block above
        cleaned = _blank_dates(line)

        claimed_spans: set[tuple[int, int]] = set()
        for match in _LABELLED.finditer(cleaned):
            value = _dec(match.group(2), match.group(3))
            if value is None:
                continue
            name = _resolve(match.group(1), index.metrics.get(entity, {}) if entity else {})
            if name is None:
                continue
            # The figure alone, not the `name: value` pair: a correction rewrites the
            # number and leaves the label it was filed under intact.
            start, end = match.start(2), match.end()
            if line[start - 1 : start] == "$":
                start -= 1  # the currency mark belongs to the figure, not the label
            labelled.append(
                Figure(
                    value=value,
                    raw=line[start:end],
                    line=line,
                    entity=entity,
                    metric=name,
                    actual=index.metrics[entity][name],
                    line_no=line_no,
                    start=start,
                    end=end,
                )
            )
            claimed_spans.add(match.span(2))

        for match in _NUMBER.finditer(cleaned):
            value = _dec(match.group(1), match.group(2))
            if value is None or _is_structure(value):
                continue
            if match.span(1) in claimed_spans:
                continue
            loose.append(
                Figure(value, match.group(0), line, line_no=line_no,
                       start=match.start(), end=match.end())
            )
    return labelled, loose


@dataclass(frozen=True)
class Verdict:
    checked: int
    contradicted: list[Figure]
    unsupported: list[Figure]
    #: How many figures got the per-ad contradiction check rather than mere existence.
    #: Only figures written `metric: value` under a named ad can get it, so a report that
    #: lays its numbers out any other way is barely checked at all — and used to say so
    #: with the words "all 60 figures agree".
    attributed: int = 0
    #: Ads the report names that the tools also reported on. With none, weak-only is
    #: simply what this report is; with several, it means the format defeated the check.
    entities: int = 0

    @property
    def ok(self) -> bool:
        return not self.contradicted and not self.unsupported

    @property
    def weakly_checked_only(self) -> bool:
        """Named ads, and not one figure tied to any of them."""
        return self.entities > 0 and self.attributed == 0 and self.checked > 0

    def _coverage(self) -> str:
        weak = self.checked - self.attributed
        return (
            f"[verify] {self.attributed} of {self.checked} figures checked against the ad "
            f"they are filed under; {weak} checked only for existence somewhere"
        )

    def render(self) -> str:
        if not self.checked:
            return "[verify] no figures in the report to check"

        if self.weakly_checked_only:
            head = [
                f"[verify] WEAK — none of the {self.checked} figures could be tied to an "
                f"ad, though the report names {self.entities}.",
                "         Write metrics as `metric: value` under the ad they belong to; "
                "a figure with no label next to it",
                "         is only checked for appearing somewhere in the tool results, "
                "which a wrong ad's number does.",
            ]
        else:
            head = [self._coverage()] if self.attributed else []

        if self.ok:
            return "\n".join(head + [
                f"[verify] no figure disagrees with the tool results ({self.checked} checked)"
            ]) if head else (
                f"[verify] all {self.checked} figures agree with the tool results"
            )

        lines = list(head)
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
        if not self.weakly_checked_only:
            lines.append(self._coverage())
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
    named = {entity for entity in index.metrics if entity in ids_mentioned(report)}
    return Verdict(
        checked=len(labelled) + len(loose),
        contradicted=contradicted,
        unsupported=unsupported,
        attributed=len(labelled),
        entities=len(named),
    )


def sources_from_steps(steps: list[dict]) -> list[str]:
    """Everything the run was told, as flat text: tool results and the opening context.

    Prefer `sources_from_nodes` where nodes are available — a step cannot say whether it
    is a compaction boundary, and a boundary is not evidence. See that function.
    """
    out: list[str] = []
    for step in steps:
        kind = step.get("type")
        if kind == "function_result":
            out += [b.get("text", "") for b in (step.get("result") or [])]
        elif kind == "user_input":
            out += [b.get("text", "") for b in (step.get("content") or [])]
    return [text for text in out if text]


def sources_from_nodes(nodes: list[Any]) -> list[str]:
    """The same, minus anything the model wrote itself.

    A compaction boundary is a `user_input` node, so the flat view counts it as a source —
    and its brief is written by a model from a truncated history. A figure invented there
    then *validates itself* in the final report: the checker reports "all figures agree"
    for a number that appears in no tool output anywhere. That is the exact laundering
    this module exists to prevent, running in reverse.

    Evidence is what a tool returned and what a human asked for. Nothing the model wrote
    counts, however plausible it looks by the time it comes back around.
    """
    return sources_from_steps(
        [node.step for node in nodes if not node.meta.get("compact_boundary")]
    )


# ---- writing the numbers down for the next turn -------------------------------


#: Enough entities to carry a ranking forward without crowding out the prose sections.
MAX_MEASURED_ENTITIES = 25


def _plain(value: Decimal) -> str:
    """A number a reader can scan: no exponent, no float tail, no trailing zeros."""
    rounded = value.quantize(Decimal("0.0001")) if abs(value) < 10_000 else value.quantize(Decimal("0.01"))
    text = format(rounded.normalize(), "f")
    return text


def render_measurements(index: Index, mentioned: set[str], limit: int = MAX_MEASURED_ENTITIES) -> str:
    """The metrics section of a continuation brief, written from the tool results.

    Not model-written, and deliberately so. The summarizer is shown a history whose tool
    results are truncated to a fixed byte budget, so for a fifty-row ranking it is asked
    for "concrete metric numbers" while holding almost none of them — and it fills the gap
    from what it already knows about the brand. Numbers are the one part of a brief that
    can be transcribed rather than recalled, so they are.
    """
    chosen = [entity for entity in index.metrics if entity in mentioned][:limit]
    if not chosen:
        return ""

    lines = [
        "Transcribed by the harness from tool results, not recalled. These override any "
        "figure elsewhere in this brief.",
        "",
    ]
    for entity in chosen:
        metrics = index.metrics[entity]
        rendered = ", ".join(f"{name}={_plain(value)}" for name, value in metrics.items())
        lines.append(f"- `{entity}` — {rendered}")
    return "\n".join(lines)


def ids_mentioned(text: str) -> set[str]:
    return set(_ID.findall(text))


def repair(text: str, sources: list[str]) -> tuple[str, list[Figure]]:
    """Rewrite figures that contradict the tool results, in place.

    Correcting beats dropping: the finding is worth keeping and the true value is known,
    so there is nothing to decide. Only figures explicitly labelled with a metric name for
    an identified entity are touched — everything else is left exactly as written.
    """
    verdict = verify(text, sources)
    if not verdict.contradicted:
        return text, []

    lines = text.splitlines()
    fixed: list[Figure] = []
    by_line: dict[int, list[Figure]] = {}
    for figure in verdict.contradicted:
        if figure.actual is not None and figure.line_no >= 0:
            by_line.setdefault(figure.line_no, []).append(figure)

    for line_no, figures in by_line.items():
        line = lines[line_no]
        # Right to left, so each splice leaves the offsets to its left still valid.
        for figure in sorted(figures, key=lambda f: f.start, reverse=True):
            written = line[figure.start : figure.end]
            # Rewrite in the form it was written in: a ratio reported as a percentage is
            # corrected as a percentage, not silently rescaled.
            actual = figure.actual * 100 if written.endswith("%") else figure.actual
            correct = _plain(actual)
            if written.startswith("$"):
                correct = "$" + correct
            if written.endswith("%"):
                correct += "%"
            lines[line_no] = line = line[: figure.start] + correct + line[figure.end :]
            fixed.append(figure)

    repaired = "\n".join(lines)
    if text.endswith("\n"):
        repaired += "\n"  # a repair must not change the shape of the text it edits
    return repaired, fixed


# ---- quoted text --------------------------------------------------------------

#: Short enough to be a label, long enough that a coincidental match is not a worry.
MIN_QUOTE_CHARS = 25
#: Compared on a prefix, so a summarizer that shortens a long hook still matches.
QUOTE_PREFIX_CHARS = 40

_QUOTE = re.compile(r'["“]([^"“”\n]{%d,})["”]' % MIN_QUOTE_CHARS)
_PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-", "…": "..."})


def _normalize(text: str) -> str:
    return " ".join(text.translate(_PUNCT).lower().split())


def unverified_quotes(text: str, sources: list[str]) -> list[str]:
    """Quoted strings that appear verbatim in no tool result.

    The figures in a brief can be corrected because the true value is known. Quoted text
    cannot — so it is marked instead. This matters more than it sounds: on a hooks-only
    task the quotes *are* the deliverable, and a summarizer working from truncated tool
    results invented eight of nine, complete with a product the account does not sell.
    """
    blob = _normalize(" ".join(sources))
    missing: list[str] = []
    for quote in _QUOTE.findall(text):
        probe = _normalize(quote)[:QUOTE_PREFIX_CHARS]
        if probe and probe not in blob:
            missing.append(quote)
    return missing


UNVERIFIED_MARK = " [unverified: not found in any tool result]"


def mark_unverified_quotes(text: str, sources: list[str]) -> tuple[str, list[str]]:
    """Annotate quotes the record does not support, leaving the text itself intact."""
    missing = unverified_quotes(text, sources)
    for quote in missing:
        for closing in ('"', "”"):
            needle = quote + closing
            if needle in text:
                text = text.replace(needle, needle + UNVERIFIED_MARK, 1)
                break
    return text, missing


# ---- aggregates the model must not compute for itself -------------------------

#: Below this an aggregate is not an account summary, it is a coincidence. Detail calls
#: return one ad; a ranking returns a page of them.
MIN_ADS_FOR_TOTALS = 5

#: An ad that has not spent this many times the typical cost of a sale never had a fair
#: chance to make one, so its ratio metrics are noise rather than a small result.
SPEND_FLOOR_CPA_MULTIPLE = 2


def derive_totals(rendered: str) -> str | None:
    """Aggregate a tool result and hand the numbers back as part of that result.

    `10-system-rules.md` forbids stating a metric no tool returned, and the verifier
    enforces it, so an agent asked to apply a floor of "twice the account CPA" is asked
    for a figure it is not allowed to produce. Rather than carve an exception for
    arithmetic — which is exactly the licence that produced `$1,470.92` — the harness does
    the arithmetic and the model reads it the way it reads any other returned figure.

    Deliberately scoped to the response in hand. A `limit=50` call on a larger account
    yields the CPA of those fifty, and the block says so rather than implying otherwise.
    """
    index = index_sources([rendered])
    ads = [m for m in index.metrics.values() if m.get("spend") is not None]
    if len(ads) < MIN_ADS_FOR_TOTALS:
        return None

    spend = sum((m["spend"] for m in ads), Decimal(0))
    purchases = sum((m.get("purchases", Decimal(0)) for m in ads), Decimal(0))
    # Without both halves there is no CPA and no floor, and the rest is worse than
    # nothing: a creative-tag response carries spend but no conversions, and reporting
    # "blended roas 0" for it invites exactly the wrong conclusion.
    if spend <= 0 or purchases <= 0:
        return None

    revenue = sum((m["spend"] * m.get("roas", Decimal(0)) for m in ads), Decimal(0))
    parts = [
        f"ads in this response {len(ads)}",
        f"total spend {_plain(spend)}",
        f"total purchases {_plain(purchases)}",
        f"blended roas {_plain(revenue / spend)}",
    ]
    cpa = spend / purchases
    parts.append(f"account cpa {_plain(cpa)}")
    parts.append(
        f"spend floor ({SPEND_FLOOR_CPA_MULTIPLE}x cpa) "
        f"{_plain(cpa * SPEND_FLOOR_CPA_MULTIPLE)}"
    )
    return (
        "\n[harness-derived from this result, not returned by the API. These cover only "
        "the ads in this response, not the whole account:\n "
        + " | ".join(parts)
        + "]"
    )
