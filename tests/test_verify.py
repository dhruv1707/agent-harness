"""Grounding the report in the tool results.

A run once produced a clean, confident ad review in which five of seven ads carried
spend and ROAS figures that appeared in no tool output — `$1,470.92` for an ad the API
reported at `$29.02`. Right about which ads won, wrong about every number beside them.
"""

import json

from harness.verify import (
    UNVERIFIED_MARK,
    ids_mentioned,
    index_sources,
    mark_unverified_quotes,
    render_measurements,
    repair,
    sources_from_steps,
    unverified_quotes,
    verify,
)

# Shaped like a bridged Atria response: an HTTP envelope whose body is a JSON string.
ADS = json.dumps(
    {
        "status_code": 200,
        "body": json.dumps(
            {
                "data": {
                    "items": [
                        {
                            "platform_ad_id": "6897420294631",
                            "ad_name": "a",
                            "metrics": {
                                "spend": 29.019999999999996,
                                "purchases": 2.0,
                                "roas": 8.210199862164025,
                                "thumbstop_ratio": 0.3247863247863248,
                            },
                        },
                        {
                            "platform_ad_id": "52539049653635",
                            "ad_name": "b",
                            "metrics": {
                                "spend": 196.53,
                                "purchases": 4.0,
                                "roas": 4.568055767567293,
                            },
                        },
                    ]
                }
            }
        ),
    }
)


def test_a_fabricated_metric_is_caught_against_the_ad_it_is_filed_under():
    report = (
        "#### 1. `6897420294631` — some ad\n"
        "- `spend`: $1,470.92\n"
        "- `purchases`: 3.0\n"
    )
    verdict = verify(report, [ADS])

    assert not verdict.ok
    reported = {(f.metric, str(f.value)) for f in verdict.contradicted}
    assert ("spend", "1470.92") in reported
    assert ("purchases", "3.0") in reported


def test_the_true_figures_pass_at_the_precision_the_report_used():
    """29.019999999999996 written as $29.02 is the same number, not a discrepancy."""
    report = (
        "#### 1. `6897420294631` — some ad\n"
        "- `spend`: $29.02\n"
        "- `roas`: 8.21\n"
        "- `thumbstop_ratio`: 0.325\n"
    )
    assert verify(report, [ADS]).ok


def test_a_ratio_written_as_a_percentage_is_the_same_figure():
    report = "#### `6897420294631`\n- `thumbstop_ratio`: 32.5%\n"
    assert verify(report, [ADS]).ok


def test_a_right_number_filed_under_the_wrong_ad_is_caught():
    """Existence checking cannot see this: 196.53 is real, but for a different ad."""
    report = "#### `6897420294631`\n- `spend`: $196.53\n"
    verdict = verify(report, [ADS])

    assert [f.metric for f in verdict.contradicted] == ["spend"]


def test_a_figure_from_no_tool_at_all_is_reported_as_unsupported():
    verdict = verify("Total account spend for the window was $88,412.10.", [ADS])

    assert not verdict.ok
    assert [f.raw for f in verdict.unsupported] == ["$88,412.10"]


def test_small_integers_are_structure_not_measurement():
    """List numbering and 'top 5' would otherwise drown the signal."""
    assert verify("#### 1. first\n#### 2. second\nTop 5 ads.\n", [ADS]).ok


def test_dates_are_not_treated_as_quantities():
    assert verify("Window: 2026-09-04 to 2026-09-10\n", [ADS]).ok


def test_prose_with_no_figures_is_not_a_failure():
    verdict = verify("No ads cleared the volume floor this window.", [ADS])
    assert verdict.ok and verdict.checked == 0


def test_sources_come_from_tool_results_and_the_opening_context():
    steps = [
        {"type": "user_input", "content": [{"type": "text", "text": "ctx"}]},
        {"type": "function_call", "name": "x", "arguments": {}},
        {"type": "function_result", "name": "x", "result": [{"type": "text", "text": "out"}]},
        {"type": "model_output", "content": [{"type": "text", "text": "report"}]},
    ]
    assert sources_from_steps(steps) == ["ctx", "out"]


def test_a_compacted_session_is_checked_against_what_it_was_ever_told():
    """Compaction starts a new root, so the live path no longer reaches the tool results
    the report quoted. Auditing against the path alone calls every one of them invented."""
    from harness.transcript import Transcript

    transcript = Transcript.create("verify-compaction")
    try:
        transcript.append({"type": "user_input", "content": [{"type": "text", "text": "go"}]})
        transcript.append(
            {"type": "function_result", "name": "ads", "result": [{"type": "text", "text": ADS}]}
        )
        transcript.compact_boundary("COMPACT SUMMARY\nTask: ads reviewed.", retained=[])

        report = "#### `6897420294631`\n- `spend`: $29.02\n"
        assert len(transcript.steps()) < len(transcript.all_steps())
        assert not verify(report, sources_from_steps(transcript.steps())).ok
        assert verify(report, sources_from_steps(transcript.all_steps())).ok
    finally:
        transcript.path.unlink(missing_ok=True)


# ---- correcting a brief before it becomes the session's memory -----------------


def test_a_correction_rewrites_the_figure_and_nothing_else():
    """The first repair replaced digits by substring search and silently rewrote an ad
    id that happened to contain the same two characters as a wrong purchase count."""
    brief = "- Ad `6897420294631` (Spend: $1,470.92, Purchases: 3)\n"
    fixed, changed = repair(brief, [ADS])

    assert "`6897420294631`" in fixed, "the identifier must survive intact"
    assert "$29.02" in fixed and "$1,470.92" not in fixed
    assert "Purchases: 2" in fixed
    assert {f.metric for f in changed} == {"spend", "purchases"}


def test_a_ratio_written_as_a_percentage_is_corrected_as_one():
    brief = "- Ad `6897420294631` (Thumbstop: 29.4%)\n"
    fixed, _ = repair(brief, [ADS])

    assert "32.4786%" in fixed, "corrected in the form it was written in, not rescaled"


def test_an_abbreviated_metric_label_still_resolves():
    """Reports write `Thumbstop:` for `thumbstop_ratio`; an unambiguous prefix counts."""
    assert repair("- Ad `6897420294631` (Thumbstop: 29.4%)\n", [ADS])[1]


def test_an_ambiguous_label_is_left_alone():
    """`cost` prefixes two metrics here, and guessing which is how a fix becomes a bug."""
    sources = [
        json.dumps(
            {
                "body": json.dumps(
                    {
                        "data": {
                            "platform_ad_id": "6897420294631",
                            "metrics": {"cost_per_purchase": 14.51, "cost_per_lead": 3.2},
                        }
                    }
                )
            }
        )
    ]
    assert repair("- Ad `6897420294631` (Cost: $99.00)\n", sources)[1] == []


def test_nothing_is_rewritten_when_the_figures_are_right():
    brief = "- Ad `6897420294631` (Spend: $29.02, Purchases: 2)\n"
    assert repair(brief, [ADS]) == (brief, []), "unchanged text, shape included"


def test_measurements_are_transcribed_for_the_ads_the_brief_names():
    index = index_sources([ADS])
    table = render_measurements(index, ids_mentioned("we looked at `6897420294631` closely"))

    assert "`6897420294631`" in table and "spend=29.02" in table
    assert "52539049653635" not in table, "only what the brief actually refers to"


def test_measurements_are_empty_when_the_brief_names_no_entity():
    assert render_measurements(index_sources([ADS]), set()) == ""


def test_an_invented_quote_is_marked_and_a_real_one_is_not():
    real = "I thought the only way to fix my 2 a.m. hot flashes was hormone therapy."
    sources = [ADS, json.dumps({"body": json.dumps({"data": {"transcript": real}})})]
    brief = f'Hook: "{real}"\nHook: "The #1 Pet Bed for Humans. As seen on Shark Tank."\n'

    marked, missing = mark_unverified_quotes(brief, sources)

    assert missing == ["The #1 Pet Bed for Humans. As seen on Shark Tank."]
    assert marked.count(UNVERIFIED_MARK) == 1
    assert f'"{real}"' + UNVERIFIED_MARK not in marked


def test_a_shortened_quote_still_matches_its_source():
    """Summarizers truncate; a prefix match keeps that from reading as fabrication."""
    real = "Okay, let's do the math. $189 divided by 365 nights equals $0.52 per night."
    sources = [json.dumps({"body": json.dumps({"data": {"transcript": real}})})]

    assert unverified_quotes('Hook: "Okay, let\'s do the math. $189 divided by 365"', sources) == []


# ---- the whole compaction path ------------------------------------------------


def test_summarize_corrects_the_brief_it_gets_back():
    """End to end: the summarizer reads a history whose tool results were truncated, so
    whatever it writes is checked and corrected against the untouched steps before it
    becomes the session's memory."""
    import asyncio
    from types import SimpleNamespace

    from harness.compaction import summarize

    fabricated = (
        "## Task\nReview ads.\n"
        "## Findings\n- Ad `6897420294631` (Spend: $1,470.92, Purchases: 3, "
        'Thumbstop: 29.4%): Hook: "The #1 Pet Bed for Humans. As seen on Shark Tank."\n'
        "## Measurements\n- Ad `6897420294631` — spend=$9,999.00\n"
        "## Examined\nlist_ad_account_ads\n"
        "## Failed approaches\nNone.\n"
        "## Current state\nRanked.\n"
        "## Next\nWrite the report.\n"
    )

    class FakeClient:
        def __init__(self):
            step = SimpleNamespace(
                type="model_output", content=[SimpleNamespace(text=fabricated, parts=None)]
            )
            response = SimpleNamespace(steps=[step], output_text="")
            self.aio = SimpleNamespace(
                interactions=SimpleNamespace(create=self._create(response))
            )

        @staticmethod
        def _create(response):
            async def create(**_kwargs):
                return response
            return create

    steps = [
        {"type": "user_input", "content": [{"type": "text", "text": "top ads"}]},
        {"type": "function_call", "name": "list_ad_account_ads", "arguments": {}},
        {"type": "function_result", "name": "list_ad_account_ads",
         "result": [{"type": "text", "text": ADS}]},
    ]
    brief = asyncio.run(summarize(FakeClient(), "fake-model", steps))

    findings = brief.sections["Findings"]
    assert "$29.02" in findings and "$1,470.92" not in findings
    assert "Purchases: 2" in findings
    assert "`6897420294631`" in findings, "the identifier survives the correction"
    assert UNVERIFIED_MARK in findings, "the invented hook is marked"

    measurements = brief.sections["Measurements"]
    assert "9,999" not in measurements, "the model's version is discarded, not merged"
    assert "spend=29.02" in measurements and "roas=8.2102" in measurements
