"""Grounding the report in the tool results.

A run once produced a clean, confident ad review in which five of seven ads carried
spend and ROAS figures that appeared in no tool output — `$1,470.92` for an ad the API
reported at `$29.02`. Right about which ads won, wrong about every number beside them.
"""

import json

from harness.verify import sources_from_steps, verify

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
