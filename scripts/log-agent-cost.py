#!/usr/bin/env python3
"""Append one line per child agent to `runs/agent-costs.csv`.

A `subagent_stop` hook, and pure observation — it never blocks, because what a run cost is
not grounds for refusing it.

It exists because a team run's cost is spread across sessions whose tails nobody sees. The
CLI prints a summary and then the terminal scrolls, so "what did last week's runs actually
cost" has no answer. This gives it one.
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "runs" / "agent-costs.csv"

COLUMNS = (
    "when", "parent_session", "agent_id", "role", "stop_reason", "is_error",
    "input_tokens", "cached_tokens", "output_tokens", "thought_tokens",
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    usage = payload.get("usage") or {}
    row = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent_session": payload.get("parent_session", ""),
        "agent_id": payload.get("agent_id", ""),
        "role": payload.get("agent_type", ""),
        "stop_reason": payload.get("stop_reason", ""),
        "is_error": payload.get("is_error", False),
        "input_tokens": usage.get("total_input_tokens", ""),
        "cached_tokens": usage.get("total_cached_tokens", ""),
        "output_tokens": usage.get("total_output_tokens", ""),
        "thought_tokens": usage.get("total_thought_tokens", ""),
    }

    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
