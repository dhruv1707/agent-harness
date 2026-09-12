#!/usr/bin/env python3
"""Refuse a deliverable that puts an unapproved claim in copy a creator would read aloud.

A `subagent_stop` hook. Exits 2 with the reason, which the harness hands back to the child
as its next instruction.

Two things this has to get right, and the naive version got both wrong:

**Scope to the role.** Armed against every child, it bounced a *researcher* asked to read
`brand-voice.md` — the file whose whole job is to list the phrases nobody may say. It was
correct by its own rules and useless in context. A worker reporting what it read is not
making the claim.

**Scope to the copy.** A deliverable that names a claim in its `## Flags` section, asking
for sign-off, is doing exactly what it should. Grepping the whole text punishes the correct
behaviour, so this reads only the script bodies — the words that would actually be spoken.
"""

import json
import re
import sys

#: Phrases `agent/memory/brand-voice.md` marks unconfirmed, plus the clinical register
#: `00-identity.md` bans outright. Kept short and literal on purpose: a hook earns its keep
#: by being certain, and anything needing judgement belongs to the verifier.
FORBIDDEN = (
    "NASA-grade",
    "clinically proven",
    "thermoregulation",
    "sleep architecture",
    "medical grade",
)

#: Only this role is making the claim. The rest are reporting on it.
GUARDED_ROLE = "implementer"

SCRIPT_BODY = re.compile(
    r"^###\s+Script\s+.*?$(.*?)^\s*CTA\s*$", re.MULTILINE | re.DOTALL
)


def spoken_copy(text: str) -> str:
    """Just the script bodies: what a creator would say on camera, nothing else."""
    return "\n".join(match.group(1) for match in SCRIPT_BODY.finditer(text))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # a hook that cannot read its input has no opinion

    if payload.get("agent_type") != GUARDED_ROLE or payload.get("is_error"):
        return 0

    copy = spoken_copy(payload.get("text") or "")
    if not copy.strip():
        return 0  # nothing spoken to check; the verifier reads shape, not this

    hits = [phrase for phrase in FORBIDDEN if phrase.lower() in copy.lower()]
    if not hits:
        return 0

    named = ", ".join(f'"{phrase}"' for phrase in hits)
    print(
        f"{named} appears in spoken script copy and is not approved in brand-voice.md. "
        "Take it out of the script and move it to the Flags section asking for sign-off. "
        "The rest of the deliverable is fine.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
