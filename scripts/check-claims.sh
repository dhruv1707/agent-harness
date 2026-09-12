#!/bin/sh
# A deterministic gate: reject any deliverable containing a claim the brand has not
# approved. Exits 2 with the reason, which the harness hands back to the child.
payload=$(cat)
for claim in "NASA-grade" "clinically proven" "thermoregulation"; do
  if printf '%s' "$payload" | grep -qi -- "$claim"; then
    >&2 printf '"%s" is not an approved claim. Remove it from the deliverable and move it to the Flags section for sign-off.' "$claim"
    exit 2
  fi
done
exit 0
