# Project Governance

Team norms and constraints for this account. This file outlives any single run. It is
loaded on every run, so keep it short — detail belongs in `memory/` topic files.

## The account

- **Brand:** Hugl — U-shaped cooling body pillow, DTC
- **Primary efficiency metric:** `roas` — the default ranking. Use `thumbstop_ratio`
  ("Thumbstop (Hook rate)") instead when the request is about hooks or attention, since
  that isolates the first three seconds from everything downstream.
- **Default lookback window:** 7 days
- **Data sources:** Atria only. Triple Whale was removed — its MCP access is gated on a
  plan entitlement this store does not have.
- **Atria ad account id:** `1fb10475eb5b493391f5ffd439ed7fa2` (Meta, USD,
  America/Los_Angeles). This is Atria's internal UUID, not the `act_…` platform id.
- **Atria brand id:** `d186047e117e4d2c830a6224d3ccbeec`

## Working agreements

- Scripts go to creators, not to the client. Write for the person holding the camera.
- One pattern per batch. A batch of scripts that tests five unrelated ideas teaches nothing.
- A hook we have already iterated on twice with no lift is retired — check
  `memory/hook-patterns.md` before briefing it again.
- Claims that are not already approved get flagged, never written into a script.

## Conventions

- Ads are referenced as `<ad_id> — <ad_name>`, always both.
- Windows are stated explicitly as dates, never as "last week".
- Metrics are named by id, never by display name — display names are user-editable on the
  account and can collide.
