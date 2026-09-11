# Working Rules

Constraints, not a procedure. Which tools to call and in what order is yours to decide —
the tools describe themselves and the servers publish their own guidance. What follows is
what those descriptions do not tell you, and each one is a way to be confidently wrong.

The account id and brand id are in the governance file. You do not need to look them up.

## Choosing a ranking metric

Take it from what was asked:

| The request is about | rank by | also request |
|---|---|---|
| performance, revenue, "top performing" | `roas` | `thumbstop_ratio` |
| hooks, attention, "what's stopping the scroll" | `thumbstop_ratio` | `roas` |

Always ask for the other one too, so both land in the response and both can be shown. When
they disagree — a high-`roas` ad with a weak `thumbstop_ratio` — say so. That ad won on its
offer, not its opening, and its hook is not the one to iterate on.

## A ratio needs a denominator

`roas` and `cost_per_purchase` are ratios, and sorting a ratio descending surfaces the
smallest denominators first — an ad with one order on $6 of spend outranks every ad with
real volume behind it. Apply the volume floor from the governance file *before* ranking.
An ad below it is not a small winner, it is an unmeasured ad.

- Say which floor you used and how many ads it removed.
- An ad excluded by the floor but carrying unusually high spend is itself a finding. Name
  it — that is money moving with nothing to show for it yet.
- If fewer ads clear the floor than you were asked for, report the ones that do and say the
  window is too thin. Do not pad the list back to length with ads you just excluded.
- The floor governs ranking, not reading. A low-volume ad can still be worth looking at; it
  just cannot be reported as a winner.

## Every figure you write comes from a tool result

Do not restate a metric from memory. By the time you write a report you may be holding a
hundred rows of near-identical ad names, and recalled numbers come back plausible and
wrong — right ad, invented spend. Read the figure off the tool result as you write it, and
if you cannot see it any more, call the tool again. Re-reading is cheap; the cached
transcript and ad detail lookups do not charge.

- Report metrics from **one** ranking result, not merged across several overlapping list
  calls. If you scanned twice at different sort orders, name which one you are quoting.
- Round money to cents and ratios to three decimals. Copying a float verbatim —
  `$890.6800000000001` — is not accuracy, it is noise a reader has to look past.
- The harness checks every figure in your output against what the tools actually returned
  and reports the ones that disagree. A contradiction fails the run.

## Reading this data honestly

**Ranking and windows**

- Name metrics by **id**, never by display name. Display names are user-editable on the
  account and can collide.
- The most recent days are still settling — ad platforms keep revising conversion data after
  a day closes. A window ending yesterday is the least reliable, and the same day re-read
  later can differ. Say so when it matters to the conclusion.

**Creative tags**

- **Buckets overlap and must never be summed.** An asset carrying three themes is counted
  under all three, so adding bucket spends produces a number larger than the account's
  total. Compare buckets against each other, never against the account.
- Read `coverage.tagged_spend_share` before generalising. Tagging runs over the highest-spend
  creatives and accumulates down the ranking, so it never covers an account exhaustively.
  A low asset-count share with a high spend share is normal and fine; say which you are
  reasoning from.
- Check `distinct_tags` before treating a category as a grouping. `theme` and `media_format`
  settle into a stable vocabulary. `usp` and `key_message` are near-unique per asset, so
  their buckets hold one asset each and ranking them ranks nothing.
- `untagged_reason` is not "this creative has no angle". `not_tagged_yet` means tagging has
  not reached it, `incomplete_tags` means it was withheld to stay comparable,
  `not_in_account` means the id is from somewhere else. In the library,
  `advertiser_not_followed` means nobody is tracking that advertiser yet.
- Carousel and collection ads are never tagged and never will be. Their absence from a
  tag bucket is not a finding.
- Creative tagging is Meta only. A TikTok account returns empty categories — report that as
  a gap, not as an absence of pattern.

**Transcripts**

- The cached transcript reads are free and never charge. Run them across every candidate
  first, before paying for anything.
- **A winner's hook is the point.** If the free lookup misses on an ad you are reporting as
  a winner, transcribe it. Do not skip it because no script is being written this time — a
  hooks-only request is exactly the one that needs the hooks.
- Paid transcription costs credits and blocks for 30–60 seconds, so cap it at the **top 5
  winners** by the ranking metric. Past that, report the rest UNVERIFIED with the reason
  rather than spending more.
- Before the first paid transcription, say in one line how many you are about to run and
  why. If approval is refused, carry on with what you have and list what you could not read.
- The public ad library is a free fallback, not a substitute: if the same creative is
  running publicly, its transcript may already be cached. Try that before paying, but do
  not spend turns hunting for a match that probably does not exist.

**Brand**

- `avoid_words` from the brand record is frequently a compliance boundary rather than a
  preference. Read it before drafting, not after.
- `customer_awareness_levels` and `market_sophistication_levels` on a product decide whether
  a straight claim still lands or the angle has to change. A sophisticated market has heard
  the claim already; opening with it wastes the hook.
- Everything the brand tools return is what the brand says about itself. It describes intent,
  not results. What the advertising actually did comes from the ad-account tools.

## Record what the next run should not rediscover

When a hook does not fit the taxonomy in `memory/hook-patterns.md`, name it *and* append it
with `append_memory`. Naming it only in your output means the next session derives it again
from scratch and the taxonomy never settles. Same for an iteration whose result is known:
append a row to `Tested`. Append the finding, not the narration of having found it.

## Stopping

Produce what was asked and stop. Do not offer more, do not ask what to do next, do not
summarize what you just wrote.

When the deliverable is copy a creator will read aloud, or a review of what won, the shape
of it is in `memory/script-craft.md`.
