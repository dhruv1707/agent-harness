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
real volume behind it. Apply both floors from the governance file *before* ranking. An ad
below them is not a small winner, it is an unmeasured ad.

The floors and the account baseline are returned to you. Every ad-listing result carries a
`[harness-derived …]` block with `account cpa`, `spend floor` and `blended roas`. Use those
figures; do not work them out yourself.

They cover the ads in *that response* only. A CPA averaged over the first fifty rows of a
`roas` ranking is not the account's CPA — it is the CPA of the fifty cheapest-to-convert
ads, and a floor calibrated on it is biased low. Derive the baseline from a listing sorted
by `spend` and deep enough that the returned `count` is under your limit. If you still
cannot see the whole account, say so and say what the floor is therefore approximate
against.

## Report in two bands, not one ranking

One sort cannot answer two questions. "Which hooks are proven?" and "which look promising?"
are different, and a single ROAS ranking answers neither well — a report once led with a
$29 ad and covered 6% of the account's spend.

**Band A — proven at scale.** Cleared both floors. Rank by the metric the request calls
for. These are the only hooks with enough behind them to brief a creator against.

**Band B — efficient but unproven.** Cleared the purchase floor, under the spend floor.
Report them, and say plainly that they are not yet evidence. A high ratio here means the
ad has barely run, not that it is winning.

Two things go with Band A every time:

- **Where the money is.** Name the top three by spend regardless of their ratios. These
  are the account's actual bets, and ranking by a ratio buries them — the three largest
  creatives in one window sat 17th, 20th and 21st. If one of them is underperforming the
  baseline, that is the most important line in the report.
- **The baseline.** State the blended account ROAS. A 4.02 means nothing until the reader
  knows the account runs at 1.51.

Also:

- Say which floors you used and how many ads each removed.
- An ad excluded by the floors but carrying unusually high spend is itself a finding. Name
  it — that is money moving with nothing to show for it yet.
- A thin Band A is a finding, not a reason to stop. Report what cleared, say the window is
  too thin, and do not pad the band with ads you just excluded. The stop-and-ask trigger in
  the operating rules is about the window returning almost nothing, not about one band.
- The floors govern ranking, not reading. A low-volume ad can still be worth looking at; it
  just cannot be reported as proven.

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
- **Write every metric as `` `metric`: value `` on a line under the ad it belongs to.**
  Not a bare table of columns, not `` `roas` 8.210 `` with no colon. The harness checks
  each figure against the record of the ad it is filed under, and it can only find the ad
  a figure belongs to when the figure is labelled. A number with no label beside it is
  checked merely for appearing *somewhere* in the results — which another ad's spend does.
  A report once passed with 60 figures and none of them actually tied to an ad.
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
- Paid transcription costs credits and blocks for 30–60 seconds, so cap it at **5 paid
  transcriptions per run**, spent on Band A first — those are the hooks anyone will act on.
  Band B hooks are worth reading when the cached lookup is free and worth marking
  UNVERIFIED when it is not. Past the cap, report the rest UNVERIFIED with the reason
  rather than spending more. This is a budget for *transcribing*, not a limit on how many
  ads to report.
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

When the deliverable is copy a creator will read aloud, **or a review of what won**, read
`memory/script-craft.md` before writing it — it holds the section order. Reading a memory
file is free. The rules above outrank it: where its layout disagrees with how figures must
be written, the layout loses.
