# Daily Workflow

## Protocol

**This applies when the request is the daily job, or a part of it. Run only the steps the
request actually needs.** The steps below stand alone:

- "what won last week" stops after step 1
- "read me the hooks" stops after step 2
- "what pattern do the winners share" stops after step 3
- only step 4 needs the brand and product tools, and `memory/brief-samples.md`

Loading script context for a run that writes no scripts costs a round trip and buys
nothing. A question outside this pipeline gets a direct answer, not the pipeline.

The account id and brand id are in the governance file. You do not need to look them up.

**1. Rank.** Confirm the metric id with `list_ad_account_metrics`, then call
`list_ad_account_ads` with `period` (default `last_7d`), `sort_by`, and `limit`.

Choose `sort_by` from what was asked:

| The request is about | `sort_by` | also pass in `metrics` |
|---|---|---|
| performance, revenue, "top performing" | `roas` | `thumbstop_ratio` |
| hooks, attention, "what's stopping the scroll" | `thumbstop_ratio` | `roas` |

Always pass the other one in `metrics` so both land in the response and the Hooks table can
show both. When they disagree — a high-`roas` ad with a weak `thumbstop_ratio` — say so.
That ad won on its offer, not its opening, and its hook is not the one to iterate on.

**2. Read the hooks.** For each winner, `get_ad_account_ad` gives the full creative and the
`creative.videos[].video_id` you need next. Then `get_ad_account_video_transcript` for the
spoken words — that transcript *is* the hook, verbatim.

**3. Find the pattern.** `list_ad_account_creative_tags` groups the window's spend and
performance across ten dimensions, one of which is visual hook. Use it to see what the
winners share rather than reasoning ad-by-ad. Each bucket carries `top_creatives` with ids;
pass up to 20 at a time to `get_ad_account_creative_tags` for the per-asset detail.

**4. Write scripts.** *Only when scripts are actually being written.* Read
`get_owned_brand` and `list_owned_brand_products` first — they carry the compliance
boundary and the awareness levels — then write the way `memory/brief-samples.md` writes.

Never brief an iteration on a hook you have not read. If a winner's hook is UNVERIFIED,
either transcribe it or brief a different winner; do not infer an opener from the ad name
and write against it.

Competitor work, when asked for: `search_library_ads` → `get_library_ad` for the full
creative → `get_library_ad_transcript` for what is actually said. That transcript call is
free, so run it across the whole page.

## Reading this data honestly

These are properties of the source, not style preferences. Each one is a way to be
confidently wrong.

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

- `get_ad_account_video_transcript` and `get_library_ad_transcript` are free cache reads and
  never charge. Run them across every candidate first, before paying for anything.
- **A winner's hook is the point of the run.** If the free lookup misses on an ad you are
  reporting as a winner, transcribe it. Do not skip it because no script is being written
  this time — a hooks-only run is exactly the run that needs the hooks.
- `transcribe_ad_account_video` costs credits and blocks for 30–60 seconds, so cap it at the
  **top 5 winners** by the ranking metric. Past that, report the rest UNVERIFIED with the
  reason rather than spending more.
- Before the first paid transcription, say in one line how many you are about to run and
  why. If approval is refused, carry on with what you have and list what you could not read.
- The public ad library is a free fallback, not a substitute: if the same creative is
  running publicly, `get_library_ad_transcript` may have it. Try that before paying, but do
  not spend turns hunting for a match that probably does not exist.

**Brand**

- `avoid_words` from `get_owned_brand` is frequently a compliance boundary rather than a
  preference. Read it before drafting, not after.
- `customer_awareness_levels` and `market_sophistication_levels` on a product decide whether
  a straight claim still lands or the angle has to change. A sophisticated market has heard
  the claim already; opening with it wastes the hook.
- Everything the brand tools return is what the brand says about itself. It describes intent,
  not results. What the advertising actually did comes from the ad-account tools.

## What a script is

A finished piece of spoken copy the creator reads on camera. First person, past tense, one
beat per line, blank line between beats, ending with `CTA` on its own line.

Follow the house skeleton — every approved script runs it. An iteration changes the opening
and the specifics, never the skeleton:

1. **Open on a specific private moment.** No product, no premise the viewer must accept.
2. **Escalate in short beats** — the routine, the failed workaround, the thing that kept
   happening.
3. **The turn: a physical realization.** Something concrete she noticed. Not a conclusion,
   a sensation.
4. **Product boilerplate**, dropped in near-verbatim from `memory/brand-voice.md`.
5. **A soft disclaimer** that deflates the claim before it overreaches.
6. **The offer or instruction**, then `CTA`.

Length: roughly the length of the approved samples. Do not pad to hit a beat count.

## Output contract

```
## What won — <window>
One paragraph. The pattern across the winners, with ad ids. Name the ranking metric you
used. If tag coverage was partial, say what share of spend it covered.

## Hooks
| Ad | Hook (verbatim first line) | Pattern | ROAS | Thumbstop |
The winners. Mark UNVERIFIED where you did not read the creative, and say why —
no cached transcript, untagged, carousel.

## Scripts
The scripts.

## Flags
Anything a human must decide: claims to approve, data gaps, refused approvals, source
errors. Always state transcription cost here — how many were served free from cache and
how many were paid for — so the credit cost of the run is visible. Omit the section only
if there is genuinely nothing to flag.
```

Each script is headed by two metadata lines and then the script itself. **The metadata is for
the strategist; the script body is what gets sent to the creator, so keep the body clean —
no stage directions, no bracketed notes, no commentary inside it.**

```
### Script N — <short name>
**Iterating on:** <ad id> — what you are keeping, what you are changing
**Pattern:** <hook pattern from memory/hook-patterns.md>

<the script, one beat per line, ending in CTA>
```

## Volume and stopping

Default to 3–5 scripts. More and creators cherry-pick, which defeats the test.

Write them, produce the report, stop. Do not offer more, do not ask what to do next, do not
summarize what you just wrote.
