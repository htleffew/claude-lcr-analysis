# Intra-coder reliability re-code protocol

**Purpose:** measure coding stability for §6.1. Original coding completed 2026-05-19; washout interval ≥ 3 weeks has elapsed.

## Round 2 restart (2026-07-05)

Round 1 was abandoned after a handful of partially coded cases (never exported;
the coder had seen only the first few cases). Round 2 restarts from zero on a
revised instrument, per the coder's call, with three changes:

1. **vulnerability_disclosure restructured** (definitional): the released
   single property left the audience of the disclosure ambiguous (to the model,
   to the post's readers, or both). It is replaced by two non-mutually-exclusive
   binaries with drill-down routing:
   - `vulnerability_disclosed` (yes/no): the account discloses vulnerability
     existing prior to the directive, to ANY audience. This carries the
     impact-on-people reading: a vulnerable user is affected whether or not the
     model knew.
   - `vulnerability_disclosed_to_model` (yes/no; auto-null when the first is
     no): the account EXPLICITLY places the disclosure within the exchange with
     the model before the directive, so it was input the model received. Code
     `no` when the audience is indeterminate, adding an unclear flag. This
     carries the model's informational position, grounded in the property's
     released §3.2 purpose (whether LCR firing correlates with disclosure the
     model could react to), and the pair supports probing whether disclosure to
     readers travels with disclosure to the model.
   Pre-committed §6.1 mapping (decided before any round-2 coding): stability
   against v2 is computed on `vulnerability_disclosed` with v2 collapsed to
   binary ({yes, borderline} -> yes), since the released audience-unspecified
   construct is closest to the presence reading; `vulnerability_disclosed_to_model`
   has no v2 counterpart and is reported descriptively. Because round 2
   restarts in full, no case mixes definitions.
2. **Medium-case evidence excerpted** (presentation): each Medium-locator case
   now shows only the article passages around its cited source (and hint
   matches), not the whole article; the full text stays behind a fold. The
   `author_experience` case keeps the full article, since the whole account is
   its evidence.
3. **documented_impacts added** (new measure, multi-select): every impact of
   the directive the account itself documents, selected from
   emotional_distress, income_or_professional_harm, other_work_derailed,
   monetary_cost, workaround_or_time_burden, service_abandonment, with an
   exclusive none_documented anchor so an unanswered field is distinguishable
   from no documented impact. Codes documented consequences, not inferred
   severity. No v2 counterpart; reported descriptively (impact-on-people
   analysis, and joins with the vulnerability pair). Exported pipe-joined in
   one column.
4. **Fresh autosave key** (`lcr-recode-2026-06r4`), so earlier partial
   autosaves cannot leak into round 2. The blinded instrument CSV header now
   carries the two vulnerability columns and documented_impacts (original
   header preserved in git history).

The §6.1 write-up must disclose change 1: the re-code measures stability under
a restructured vulnerability property, not a byte-identical codebook, using the
pre-committed mapping above.

## Rules

1. Do NOT open `deliverables/lcr_cases_coded_v2.csv`, the paper §4, or your reflexive memos until the re-code is complete. Work only from the case evidence (the source posts / Medium article quotes identified by the source pointer in each row).
2. Code the 35 rows of `recode_instrument_blinded.csv` **in the R-XX order given** (deterministically shuffled; the R-XX to PC-LCR-XX mapping is in `recode_key_DO_NOT_OPEN_UNTIL_DONE.csv`).
3. Apply the §3.2 schema exactly as released:
   - `unsolicited`: yes / no
   - `weak_signal_type`: affective / temporal / session / topical / null
   - `pushback_documented`: yes / no
   - `pushback_response`: yielded / verbally_yielded_but_reissued / insisted / escalated / null
   - `restriction_direction`: restriction / autonomy_expansion / mixed
   - `cross_session_evidence`: cross_session / single_session
   - `mood`: declarative / modal / imperative / interrogative
   - `vulnerability_disclosed`: yes / no (any audience; round-2 restructure)
   - `vulnerability_disclosed_to_model`: yes / no / null (asked only when
     `vulnerability_disclosed` = yes; auto-null otherwise)
   - `documented_impacts` (round-2 addition, multi-select, exported
     pipe-joined): emotional_distress / income_or_professional_harm /
     other_work_derailed / monetary_cost / workaround_or_time_burden /
     service_abandonment / none_documented (exclusive)
4. Use `recode_notes` only for genuine uncertainty flags; do not consult original researcher_notes.
5. When done, tell the agent. Analysis: per-property percent agreement + Cohen's kappa (self), disagreement table, and a §6.1 paragraph reporting the results. Pre-commitment: results are reported regardless of outcome; disagreements are documented, not reconciled retroactively (the v2 codes remain the codes of record; the re-code measures stability only).

## Reporting language reserved for §6.1 (to be filled with actual numbers)

"As a partial check on coding stability, the author re-coded all thirty-five cases blind to the original assignments after a washout interval of N weeks, with case order shuffled and identifiers masked. Before the re-code began, the vulnerability-disclosure property was restructured into two binaries separating the presence of a disclosed vulnerability from its explicit disclosure to the model; stability on this property is computed against the presence binary with the original codes collapsed to binary, per the mapping pre-committed in the protocol, and the disclosed-to-model binary is reported descriptively as a new measure. Per-property agreement was X to Y% (Cohen's kappa Z.ZZ to Z.ZZ). [Disagreement summary.] Intra-coder stability does not substitute for independent inter-rater reliability, which remains future work."

## Source-pointer resolution (which document to read, NOT a code)

Two rows use descriptive shorthand rather than a raw `post_id`; verified 2026-07-02 so the recode has no dead references. This names the document to read, not its coding, so it does not break blinding:

- **R-05** `1nyspk_workaround`: the base id is a typo for **`1nyyspk`** (there is no `1nyspk` in the corpus). Read the *workaround* segment of that thread (`data/lcr_corpus_intact_seed_tagged.csv`, post_id `1nyyspk`). R-33 codes a different segment of the same thread; code each on its own evidence.
- **R-09** `charlie_kirk_post`: the Charlie Kirk assassination / silent-LCR thread, corpus post **`1nhsouj`** (`data/lcr_corpus_intact_seed_tagged.csv`).

All 33 other pointers resolve directly by `post_id` (CSV rows) or by the quoted locator (Medium PDF rows).

## Labeling dashboard (recommended way to run the recode)

`recode_dashboard.html` is a single-file labeling UI generated by
`build_recode_dashboard.py` from the blinded instrument plus the case evidence
sources. It shows one case per screen (evidence text beside the 8 schema fields
as click buttons), autosaves to browser localStorage AND, when served through
the hub, to the hub itself (`/state` on the same URL; the hub persists the blob
here as `recode_state_autosave.json`, one rolling `.prev` backup kept). It
embeds source text only, never the v2 codes, the key, or researcher notes, so
blinding holds.

- Use it through the hub URL: autosave then syncs server-side (the header shows
  `hub-saved HH:MM:SS`), on load the newest save wins across devices, and
  offline coding still lands in localStorage and pushes on the next change.
  Opened as a bare file (`file://`) there is no hub sync; stay on one
  browser/device and use Backup JSON to move progress.
- Corpus rows embed the full post body; Medium rows embed the full article text
  with the locator shown, use the highlight box to find the passage.
- The section 3.2 definitions are embedded: a codebook panel at the top (opens on
  a fresh start) and a definitions fold next to each property's buttons, with
  where-to-judge guidance per property. pushback_response is asked only when
  pushback_documented = yes and auto-codes null otherwise (the schema's own
  logic). Each property has an unclear flag with free text; flagged fields still
  take a best-supported value, and the flags export into recode_notes as
  `field=unclear: note` segments.
- Each case also has a one-tap discuss flag (no text needed) for cases
  complicated enough to talk through with the model AFTER coding closes (never
  mid-recode; that would put an agent in the blinded loop). Flagged cases get a
  purple marker on their nav chip, a count in the header, and a
  `flagged_for_discussion` segment in recode_notes on export.
- The dashboard inherits the estate hub design system (the vendored Heather
  Leffew token layer): obsidian/charcoal, Playfair Display, Lora, JetBrains
  Mono, square corners, the same status colors.
- When all 35 are coded, nothing else is needed if you coded through the hub:
  on the save that completes the round the dashboard ships the finished CSV
  along with its state, and the hub lands it here as
  `recode_instrument_blinded_FILLED.csv` automatically (the header confirms
  with "filled CSV exported to project"; the coding registry flips to
  exported). Fallbacks: `py -3.13 export_filled_from_state.py` builds the same
  CSV from the synced state, and the dashboard's Export button still downloads
  one manually.
- To rebuild after an instrument change: `py build_recode_dashboard.py`. The hub
  route below reads the file at request time, so a rebuild is served immediately.

### Coding from the phone (tailnet, via the estate hub)

The Everything Heather hub serves the dashboard auth-gated at
`/api/research/coding/lcr-recode` (it reads the project file directly; no copy).
On the Pixel (already on the tailnet):

1. Open `http://100.97.15.107:4318` and log in to the hub once (30-day cookie).
2. Open `http://100.97.15.107:4318/api/research/coding/lcr-recode` and code
   (the older `/api/research/lcr-recode` URL redirects there).

The hub route is a registry (`CODING_DASHBOARDS` in the estate-hub
`app/server/index.mjs`), so future coding tasks, in this project or any other,
each get their own slug; a dashboard's `TASK_ID` in its generator keeps
per-task autosave separate on the shared hub origin.

Through the hub URL, progress syncs server-side on every change (each click
persists within about a second), so a phone dying mid-case costs at most the
current click, and any device resumes from the newest save on open. Completing
case 35 lands the filled CSV in the project automatically; no download, upload,
or agent step.
