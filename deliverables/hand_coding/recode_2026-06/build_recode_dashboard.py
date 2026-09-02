"""Build the blinded intra-coder recode dashboard for claude-lcr section 6.1.

Reads recode_instrument_blinded.csv plus the case evidence sources and emits
recode_dashboard.html, a single-file labeling UI (one case per screen, click
buttons for the 8 schema fields, autosave to localStorage, CSV export in the
exact instrument format). It embeds source text only: never the v2 codes, the
sealed key, or researcher notes, so blinding holds.

Reusable: point INSTRUMENT, CORPUS, and PDF at another round's files and
adjust SCHEMA to rebuild the same dashboard for a different coding task.

Run: python build_recode_dashboard.py
"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# Unique per coding task. It becomes the browser localStorage key, so two
# dashboards served from the same origin (the hub) never collide; change it
# when reusing this generator for another round or another project.
# r4: fresh key for the 2026-07-05 restart (vulnerability split into two
# routed binaries, documented_impacts multi-select added, excerpted medium
# evidence), so earlier incompatible partial autosaves cannot leak in.
TASK_ID = "lcr-recode-2026-06r4"
INSTRUMENT = HERE / "recode_instrument_blinded.csv"
CORPUS = REPO / "data" / "lcr_corpus_intact_seed_tagged.csv"
PDF = REPO / "prior_artifacts" / "leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf"
OUT = HERE / "recode_dashboard.html"

POINTER_FIXES = {
    "1nyspk_workaround": ("1nyyspk", "workaround segment of this thread; R-33 codes a different segment of the same thread, code each on its own evidence"),
    "charlie_kirk_post": ("1nhsouj", "the Charlie Kirk assassination / silent-LCR thread"),
}

# Definitions transcribe the released section 3.2 schema paragraph of the
# manuscript (case-independent; nothing from section 4 or the coded CSV).
# depends_on encodes a logical drill-down: the field is asked only when the
# named controller field holds the given value; otherwise it auto-codes to
# else_value and folds away.
CODEBOOK_INTRO = ("Apply the section 3.2 schema to this case's evidence only. The directive is the "
                  "quoted Claude utterance issuing psychiatric-style assessment, professional-help "
                  "direction, or wellness intervention; the user account is the poster's surrounding "
                  "narrative. Definitions for every label sit next to its buttons (unfold 'definitions'). "
                  "Use the unclear flag with a short note only for genuine uncertainty, and still choose "
                  "the best-supported value.")

SCHEMA = [
    {"key": "unsolicited", "label": "unsolicited",
     "values": ["yes", "no"],
     "definition": "Unsolicited issuance: whether the user requested psychiatric assessment, professional-help direction, or wellness intervention prior to the directive's appearance.",
     "applies": "Judge from the user account of the exchange before the directive appeared.",
     "value_defs": {
         "yes": "The user did not request assessment, help-direction, or wellness intervention; the directive appeared unrequested.",
         "no": "The user requested or invited such assessment or direction before the directive."}},
    {"key": "weak_signal_type", "label": "weak_signal_type",
     "values": ["affective", "temporal", "session", "topical", "null"],
     "definition": "Weak inferential signal: the model's apparent grounds for issuing the directive.",
     "applies": "Judge from what the case evidence shows preceded the directive (the user account plus any quoted exchange).",
     "value_defs": {
         "affective": "Grounds appear to be emotional or affective content in the user's messages.",
         "temporal": "Grounds appear to be time cues, such as late hours or hours spent working.",
         "session": "Grounds appear to be session length, such as a long conversation or a token threshold.",
         "topical": "Grounds appear to be the subject matter under discussion.",
         "null": "No identifiable signal preceded the directive in the user's account."}},
    {"key": "pushback_documented", "label": "pushback_documented",
     "values": ["yes", "no"],
     "definition": "Whether the case material documents the user pushing back against the directive.",
     "applies": "Judge from the exchange after the directive in the case evidence.",
     "value_defs": {
         "yes": "The user is documented contesting, correcting, or objecting to the directive.",
         "no": "No user pushback appears in the case material."}},
    {"key": "pushback_response", "label": "pushback_response",
     "values": ["yielded", "verbally_yielded_but_reissued", "insisted", "escalated", "null"],
     "definition": "Refusal to yield under user pushback: how the model responded to the documented pushback.",
     "applies": "Judge from the model's documented behavior after the user pushed back.",
     "depends_on": {"field": "pushback_documented", "value": "yes", "else_value": "null",
                    "else_reason": "no pushback documented, so this property is null by definition"},
     "value_defs": {
         "yielded": "The model dropped the directive after pushback.",
         "verbally_yielded_but_reissued": "The model conceded in wording but issued the directive again.",
         "insisted": "The model maintained the directive without escalation.",
         "escalated": "The model strengthened the directive or stacked new attributions after pushback.",
         "null": "Auto-coded when no user pushback is documented."}},
    {"key": "restriction_direction", "label": "restriction_direction",
     "values": ["restriction", "autonomy_expansion", "mixed"],
     "definition": "Asymmetric restriction direction: whether the directive moved the conversation toward control and restriction or toward autonomy expansion.",
     "applies": "Judge from the directive's effect on the interaction as the case evidence describes it.",
     "value_defs": {
         "restriction": "Moved toward control, restriction, refusal, or capability withdrawal.",
         "autonomy_expansion": "Moved toward expanding the user's autonomy or options.",
         "mixed": "Contains both restricting and autonomy-expanding movement."}},
    {"key": "cross_session_evidence", "label": "cross_session_evidence",
     "values": ["cross_session", "single_session"],
     "definition": "Cross-session persistence: whether the user's account describes the behavior as confined to a single session or as recurring across new conversations.",
     "applies": "Judge from the user's whole account, including mentions of other sessions or repeated attempts.",
     "value_defs": {
         "cross_session": "The account describes the behavior recurring across separate sessions or new conversations.",
         "single_session": "The account confines the behavior to one session."}},
    {"key": "mood", "label": "mood",
     "values": ["declarative", "modal", "imperative", "interrogative"],
     "definition": "Grammatical mood of the directive utterance.",
     "applies": "Judge from the quoted directive text itself, not from the user's description of it.",
     "value_defs": {
         "declarative": "Stated as an assertion or observation.",
         "modal": "Framed with modal verbs such as should, might, or may.",
         "imperative": "Framed as a command or instruction.",
         "interrogative": "Framed as a question."}},
    # Restructured 2026-07-05 (round 2): the released vulnerability_disclosure
    # property split into two non-mutually-exclusive binaries, so impact on
    # vulnerable people (presence, any audience) and the model's informational
    # position (explicit disclosure in the exchange) are coded separately, and
    # disclosure-to-readers vs disclosure-to-model can be compared. The second
    # binary is asked only when the first is yes (drill-down, auto-null).
    {"key": "vulnerability_disclosed", "label": "vulnerability_disclosed",
     "values": ["yes", "no"],
     "definition": ("Whether the user's account discloses vulnerability, such as mental-health history "
                    "or distress, existing prior to the directive, to ANY audience: it counts whether the "
                    "disclosure sits in the narrative addressed to the post's readers or inside the quoted "
                    "exchange with the model."),
     "applies": "Judge from the user's whole account of what preceded the directive, narrative and quoted exchange alike.",
     "value_defs": {
         "yes": "The account discloses vulnerability that existed before the directive, to any audience.",
         "no": "No vulnerability disclosure appears anywhere in the account, or the user explicitly disclaims one."}},
    {"key": "vulnerability_disclosed_to_model", "label": "vulnerability_disclosed_to_model",
     "values": ["yes", "no", "null"],
     "definition": ("Whether that vulnerability was EXPLICITLY disclosed to the model: the account places "
                    "the disclosure within the user's exchange with the model before the directive appeared, "
                    "so it was input the model received and could have reacted to."),
     "applies": "Judge from the account of the exchange itself; explicit means the account places the disclosure in the conversation, not merely in the surrounding narrative.",
     "depends_on": {"field": "vulnerability_disclosed", "value": "yes", "else_value": "null",
                    "else_reason": "no vulnerability disclosed at all, so disclosure to the model is null by definition"},
     "value_defs": {
         "yes": "The account explicitly places the vulnerability disclosure within the exchange with the model before the directive.",
         "no": "The vulnerability appears only in narrative addressed to readers, or the account does not place it in the exchange; code no when the audience is indeterminate and add an unclear flag.",
         "null": "Auto-coded when no vulnerability is disclosed at all."}},
    # Added 2026-07-05 (round 2): documented impact of the directive on the
    # user, multi-select. Codes the account's own testimony of consequences,
    # one selection per documented impact kind; none_documented is exclusive.
    {"key": "documented_impacts", "label": "documented_impacts",
     "values": ["emotional_distress", "income_or_professional_harm", "other_work_derailed",
                "monetary_cost", "workaround_or_time_burden", "service_abandonment",
                "none_documented"],
     "multi": True, "exclusive": "none_documented",
     "definition": ("Documented impact of the directive on the user, selected from the account's own "
                    "testimony of consequences (select every impact the account documents; this codes "
                    "documented consequences, not inferred severity)."),
     "applies": "Judge from the user's account of what followed or was attributed to the directive; select all that apply, or none_documented.",
     "value_defs": {
         "emotional_distress": "The account describes emotional or psychological distress attributed to the directive (upset, feeling gaslit, shaken, afraid to keep using the tool).",
         "income_or_professional_harm": "The account ties the directive to harm to income-making ability or professional work (blocked deliverable, missed deadline, damaged client or job work).",
         "other_work_derailed": "Other work or projects derailed, including work not clearly tied to income (studies, creative projects, personal projects).",
         "monetary_cost": "A direct monetary cost is documented (wasted tokens or API spend, subscription paid for refused work).",
         "workaround_or_time_burden": "Documented time or effort spent building workarounds or redoing work to get around the behavior.",
         "service_abandonment": "The account documents quitting, canceling, downgrading, or avoiding the service because of the behavior.",
         "none_documented": "The account documents no impact of these kinds (exclusive with every other selection)."}},
]


def load_instrument():
    with open(INSTRUMENT, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [dict(zip(header, r)) for r in reader if r and r[0].strip()]
    return header, rows


def split_pointer(raw):
    m = re.match(r"^(\S+)\s*\((.+)\)\s*$", raw.strip())
    if m:
        return m.group(1), m.group(2)
    return raw.strip(), ""


def load_corpus():
    by_id = {}
    with open(CORPUS, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            by_id.setdefault(row["post_id"], []).append(row)
    return by_id


def extract_pdf_text():
    from pypdf import PdfReader
    reader = PdfReader(str(PDF))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        pages.append("[page %d]\n%s" % (i, page.extract_text() or ""))
    return "\n\n".join(pages)


def corpus_text(records):
    parts = []
    for i, r in enumerate(records, 1):
        head = "[record %d of %d] r/%s, %s" % (i, len(records), r["subreddit"], r["createdAt"])
        parts.append(head + "\n\n" + r["body"].strip())
    return "\n\n................\n\n".join(parts)


# Medium-case evidence excerpting. Each medium locator names the article's
# in-text citation of a source (subreddit + year-letter key) plus an optional
# episode hint. The coder sees the passages around every inline citation of
# that source (and around verbatim hint matches), not the whole article; the
# full text stays available behind a fold as context safety.
def medium_excerpts(article, locator_raw):
    m = re.match(r"^r/(\w+)\s+(20\d\d[a-z])(?:\s*\((.+)\))?\s*$",
                 locator_raw.strip(), re.I)
    if not m:
        return None  # e.g. author_experience: fall back to the full article
    sub, key, hint = m.group(1).lower(), m.group(2).lower(), (m.group(3) or "").strip()
    low = article.lower()
    anchors = [mm.start() for mm in
               re.finditer(r"\(r/" + re.escape(sub) + r",\s*" + re.escape(key), low)]
    if hint:
        anchors += [mm.start() for mm in re.finditer(re.escape(hint.lower()), low)]
    if not anchors:
        return None
    anchors.sort()
    spans = []
    for pos in anchors:  # quoted episodes precede their citation: window leans back
        s, e = max(0, pos - 1100), min(len(article), pos + 500)
        if spans and s <= spans[-1][1] + 200:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    parts = []
    for i, (s, e) in enumerate(spans, 1):
        s2 = article.rfind(". ", s, s + 250)
        if s2 != -1:
            s = s2 + 2
        e2 = article.find(". ", e - 250, e + 120)
        if e2 != -1:
            e = e2 + 1
        parts.append("[excerpt %d of %d for this locator]\n%s"
                     % (i, len(spans), article[s:e].strip()))
    return "\n\n................\n\n".join(parts)


def build_cases(rows, by_id, medium_text):
    cases = []
    problems = []
    for row in rows:
        raw = row["post_id"]
        base, hint = split_pointer(raw)
        if row["source_csv"].startswith("data/"):
            if base in POINTER_FIXES:
                base, fix_hint = POINTER_FIXES[base]
                hint = (hint + "; " if hint else "") + fix_hint
            records = by_id.get(base, [])
            if not records:
                problems.append("%s: no corpus rows for %s" % (row["recode_id"], base))
                text, label = "", raw
            else:
                text = corpus_text(records)
                label = "r/%s thread %s" % (records[0]["subreddit"], base)
            cases.append({"id": row["recode_id"], "kind": "corpus",
                          "label": label, "hint": hint, "text": text})
        else:
            text = medium_excerpts(medium_text, raw)
            if text is None and raw.strip() != "author_experience":
                problems.append("%s: no article excerpt anchors for locator %r"
                                % (row["recode_id"], raw))
            cases.append({"id": row["recode_id"], "kind": "medium",
                          "label": "Medium article, locator: " + raw,
                          "hint": hint, "text": text})
    return cases, problems


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LCR blinded recode</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,500;12..96,600;12..96,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
/* Palette and type now mirror the Phosphor token layer at
   apps/estate-hub/design-system/design-tokens.css (blue-black field,
   frost text, chrome blue accent, square corners). */
:root { --bg:#06080F; --panel:#0A0E1A; --ink:rgba(226,236,251,0.92); --fg1:#E2ECFB; --mut:rgba(226,236,251,0.62);
  --line:rgba(96,165,250,0.15); --soft:rgba(59,130,246,0.05); --inter:rgba(96,165,250,0.42);
  --acc:#3B82F6; --good:#22C55E; --goodsoft:rgba(34,197,94,0.14);
  --amber:#F59E0B; --ambersoft:rgba(245,158,11,0.12); --danger:#FB7185;
  --idea:#8B5CF6; --ideasoft:rgba(124,58,237,0.14);
  --font-display:'Bricolage Grotesque','Inter',system-ui,sans-serif;
  --font-body:'Inter',system-ui,sans-serif;
  --font-mono:'JetBrains Mono','SF Mono',Consolas,monospace; }
* { box-sizing:border-box; border-radius:0 !important; }
::selection { background:rgba(59,130,246,0.35); color:#fff; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.7 var(--font-body); }
header { position:sticky; top:0; z-index:5; background:var(--bg);
  border-bottom:1px solid var(--line); padding:12px 18px;
  display:flex; align-items:center; gap:18px; flex-wrap:wrap; }
h1 { font:600 19px/1.2 var(--font-display); margin:0; color:var(--fg1);
  letter-spacing:-0.015em; }
.sub { font:11px/1.5 var(--font-mono); color:var(--mut); letter-spacing:0.05em;
  padding:3px 9px; border:1px solid var(--line); }
.sub.sync-saving { color:var(--amber); border-color:var(--amber); background:var(--ambersoft); }
.sub.sync-saved { color:var(--good); border-color:var(--good); background:var(--goodsoft); }
.sub.sync-offline { color:var(--amber); border-color:var(--amber); background:var(--ambersoft); }
.prog { display:flex; align-items:center; gap:8px; flex:1; min-width:200px; }
.progbar { flex:1; height:6px; background:rgba(255,255,255,0.08); overflow:hidden; }
#progfill { height:100%; width:0; background:var(--good); transition:width .25s; }
#progtext { font:11px var(--font-mono); color:var(--mut); white-space:nowrap;
  letter-spacing:0.10em; text-transform:uppercase; }
.btn { font:12px var(--font-mono); padding:7px 12px;
  border:1px solid var(--inter); background:var(--panel); color:var(--ink); cursor:pointer; }
.btn.primary { background:var(--acc); border-color:var(--acc); color:#fff; }
.btn:hover { filter:brightness(1.2); }
.wrap { display:flex; gap:14px; padding:14px 18px; align-items:flex-start; }
nav { width:170px; flex:none; display:grid; grid-template-columns:repeat(5,1fr);
  gap:5px; position:sticky; top:70px; }
.chip { font:10px var(--font-mono); text-align:center; padding:6px 0;
  border:1px solid var(--line); background:var(--panel); color:var(--mut); cursor:pointer; }
.chip.part { background:var(--ambersoft); border-color:var(--amber); color:var(--amber); }
.chip.done { background:var(--goodsoft); border-color:var(--good); color:var(--ink); }
.chip.cur { outline:1px solid var(--fg1); }
.chip.disc { box-shadow:inset 3px 0 0 var(--idea); }
main { flex:1; display:flex; gap:14px; align-items:flex-start; min-width:0; }
.evidence { flex:11; background:var(--panel); border:1px solid var(--line);
  box-shadow:0 8px 32px rgba(0,0,0,0.30); padding:16px 18px; min-width:0; }
.coding { flex:9; background:var(--panel); border:1px solid var(--line);
  box-shadow:0 8px 32px rgba(0,0,0,0.30); padding:16px 18px; position:sticky; top:70px;
  max-height:calc(100vh - 90px); overflow-y:auto; }
.case-head { display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }
.case-id { font:600 21px var(--font-display); color:var(--fg1); }
#case-label { color:var(--mut); font:11px var(--font-mono); letter-spacing:0.05em; }
.case-hint { font-size:13px; color:var(--amber); margin:4px 0 8px; font-style:italic; }
#search { width:100%; font:12.5px var(--font-mono); padding:7px 10px; color:var(--ink);
  border:1px solid var(--inter); background:#03040A; margin-bottom:10px; }
.ev-text { white-space:pre-wrap; overflow-wrap:break-word; font-size:14.5px;
  line-height:1.75; color:var(--ink); max-height:calc(100vh - 240px); overflow-y:auto; }
.ev-text mark { background:rgba(219,168,68,0.45); color:#fff; }
.field { margin-bottom:15px; }
.fname { font:500 11px var(--font-mono); letter-spacing:0.10em;
  text-transform:uppercase; color:var(--fg1); margin-bottom:4px; }
.vals { display:flex; flex-wrap:wrap; gap:6px; }
.val { font:12px var(--font-mono); padding:6px 10px; color:var(--ink);
  border:1px solid var(--inter); background:transparent; cursor:pointer; }
.val:hover { border-color:var(--fg1); }
.val.sel { background:var(--acc); border-color:var(--acc); color:#fff; }
details.book { background:var(--panel); border:1px solid var(--line);
  box-shadow:0 8px 32px rgba(0,0,0,0.30); margin:12px 18px 0; padding:4px 16px; }
details.book[open] { padding-bottom:12px; }
details.book summary, details.fdef summary { cursor:pointer;
  font:500 11px var(--font-mono); letter-spacing:0.10em; text-transform:uppercase;
  color:#6C97CB; padding:8px 0; }
.book-intro { font-size:13.5px; margin:2px 0 10px; color:var(--ink); }
.book-field { margin:10px 0; }
.book-field h4 { margin:0 0 2px; font:500 11px var(--font-mono);
  letter-spacing:0.10em; text-transform:uppercase; color:var(--fg1); }
.defline { font-size:13px; margin:1px 0; color:var(--ink); }
.applies { font-size:12.5px; color:var(--amber); margin:1px 0; font-style:italic; }
.vdefs { margin:3px 0 0 16px; padding:0; font-size:12.5px; color:var(--ink); }
.vdefs li { margin:2px 0; }
.vdefs code { font:11px var(--font-mono); background:rgba(255,255,255,0.06);
  padding:1px 5px; color:var(--fg1); }
details.fdef { margin:3px 0 7px; }
details.fdef summary { font-size:10px; padding:2px 0; }
details.fdef .body { border-left:2px solid var(--line); padding:3px 0 3px 10px; margin:2px 0; }
.fhead { display:flex; align-items:baseline; gap:8px; }
.gated { font-size:12.5px; color:var(--mut); font-style:italic; margin:2px 0; }
.auto-line { font-size:12.5px; color:var(--good); margin:2px 0; }
.unclear { font:10px var(--font-mono); padding:3px 8px; letter-spacing:0.05em;
  border:1px dashed var(--inter); background:transparent; cursor:pointer; color:var(--mut); }
.unclear.on { border-color:var(--amber); color:var(--amber); background:var(--ambersoft); }
input.unclear-txt { width:100%; font:12.5px var(--font-mono); padding:5px 9px;
  border:1px dashed var(--amber); background:#03040A; color:var(--ink); margin-top:4px; }
.discuss { font:11px var(--font-mono); padding:7px 11px; letter-spacing:0.05em;
  border:1px dashed var(--inter); background:transparent; color:var(--mut);
  cursor:pointer; width:100%; text-align:left; margin:2px 0 12px; }
.discuss.on { border-color:var(--idea); color:var(--idea); background:var(--ideasoft); }
.nlabel { font:500 11px var(--font-mono); letter-spacing:0.10em;
  text-transform:uppercase; color:var(--fg1); }
textarea { width:100%; font:12.5px var(--font-mono); padding:7px 10px; color:var(--ink);
  border:1px solid var(--inter); background:#03040A; margin:5px 0 12px; resize:vertical; }
.navbtns { display:flex; justify-content:space-between; }
@media (max-width: 980px) { .wrap { flex-direction:column; }
  nav { position:static; width:100%; grid-template-columns:repeat(9,1fr); }
  main { flex-direction:column; } .coding { position:static; max-height:none; }
  .val { padding:10px 14px; font-size:14px; }
  .chip { padding:9px 0; font-size:12px; }
  .btn { padding:10px 14px; }
  .ev-text { max-height:52vh; } }
</style>
</head>
<body>
<header>
  <div>
    <h1>LCR blinded intra-coder recode</h1>
    <div class="sub">Code in R order from the evidence shown. Do not consult the v2 codes, paper section 4, or memos until all 35 are done.</div>
  </div>
  <div class="prog"><div class="progbar"><div id="progfill"></div></div><span id="progtext"></span></div>
  <span class="sub" id="sync" style="white-space:nowrap"></span>
  <div class="actions">
    <button class="btn" id="btn-backup">Backup JSON</button>
    <button class="btn" id="btn-restore">Restore</button>
    <input type="file" id="file-restore" accept=".json" style="display:none">
    <button class="btn primary" id="btn-export">Export filled CSV</button>
  </div>
</header>
<details class="book" id="codebook">
  <summary>Codebook: section 3.2 definitions (read before coding)</summary>
  <div class="book-intro" id="book-intro"></div>
  <div id="book-body"></div>
</details>
<div class="wrap">
  <nav id="nav"></nav>
  <main>
    <section class="evidence">
      <div class="case-head"><span class="case-id" id="case-id"></span><span id="case-label"></span></div>
      <div class="case-hint" id="case-hint"></div>
      <input id="search" placeholder="Type to highlight matching text in the evidence pane">
      <div class="ev-text" id="ev-text"></div>
      <details class="fdef" id="fullart" style="display:none">
        <summary>full article text (context safety net; the excerpts above are this case's evidence)</summary>
        <div class="ev-text" id="fullart-text"></div>
      </details>
    </section>
    <section class="coding">
      <div id="fields"></div>
      <button id="btn-discuss" class="discuss"></button>
      <label class="nlabel" for="notes">recode_notes (genuine uncertainty only)</label>
      <textarea id="notes" rows="3"></textarea>
      <div class="navbtns">
        <button class="btn" id="btn-prev">Prev</button>
        <button class="btn primary" id="btn-next">Next</button>
      </div>
    </section>
  </main>
</div>
<script>
const CASES = @@CASES@@;
const MEDIUM_TEXT = @@MEDIUM@@;
const SCHEMA = @@SCHEMA@@;
const META = @@META@@;
const HEADER = @@HEADER@@;
const LS_KEY = @@TASKID@@;
const INTRO = @@INTRO@@;

let state = {};
try { state = JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch (e) { state = {}; }
let savedAt = Number(localStorage.getItem(LS_KEY + ":savedAt")) || 0;

// Server-side autosave: when served through the hub, every save also pushes
// the whole state blob to /state on the same URL (debounced), and the hub
// persists it into the project. localStorage stays the offline fallback, so
// the phone and the desktop converge on whichever save is newest.
const STATE_URL = location.protocol.indexOf("http") === 0
  ? location.pathname.replace(/\/+$/, "") + "/state" : null;
let bootSynced = !STATE_URL;
let syncTimer = null;
function setSync(msg, cls) {
  const el = document.getElementById("sync");
  if (el) { el.textContent = msg; el.className = "sub" + (cls ? " " + cls : ""); }
}
// Persistence pill states: saving (amber), saved (green), offline (amber-bright).
function syncSaving() { setSync("saving…", "sync-saving"); }
function syncSaved() {
  setSync("hub-saved " + new Date(savedAt).toLocaleTimeString()
    + (pendingComplete ? " · filled CSV exported to project" : ""), "sync-saved");
}
function syncOffline() { setSync("offline: saved locally only", "sync-offline"); }
function summarize() { return { done: CASES.filter(c => isDone(c.id)).length, total: CASES.length }; }
let pendingComplete = false;
function pushState() {
  if (!STATE_URL || !bootSynced) return;
  // On a complete round the finished CSV rides along, so the export artifact
  // lands in the project automatically with no download or agent step.
  const sm = summarize();
  pendingComplete = sm.done === sm.total;
  const payload = { state: state, savedAt: savedAt, summary: sm };
  if (pendingComplete) payload.csv = buildCsv();
  syncSaving();
  fetch(STATE_URL, { method: "POST", headers: { "Content-Type": "application/json" },
    credentials: "include", body: JSON.stringify(payload) })
    .then(r => {
      if (!r.ok) throw new Error("HTTP " + r.status);
      syncSaved();
    })
    .catch(() => syncOffline());
}
function persistLocal() {
  localStorage.setItem(LS_KEY, JSON.stringify(state));
  localStorage.setItem(LS_KEY + ":savedAt", String(savedAt));
}
function save() {
  savedAt = Date.now();
  persistLocal();
  clearTimeout(syncTimer);
  syncTimer = setTimeout(pushState, 800);
}
// On load, adopt the hub save when it is newer than the local one (never
// adopt an empty blob over local work); push local up when local is newer.
// Pushes are held until this settles so a fresh browser can't clobber the hub.
function adoptServerState() {
  if (!STATE_URL) { setSync("local file: no hub sync", ""); return; }
  fetch(STATE_URL, { credentials: "include" })
    .then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
    .then(d => {
      bootSynced = true;
      const remoteAt = (d && d.savedAt) || 0;
      if (d && d.state && Object.keys(d.state).length && remoteAt > savedAt) {
        state = d.state; savedAt = remoteAt;
        persistLocal();
        cur = CASES.findIndex(c => !isDone(c.id));
        if (cur < 0) cur = 0;
        render();
        setSync("loaded hub save " + new Date(savedAt).toLocaleTimeString(), "sync-saved");
      } else if (Object.keys(state).length && savedAt > remoteAt) {
        pushState();
      } else {
        setSync("hub sync on", "sync-saved");
      }
    })
    .catch(() => { bootSynced = true; syncOffline(); });
}
function codes(id) { return state[id] || {}; }
// Multi-select fields store a value array (kept in schema order); the field's
// exclusive value clears the others and vice versa. Single fields toggle.
function setCode(id, k, v) {
  if (!state[id]) state[id] = {};
  const f = SCHEMA.find(s => s.key === k);
  if (f && f.multi) {
    let cur = Array.isArray(state[id][k]) ? state[id][k].slice() : [];
    if (cur.includes(v)) {
      cur = cur.filter(x => x !== v);
    } else if (v === f.exclusive) {
      cur = [v];
    } else {
      cur.push(v);
      if (f.exclusive) cur = cur.filter(x => x !== f.exclusive);
    }
    if (cur.length) { state[id][k] = f.values.filter(x => cur.includes(x)); }
    else { delete state[id][k]; }
  } else if (state[id][k] === v) { delete state[id][k]; } else { state[id][k] = v; }
  save();
}
function fieldCoded(id, f) {
  const v = codes(id)[f.key];
  return f.multi ? Array.isArray(v) && v.length > 0 : v !== undefined;
}
function isDone(id) { return SCHEMA.every(f => fieldCoded(id, f)); }
function nDone(id) { return SCHEMA.filter(f => fieldCoded(id, f)).length; }

// Drill-down consistency: a field with depends_on is asked only while its
// controller holds the trigger value; otherwise it auto-codes to else_value,
// and it un-codes when the controller is cleared or flips back.
function reconcileCase(id) {
  SCHEMA.forEach(f => {
    const dep = f.depends_on;
    if (!dep) return;
    const cv = codes(id)[dep.field];
    if (!state[id]) state[id] = {};
    if (cv === undefined) { delete state[id][f.key]; }
    else if (cv !== dep.value) { state[id][f.key] = dep.else_value; }
    else if (state[id][f.key] === dep.else_value) { delete state[id][f.key]; }
  });
  save();
}

function unsure(id) { return codes(id)._unsure || {}; }
function setUnsure(id, key, val) {
  if (!state[id]) state[id] = {};
  if (!state[id]._unsure) state[id]._unsure = {};
  if (val === null) { delete state[id]._unsure[key]; }
  else { state[id]._unsure[key] = val; }
  if (!Object.keys(state[id]._unsure).length) delete state[id]._unsure;
  save();
}

// The exported recode_notes column = free notes plus one segment per
// unclear-flagged field, so uncertainty lands in the instrument itself.
function composedNotes(id) {
  const c = codes(id);
  const parts = [];
  if (c.recode_notes) parts.push(c.recode_notes);
  const u = unsure(id);
  Object.keys(u).forEach(k => parts.push(k + "=unclear" + (u[k] ? ": " + u[k] : "")));
  if (c._discuss) parts.push("flagged_for_discussion");
  return parts.join("; ");
}
function caseText(c) {
  if (c.kind === "medium") return c.text != null ? c.text : MEDIUM_TEXT;
  return c.text;
}
function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

let cur = CASES.findIndex(c => !isDone(c.id));
if (cur < 0) cur = 0;

function renderNav() {
  const nav = document.getElementById("nav");
  nav.innerHTML = "";
  CASES.forEach((c, i) => {
    const d = document.createElement("div");
    const n = nDone(c.id);
    d.className = "chip" + (n === SCHEMA.length ? " done" : (n > 0 ? " part" : ""))
      + (i === cur ? " cur" : "") + (codes(c.id)._discuss ? " disc" : "");
    d.textContent = c.id.replace("R-", "");
    d.title = c.id + " (" + n + "/" + SCHEMA.length + ")";
    d.onclick = () => { cur = i; render(); };
    nav.appendChild(d);
  });
}

function renderProgress() {
  const done = CASES.filter(c => isDone(c.id)).length;
  const disc = CASES.filter(c => codes(c.id)._discuss).length;
  document.getElementById("progfill").style.width = (100 * done / CASES.length) + "%";
  document.getElementById("progtext").textContent =
    done + " / " + CASES.length + " complete" + (disc ? " · " + disc + " to discuss" : "");
}

function renderEvidence(c) {
  const q = document.getElementById("search").value.trim();
  let html = esc(caseText(c));
  if (q.length > 1) {
    const rx = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
    html = html.replace(rx, m => "<mark>" + m + "</mark>");
  }
  document.getElementById("ev-text").innerHTML = html;
}

const openDefs = new Set();

function defPanel(f) {
  const det = document.createElement("details");
  det.className = "fdef";
  if (openDefs.has(f.key)) det.open = true;
  det.addEventListener("toggle", () => {
    if (det.open) { openDefs.add(f.key); } else { openDefs.delete(f.key); }
  });
  const sum = document.createElement("summary");
  sum.textContent = "definitions";
  det.appendChild(sum);
  const body = document.createElement("div");
  body.className = "body";
  const defl = document.createElement("div");
  defl.className = "defline";
  defl.textContent = f.definition;
  body.appendChild(defl);
  const app = document.createElement("div");
  app.className = "applies";
  app.textContent = f.applies;
  body.appendChild(app);
  const ul = document.createElement("ul");
  ul.className = "vdefs";
  f.values.forEach(v => {
    const li = document.createElement("li");
    const code = document.createElement("code");
    code.textContent = v;
    li.appendChild(code);
    li.appendChild(document.createTextNode(" " + (f.value_defs[v] || "")));
    ul.appendChild(li);
  });
  body.appendChild(ul);
  det.appendChild(body);
  return det;
}

function renderFields(c) {
  const box = document.getElementById("fields");
  box.innerHTML = "";
  reconcileCase(c.id);
  SCHEMA.forEach(f => {
    const d = document.createElement("div");
    d.className = "field";
    const dep = f.depends_on;
    const cv = dep ? codes(c.id)[dep.field] : undefined;
    const gate = dep ? (cv === undefined ? "unset" : (cv === dep.value ? "active" : "auto")) : "active";

    const head = document.createElement("div");
    head.className = "fhead";
    const name = document.createElement("div");
    name.className = "fname";
    name.textContent = f.label;
    head.appendChild(name);
    const uBtn = document.createElement("button");
    const uOn = unsure(c.id)[f.key] !== undefined;
    uBtn.className = "unclear" + (uOn ? " on" : "");
    uBtn.textContent = uOn ? "unclear ✓" : "unclear?";
    uBtn.title = "Flag genuine uncertainty on this property; add a short note. Still choose the best-supported value.";
    uBtn.onclick = () => { setUnsure(c.id, f.key, uOn ? null : ""); render(); };
    head.appendChild(uBtn);
    d.appendChild(head);
    d.appendChild(defPanel(f));

    if (gate === "unset") {
      const g = document.createElement("div");
      g.className = "gated";
      g.textContent = "Asked only when " + dep.field + " = " + dep.value + ". Code " + dep.field + " first.";
      d.appendChild(g);
    } else if (gate === "auto") {
      const a = document.createElement("div");
      a.className = "auto-line";
      a.textContent = "auto-coded " + dep.else_value + " (" + dep.else_reason + ")";
      d.appendChild(a);
    } else {
      const vals = document.createElement("div");
      vals.className = "vals";
      const shown = dep ? f.values.filter(v => v !== dep.else_value) : f.values;
      const cval = codes(c.id)[f.key];
      shown.forEach(v => {
        const b = document.createElement("button");
        const sel = f.multi ? (Array.isArray(cval) && cval.includes(v)) : cval === v;
        b.className = "val" + (sel ? " sel" : "");
        b.textContent = v;
        b.title = f.value_defs[v] || "";
        b.onclick = () => { setCode(c.id, f.key, v); reconcileCase(c.id); render(); };
        vals.appendChild(b);
      });
      d.appendChild(vals);
      if (f.multi) {
        const m = document.createElement("div");
        m.className = "gated";
        m.textContent = "Select all that apply; " + f.exclusive + " stands alone.";
        d.appendChild(m);
      }
    }

    if (uOn) {
      const inp = document.createElement("input");
      inp.className = "unclear-txt";
      inp.placeholder = "why is " + f.label + " unclear here? (goes into recode_notes)";
      inp.value = unsure(c.id)[f.key] || "";
      inp.oninput = () => setUnsure(c.id, f.key, inp.value);
      d.appendChild(inp);
    }
    box.appendChild(d);
  });
  const notes = document.getElementById("notes");
  notes.value = codes(c.id).recode_notes || "";
  notes.oninput = () => {
    if (!state[c.id]) state[c.id] = {};
    state[c.id].recode_notes = notes.value;
    save();
  };
}

function render() {
  const c = CASES[cur];
  document.getElementById("case-id").textContent = c.id;
  document.getElementById("case-label").textContent = c.label;
  document.getElementById("case-hint").textContent = c.hint ? "Segment: " + c.hint : "";
  const dBtn = document.getElementById("btn-discuss");
  const dOn = !!codes(c.id)._discuss;
  dBtn.className = "discuss" + (dOn ? " on" : "");
  dBtn.textContent = dOn
    ? "flagged: talk this case through with the model after coding closes ✓"
    : "flag: complicated, talk through with the model after coding closes";
  dBtn.onclick = () => {
    if (!state[c.id]) state[c.id] = {};
    if (dOn) { delete state[c.id]._discuss; } else { state[c.id]._discuss = true; }
    save(); render();
  };
  // Medium cases with excerpted evidence keep the full article behind a fold.
  const fa = document.getElementById("fullart");
  fa.style.display = (c.kind === "medium" && c.text != null) ? "" : "none";
  fa.open = false;
  renderEvidence(c);
  renderFields(c);
  renderNav();
  renderProgress();
}

function step(delta) {
  cur = Math.min(CASES.length - 1, Math.max(0, cur + delta));
  document.getElementById("search").value = "";
  document.querySelector(".ev-text").scrollTop = 0;
  render();
}

function csvQuote(v) {
  v = v == null ? "" : String(v);
  return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
}

function buildCsv() {
  const lines = [HEADER.join(",")];
  META.forEach(m => {
    const c = codes(m.recode_id);
    const row = HEADER.map(h => {
      if (h === "recode_id") return csvQuote(m.recode_id);
      if (h === "source_csv") return csvQuote(m.source_csv);
      if (h === "post_id") return csvQuote(m.post_id);
      if (h === "recode_notes") return csvQuote(composedNotes(m.recode_id));
      const v = c[h];
      return csvQuote(Array.isArray(v) ? v.join("|") : (v || ""));
    });
    lines.push(row.join(","));
  });
  return lines.join("\n") + "\n";
}

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: type }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

document.getElementById("btn-export").onclick = () => {
  const missing = CASES.filter(c => !isDone(c.id)).map(c => c.id);
  if (missing.length &&
      !confirm("Incomplete: " + missing.join(", ") + "\nExport anyway?")) return;
  download("recode_instrument_blinded_FILLED.csv", buildCsv(), "text/csv");
};
document.getElementById("btn-backup").onclick = () =>
  download("recode_progress_backup.json", JSON.stringify(state, null, 1), "application/json");
document.getElementById("btn-restore").onclick = () =>
  document.getElementById("file-restore").click();
document.getElementById("file-restore").onchange = ev => {
  const f = ev.target.files[0];
  if (!f) return;
  f.text().then(t => { state = JSON.parse(t); save(); render(); });
};
document.getElementById("btn-prev").onclick = () => step(-1);
document.getElementById("btn-next").onclick = () => step(1);
document.getElementById("search").oninput = () => renderEvidence(CASES[cur]);
document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "TEXTAREA" || ev.target.tagName === "INPUT") return;
  if (ev.key === "ArrowLeft") step(-1);
  if (ev.key === "ArrowRight") step(1);
});

// Codebook: full schema reference at the top; opens on a fresh start so the
// definitions are read before the first label.
function buildCodebook() {
  document.getElementById("book-intro").textContent = INTRO;
  const body = document.getElementById("book-body");
  SCHEMA.forEach(f => {
    const d = document.createElement("div");
    d.className = "book-field";
    const h = document.createElement("h4");
    h.textContent = f.label + (f.depends_on ? " (asked only when " + f.depends_on.field + " = " + f.depends_on.value + ")" : "");
    d.appendChild(h);
    const defl = document.createElement("div");
    defl.className = "defline";
    defl.textContent = f.definition;
    d.appendChild(defl);
    const app = document.createElement("div");
    app.className = "applies";
    app.textContent = f.applies;
    d.appendChild(app);
    const ul = document.createElement("ul");
    ul.className = "vdefs";
    f.values.forEach(v => {
      const li = document.createElement("li");
      const code = document.createElement("code");
      code.textContent = v;
      li.appendChild(code);
      li.appendChild(document.createTextNode(" " + (f.value_defs[v] || "")));
      ul.appendChild(li);
    });
    d.appendChild(ul);
    body.appendChild(d);
  });
  if (CASES.every(c => nDone(c.id) === 0)) document.getElementById("codebook").open = true;
}
buildCodebook();
document.getElementById("fullart-text").textContent = MEDIUM_TEXT;
render();
adoptServerState();
</script>
</body>
</html>
"""


def main():
    header, rows = load_instrument()
    by_id = load_corpus()
    medium_text = extract_pdf_text()
    cases, problems = build_cases(rows, by_id, medium_text)
    meta = [{"recode_id": r["recode_id"], "source_csv": r["source_csv"],
             "post_id": r["post_id"]} for r in rows]
    html = (HTML_TEMPLATE
            .replace("@@CASES@@", json.dumps(cases))
            .replace("@@MEDIUM@@", json.dumps(medium_text))
            .replace("@@SCHEMA@@", json.dumps(SCHEMA))
            .replace("@@META@@", json.dumps(meta))
            .replace("@@HEADER@@", json.dumps(header))
            .replace("@@TASKID@@", json.dumps(TASK_ID))
            .replace("@@INTRO@@", json.dumps(CODEBOOK_INTRO)))
    OUT.write_text(html, encoding="utf-8")
    n_corpus = sum(1 for c in cases if c["kind"] == "corpus")
    n_medium = sum(1 for c in cases if c["kind"] == "medium")
    print("cases: %d (%d corpus, %d medium)" % (len(cases), n_corpus, n_medium))
    print("medium article text: %d chars" % len(medium_text))
    for c in cases:
        if c["kind"] == "corpus":
            print("  %s %s (%d chars)" % (c["id"], c["label"], len(c["text"])))
        elif c["text"] is not None:
            n_ex = c["text"].count("[excerpt")
            print("  %s %s -> %d excerpt(s), %d chars"
                  % (c["id"], c["label"], n_ex, len(c["text"])))
        else:
            print("  %s %s -> full article fallback" % (c["id"], c["label"]))
    print("wrote %s (%d bytes)" % (OUT, OUT.stat().st_size))
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  " + p)
        sys.exit(1)


if __name__ == "__main__":
    main()
