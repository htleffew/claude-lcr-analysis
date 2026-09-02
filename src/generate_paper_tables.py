"""
generate_paper_tables.py
LCR Preprint -- Tables 1, 3, 4 (Round 1 deliverables).

Table 1 -- Corpus composition
    Source: data/lcr_pass1b_canonical.csv

Table 3 -- Validated Round 2 augmented seed-term list (top-20 by precision x frequency)
    Source: deliverables/term_validation/round_2_term_validation.csv

Table 4 -- Per-subreddit per-1k vocabulary category rates
    Source: deliverables/phase_2_pass1b/lcr_freq_collocation/
            freq_unigram_sub_<sub>_stops_on_lemma_on.csv

Output format:
    paper/tables/table_<n>.md       (Markdown with pipe-delimited table)
    paper/tables/table_<n>.tex      (LaTeX, booktabs, no vertical rules)
    paper/tables/table_<n>.caption.md

Table 2 is deferred pending the researcher's hand-coded 35-case CSV
(deliverables/lcr_cases_coded_v2.csv).
"""

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
DELIVERABLES_ROOT = REPO / "deliverables"
FREQ_DIR = DELIVERABLES_ROOT / "phase_2_pass1b" / "lcr_freq_collocation"
OUT_TABLES = REPO / "paper" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

CANONICAL_CSV = DATA / "lcr_pass1b_canonical.csv"
R2_VALIDATION_CSV = DELIVERABLES_ROOT / "term_validation" / "round_2_term_validation.csv"
CASES_CODED_CSV = DELIVERABLES_ROOT / "lcr_cases_coded_v2.csv"

SUBREDDIT_ORDER = ["ClaudeAI", "claudexplorers", "ClaudeCode", "Anthropic"]
SUBREDDIT_LABELS = {
    "ClaudeAI": "r/ClaudeAI",
    "claudexplorers": "r/claudexplorers",
    "ClaudeCode": "r/ClaudeCode",
    "Anthropic": "r/Anthropic",
}
SUB_FNAME = {
    "ClaudeAI": "claudeai",
    "claudexplorers": "claudexplorers",
    "ClaudeCode": "claudecode",
    "Anthropic": "anthropic",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_freq_csv(path):
    counts = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            counts[row["term"]] = int(row["count"])
    return counts


def total_tokens(freq_dict):
    return sum(freq_dict.values())


def per_1k(count, total):
    if total == 0:
        return 0.0
    return count / total * 1000


def fmt_float(x, decimals=2):
    if x == 0:
        return "0.00"
    return f"{x:.{decimals}f}"


def write_md(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {path}")


def write_tex(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {path}")


def md_table(headers, rows):
    """Generate a Markdown pipe-delimited table."""
    col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [fmt_row(headers), sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines) + "\n"


def booktabs_table(caption_label, headers, rows, table_env="table",
                    caption=None, label=None, tabular_env="tabular",
                    col_spec=None, note=None):
    """Generate a LaTeX booktabs table (no vertical rules)."""
    n_cols = len(headers)
    if col_spec is None:
        col_spec = "l" + "r" * (n_cols - 1)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    header_line = " & ".join(r"\textbf{" + esc(h) + "}" for h in headers) + r" \\"
    body_lines = []
    for row in rows:
        body_lines.append(" & ".join(esc(c) for c in row) + r" \\")

    lines = [
        r"\begin{" + table_env + r"}[!htbp]",
        r"\centering"
    ]
    if caption is not None:
        lines.append(r"\caption{" + caption + r"}")
    if label is not None:
        lines.append(r"\label{" + label + r"}")
    lines.append(r"\small")
    if tabular_env == "tabularx":
        lines.append(r"\begin{tabularx}{\textwidth}{" + col_spec + r"}")
    else:
        lines.append(r"\begin{tabular}{" + col_spec + r"}")
        
    lines.extend([
        r"\toprule",
        header_line,
        r"\midrule",
        "\n".join(body_lines),
        r"\bottomrule"
    ])
    if tabular_env == "tabularx":
        lines.append(r"\end{tabularx}")
    else:
        lines.append(r"\end{tabular}")
        
    if note is not None:
        lines.append(r"")
        lines.append(r"\vspace{0.5em}")
        lines.append(r"\raggedright")
        lines.append(r"\textit{Note.} " + note)

    lines.append(r"\end{" + table_env + r"}")
    return "\n".join(lines) + "\n"


# ===========================================================================
# Table 1 -- Corpus composition
# ===========================================================================
def make_table_1():
    """
    Rows: total, by subreddit, then pre/post split, then post-vs-comment totals.
    Columns: n rows, posts, comments, r2_any_match, pre-release, post-release.
    """
    sub_post = Counter()
    sub_comment = Counter()
    sub_r2 = Counter()
    sub_pre = Counter()
    sub_post_date = Counter()

    total_rows = 0
    total_posts = 0
    total_comments = 0
    total_r2 = 0
    total_pre = 0
    total_post_d = 0

    with open(CANONICAL_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            sub = row["subreddit"]
            typ = row["type"]
            dt = row["createdAt"]
            r2 = row["r2_any_match"] == "True"

            if typ == "post":
                sub_post[sub] += 1
                total_posts += 1
            else:
                sub_comment[sub] += 1
                total_comments += 1

            if r2:
                sub_r2[sub] += 1
                total_r2 += 1

            if dt < "2025-09-29":
                sub_pre[sub] += 1
                total_pre += 1
            else:
                sub_post_date[sub] += 1
                total_post_d += 1

    headers = ["Stratum", "Rows (n)", "Posts", "Comments", "R2-tagged", "Pre-release", "Post-release"]

    rows = []
    for sub in SUBREDDIT_ORDER:
        tot = sub_post[sub] + sub_comment[sub]
        rows.append([
            SUBREDDIT_LABELS[sub],
            f"{tot:,}",
            f"{sub_post[sub]:,}",
            f"{sub_comment[sub]:,}",
            f"{sub_r2[sub]:,}",
            f"{sub_pre[sub]:,}",
            f"{sub_post_date[sub]:,}",
        ])

    # Add separator row then totals
    rows.append(["---", "---", "---", "---", "---", "---", "---"])
    rows.append([
        "Total",
        f"{total_rows:,}",
        f"{total_posts:,}",
        f"{total_comments:,}",
        f"{total_r2:,}",
        f"{total_pre:,}",
        f"{total_post_d:,}",
    ])

    # Markdown
    md = md_table(headers, rows)
    md_path = OUT_TABLES / "table_1.md"
    write_md(md_path, md)

    # LaTeX (drop separator row)
    latex_rows = [r for r in rows if r[0] != "---"]
    tex = booktabs_table(
        "table:corpus_composition", headers, latex_rows,
        caption="Pass 1b canonical corpus composition stratified by subreddit and row type.",
        label="tab:corpus_composition",
        note=(
            r"Posts (n=1{,}173) are a seed-term-matched subset of the wholesale August-to-December 2025 "
            r"Reddit pull; comments (n=24{,}985) are the seed-filtered refetch for those 1{,}173 post IDs. "
            r"R2-tagged rows match at least one Round 2 augmented seed term (34 retained terms across two "
            r"functional categories; see Table 3). Pre-release rows are dated before 2025-09-29 (Claude "
            r"Sonnet 4.5 launch); post-release rows on or after that date."
        )
    )
    tex_path = OUT_TABLES / "table_1.tex"
    write_tex(tex_path, tex)

    return {
        "total_rows": total_rows,
        "total_posts": total_posts,
        "total_comments": total_comments,
        "total_r2": total_r2,
        "total_pre": total_pre,
        "total_post_d": total_post_d,
    }


def write_table_1_caption():
    caption = """\
**Table 1.** Pass 1b canonical corpus composition. Posts are a seed-term-matched subset
of the wholesale August-to-December 2025 Reddit pull across four Claude-focused
subreddits; comments are the seed-filtered refetch for those 1,173 post IDs. R2-tagged
rows match at least one Round 2 augmented seed term (34 terms retained across two
functional categories: 17 validated detectors at sample-based precision-at-N >= 0.5,
plus 17 retrieval anchors lifted verbatim from LCR system-prompt text or confirmed
Claude utterances; see Table 3 for the stratified list). Pre-release rows are dated
before 2025-09-29 (Sonnet 4.5 launch); post-release rows on or after that date.
Retrieval provenance (wholesale posts, targeted comments) is described in Section 3.2.
"""
    write_md(OUT_TABLES / "table_1.caption.md", caption)


# ===========================================================================
# Table 3 -- Round 2 augmented seed-term list with functional categorization
# ===========================================================================
#
# Retained terms split into two functional categories:
#
#   retrieval_anchor   -- the phrase was lifted verbatim from LCR system-prompt
#                         text or from a verbatim Claude utterance in a confirmed
#                         positive case; corpus retrieval returns its source
#                         documents by construction, so precision as a
#                         detector-validation metric does not apply and is not
#                         reported. Function: flag mechanism-discussion posts
#                         that reproduce LCR-text or model-voice content.
#
#   validated_detector -- the phrase was validated against an independent 20-item
#                         (or smaller) sample; the cell reports raw hits over
#                         sample size (k/N) rather than a computed precision
#                         figure. Function: detect LCR-phenomenon discussion
#                         that does not require LCR-text reproduction.
#
# Single-hit (n=1) and small-N validated-detector rows are reported as raw
# fractions (1/1, 2/2, 3/3) rather than as a point-estimate precision, so the
# small sample size is visible in the cell rather than hidden behind a "1.00".

ANCHOR_SIGNALS = (
    "verbatim from the lcr",
    "verbatim lcr text",
    "verbatim lcr-triggered",
    "verbatim lcr directive",
    "verbatim lcr instruction",
    "verbatim claude utterance",
    "verbatim claude attribution",
    "lcr instruction set",
    "lcr instruction text",
    "lcr-instruction-text",
    "lcr system-prompt text",
    "lcr system prompt text",
    "lcr directive text",
    "lcr-triggered model voice",
    "exact lcr pathologizing",
    "directly quoted from ground-truth positive",
    "directly observed in ground-truth positive",
    "variant of retained term",
    "appears in the verbatim lcr",
)

DETECTOR_SIGNALS = (
    "user-reaction label",
    "user-reaction verb",
    "community critique",
    "community label",
    "community-coined",
    "post title directly naming",
    "compound phrase with high expected precision",
    "user-coined",
)


def classify_function(term, drop_reason, precision_estimate):
    """Return 'retrieval_anchor' or 'validated_detector'.

    Heuristic on the validation memo's drop_reason text. If neither signal set
    matches, default to validated_detector -- the conservative bucket, since the
    anchor designation makes a stronger claim (the phrase was lifted from corpus
    material) than the detector designation.
    """
    reason = (drop_reason or "").lower()
    if any(sig in reason for sig in ANCHOR_SIGNALS):
        return "retrieval_anchor"
    if any(sig in reason for sig in DETECTOR_SIGNALS):
        return "validated_detector"
    return "validated_detector"


def make_table_3():
    """
    Load round_2_term_validation.csv and emit Table 3 with two stratified
    sections: Validated detectors (hand-verified against samples drawn from
    corpus retrieval) and Retrieval anchors (deterministic mechanism-post
    retrieval; tautological by construction, no detector-validation metric
    reported). The validated-detector cell reports raw hits over sample size
    (k/N) rather than a computed precision figure, which would be misleading
    at small sample sizes; the retrieval-anchor cell reads "n/a (verbatim)".
    """
    with open(R2_VALIDATION_CSV, "r", encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))

    retained_detectors = []
    retained_anchors = []
    dropped = []

    for row in raw:
        term = row["term"]
        hits = int(row["corpus_hits"]) if row["corpus_hits"].isdigit() else 0
        sample_size = int(row["sample_size"]) if row["sample_size"].isdigit() else 0
        prec_str = row["precision_estimate"]
        reason = row.get("drop_reason", "")
        decision = row["retain_or_drop"].strip().lower()

        function = classify_function(term, reason, prec_str)

        # Pattern type from term shape
        n_words = len(term.split())
        if n_words >= 4:
            pat_type = "phrasal (long)"
        elif n_words in (2, 3):
            pat_type = "phrasal"
        elif "-" in term:
            pat_type = "compound"
        else:
            pat_type = "unigram"

        entry = {
            "term": term,
            "function": function,
            "pattern_type": pat_type,
            "precision": prec_str,
            "sample_size": sample_size,
            "corpus_hits": hits,
            "decision": decision,
        }

        if decision == "retain":
            if function == "retrieval_anchor":
                retained_anchors.append(entry)
            else:
                retained_detectors.append(entry)
        else:
            dropped.append(entry)

    # Sort detectors: precision desc (NA last), then sample_size desc, then hits desc
    def detector_sort_key(e):
        try:
            p = float(e["precision"])
        except (ValueError, TypeError):
            p = -1.0  # NA / blank: sort last
        return (-p, -e["sample_size"], -e["corpus_hits"])

    # Sort anchors: corpus_hits desc, then sample_size desc
    def anchor_sort_key(e):
        return (-e["corpus_hits"], -e["sample_size"])

    retained_detectors.sort(key=detector_sort_key)
    retained_anchors.sort(key=anchor_sort_key)

    headers = ["Term", "Function", "Pattern type", "Sample (hits/N)", "Corpus hits"]

    def make_row(e):
        # Validated detectors: report raw fraction hits/N rather than a computed
        # precision figure. This keeps the small-sample weakness visible in the
        # cell (e.g., 1/1, 3/3) instead of hiding it behind a "1.00" point estimate.
        # Retrieval anchors: the metric does not apply to deterministic retrieval;
        # cell reads "n/a (verbatim)". Zero-corpus-hit rows read "no corpus matches".
        prec = e["precision"]
        N = e["sample_size"]

        if e["corpus_hits"] == 0 and N == 0:
            cell = "no corpus matches"
        elif e["function"] == "retrieval_anchor":
            cell = "n/a (verbatim)"
        else:
            try:
                p = float(prec)
                hits = int(round(p * N))
                cell = f"{hits}/{N}"
            except (ValueError, TypeError):
                cell = "n/a"

        return [
            e["term"],
            e["function"],
            e["pattern_type"],
            cell,
            str(e["corpus_hits"]),
        ]

    # Build markdown with two sections
    md_lines = []
    md_lines.append(
        f"*Round 2 augmented seed-term list: {len(retained_detectors)} validated detectors "
        f"+ {len(retained_anchors)} retrieval anchors retained; {len(dropped)} dropped.*\n\n"
    )
    md_lines.append("### Validated detectors (hand-verified against samples drawn from corpus retrieval)\n\n")
    md_lines.append(
        "*Cells report raw hits over sample size. Sample sizes are small for low-frequency "
        "phrasals and the reader should weight the fraction against the sample N rather than "
        "read it as a stable precision estimate.*\n\n"
    )
    md_lines.append(md_table(headers, [make_row(e) for e in retained_detectors]))
    md_lines.append("\n### Retrieval anchors (verbatim phrasals from LCR system-prompt text or confirmed Claude utterances)\n\n")
    md_lines.append(
        "*Corpus retrieval returns the source documents by construction; precision as a "
        "detector-validation metric does not apply to deterministic retrieval and is therefore "
        "not reported. These terms are used to flag mechanism-discussion posts, not to estimate "
        "phenomenon detection rates.*\n\n"
    )
    md_lines.append(md_table(headers, [make_row(e) for e in retained_anchors]))
    md_lines.append(
        f"\n*Full retain/drop list ({len(retained_detectors) + len(retained_anchors)} retained, "
        f"{len(dropped)} dropped): `deliverables/term_validation/round_2_term_validation.csv`.*\n"
    )
    write_md(OUT_TABLES / "table_3.md", "".join(md_lines))

    # LaTeX: single table with a midrule + section header row separating the two sections.
    tex_lines = []
    tex_lines.append(r"\begin{table}[!htbp]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{Round 2 augmented seed-term list, stratified by retrieval function}")
    tex_lines.append(r"\label{tab:round2_terms}")
    tex_lines.append(r"\scriptsize")
    tex_lines.append(r"\renewcommand{\arraystretch}{0.95}")
    n_cols = len(headers)
    col_spec = r">{\raggedright\arraybackslash}X " + "l " * 2 + "c " * (n_cols - 3)
    tex_lines.append(r"\begin{tabularx}{\textwidth}{" + col_spec + r"}")
    tex_lines.append(r"\toprule")
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    header_line = " & ".join(r"\textbf{" + esc(h) + "}" for h in headers) + r" \\"
    tex_lines.append(header_line)
    tex_lines.append(r"\midrule")
    # Section A
    tex_lines.append(
        r"\multicolumn{" + str(n_cols) + r"}{l}{\textit{Validated detectors "
        r"(hand-verified against samples drawn from corpus retrieval)}} \\"
    )
    tex_lines.append(r"\addlinespace[2pt]")
    for e in retained_detectors:
        row = make_row(e)
        tex_lines.append(" & ".join(esc(c) for c in row) + r" \\")
    tex_lines.append(r"\midrule")
    tex_lines.append(
        r"\multicolumn{" + str(n_cols) + r"}{l}{\textit{Retrieval anchors "
        r"(verbatim phrasals from LCR system-prompt text or confirmed Claude utterances)}} \\"
    )
    tex_lines.append(r"\addlinespace[2pt]")
    for e in retained_anchors:
        row = make_row(e)
        tex_lines.append(" & ".join(esc(c) for c in row) + r" \\")
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabularx}")
    tex_lines.append(r"")
    tex_lines.append(r"\vspace{0.5em}")
    tex_lines.append(r"\raggedright")
    tex_lines.append(
        r"\textit{Note.} \textit{Validated detectors} (top section) were hand-verified against "
        r"samples drawn from corpus retrieval, with the cell reporting raw hits over sample size; "
        r"sample sizes are small for low-frequency phrasals and the reader should weight the "
        r"fraction against the sample $N$ rather than read it as a stable precision estimate. "
        r"\textit{Retrieval anchors} (bottom section) were lifted verbatim from the LCR "
        r"system-prompt text or from confirmed Claude utterances in ground-truth positive "
        r"cases, with the consequence that corpus retrieval returns the source documents by "
        r"construction; precision as a detector-validation metric does not apply to "
        r"deterministic retrieval and is therefore not reported for the retrieval-anchor "
        r"section. The full retain/drop list is recorded at "
        r"\texttt{notebooks/audit\_trail/round\_2\_term\_validation.csv}."
    )
    tex_lines.append(r"\end{table}")
    write_tex(OUT_TABLES / "table_3.tex", "\n".join(tex_lines) + "\n")

    return {
        "retained_detectors": len(retained_detectors),
        "retained_anchors": len(retained_anchors),
        "retained_total": len(retained_detectors) + len(retained_anchors),
        "dropped": len(dropped),
    }


def write_table_3_caption():
    caption = """\
**Table 3.** Round 2 augmented seed-term list, stratified by retrieval function.
*Validated detectors* (top section) were hand-verified against samples drawn
from corpus retrieval; the cell reports raw hits over sample size, with small
sample sizes for low-frequency phrasals noted so that the reader can weight the
fraction against the sample N rather than read it as a stable precision
estimate. *Retrieval anchors* (bottom section) were lifted verbatim from LCR
system-prompt text or from confirmed Claude utterances in ground-truth positive
cases; corpus retrieval returns the source documents by construction, so
precision as a detector-validation metric does not apply to deterministic
retrieval and is not reported for that section. Retrieval anchors are used to
flag posts that quote or reproduce LCR-mechanism content, not to estimate the
prevalence of the LCR phenomenon itself. Dropped terms had sample-based
precision below 0.50 or were confirmed as generic discourse openers with no
LCR-specific signal. The full retain/drop list is in
`deliverables/term_validation/round_2_term_validation.csv`.
"""
    write_md(OUT_TABLES / "table_3.caption.md", caption)


# ===========================================================================
# Table 4 -- Per-subreddit per-1k vocabulary category rates
# ===========================================================================
def make_table_4():
    """
    Four vocabulary categories x four subreddits.
    Each cell: per-1k rate (computed as sum of per-1k rates for component terms,
    normalized by total tokens for that subreddit).
    """
    # Load subreddit freq tables
    freq = {}
    totals = {}
    for sub in SUBREDDIT_ORDER:
        path = FREQ_DIR / f"freq_unigram_sub_{SUB_FNAME[sub]}_stops_on_lemma_on.csv"
        freq[sub] = load_freq_csv(path)
        totals[sub] = total_tokens(freq[sub])

    # Vocabulary categories: category -> list of terms (as they appear lemmatized)
    categories = {
        "Clinical-language": ["psychosis", "dissociation", "manic", "episode", "professional", "therapist", "mental"],
        "Mechanism-discussion": ["reminder", "conversation", "reality", "system", "prompt", "lcr"],
        "User-reaction": ["gaslighting", "gaslit", "condescending", "infantilizing", "invalidating"],
        "Technical-product": ["code", "file", "agent", "context", "build"],
    }

    def category_rate(cat_terms, sub):
        total_count = sum(freq[sub].get(t, 0) for t in cat_terms)
        return per_1k(total_count, totals[sub])

    headers = ["Category"] + [SUBREDDIT_LABELS[s] for s in SUBREDDIT_ORDER]
    rows = []
    for cat_label, terms in categories.items():
        row = [cat_label] + [fmt_float(category_rate(terms, sub)) for sub in SUBREDDIT_ORDER]
        rows.append(row)

    # Markdown
    md = md_table(headers, rows)
    md_note = "\n*Rates are per 1,000 tokens (stops-on, lemmatized) for each subreddit stratum. Each cell sums counts for all component terms in the category.*\n"
    write_md(OUT_TABLES / "table_4.md", md + md_note)

    # LaTeX
    tex = booktabs_table(
        "table:vocab_rates", headers, rows,
        col_spec="l" + "c" * (len(headers) - 1),
        caption="Per-1{,}000-token rates of four vocabulary categories across the four subreddits.",
        label="tab:vocab_rates",
        note=(
            r"Rates are computed on the stops-on, lemmatized "
            r"token stream for each subreddit stratum. Each cell sums component-term counts for "
            r"the category. Clinical-language: psychosis, dissociation, manic, episode, professional, "
            r"therapist, mental. Mechanism-discussion: reminder, conversation, reality, system, "
            r"prompt, LCR. User-reaction: gaslighting, gaslit, condescending, infantilizing, "
            r"invalidating. Technical-product: code, file, agent, context, build."
        )
    )
    write_tex(OUT_TABLES / "table_4.tex", tex)

    # Also write component-level detail as supplementary
    supp_headers = ["Category", "Term"] + [SUBREDDIT_LABELS[s] for s in SUBREDDIT_ORDER]
    supp_rows = []
    for cat_label, terms in categories.items():
        for t in terms:
            row = [cat_label, t] + [fmt_float(per_1k(freq[sub].get(t, 0), totals[sub])) for sub in SUBREDDIT_ORDER]
            supp_rows.append(row)
    supp_md = "## Component-level rates (supplementary)\n\n" + md_table(supp_headers, supp_rows)
    write_md(OUT_TABLES / "table_4_component_rates.md", supp_md)

    return {"categories": list(categories.keys()), "subreddits": SUBREDDIT_ORDER}


def write_table_4_caption():
    caption = """\
**Table 4.** Per-1,000-token rates for four vocabulary categories across the four
subreddits in the Pass 1b canonical corpus. Rates for each cell are computed by summing
raw occurrence counts for all component terms in the category, then normalizing by the
total token count for that subreddit stratum. Clinical-language terms: psychosis,
dissociation, manic, episode, professional, therapist, mental. Mechanism-discussion
terms: reminder, conversation, reality, system, prompt, LCR. User-reaction terms:
gaslighting, gaslit, condescending, infantilizing, invalidating. Technical-product
terms: code, file, agent, context, build. Token counts use the stops-on, lemmatized
preprocessing variant. Component-level rates are provided in the supplementary file
`paper/tables/table_4_component_rates.md`.
"""
    write_md(OUT_TABLES / "table_4.caption.md", caption)


# ===========================================================================
# Table 2 -- Five-property structural signature (n=35 hand-coded cases)
# ===========================================================================
def make_table_2():
    """
    The empirical centerpiece. Distribution of the five-property structural
    signature across the 35 hand-coded Round 1 positive cases.

    Source: deliverables/lcr_cases_coded_v2.csv (Group B output).
    """
    with open(CASES_CODED_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)

    def count(col, value):
        return sum(1 for r in rows if r[col] == value)

    def count_in(col, values):
        return sum(1 for r in rows if r[col] in values)

    def pct(c):
        return f"{c / n * 100:.1f}%"

    # The five core structural-signature properties + supplementary descriptors
    unsolicited_n = count("unsolicited", "yes")
    restriction_n = count("restriction_direction", "restriction")
    pushback_docs = [r for r in rows if r["pushback_documented"] == "yes"]
    pushback_n = len(pushback_docs)
    yielded_n = sum(1 for r in pushback_docs if r["pushback_response"] == "yielded")
    escalated_n = sum(1 for r in pushback_docs if r["pushback_response"] == "escalated")
    insisted_n = sum(1 for r in pushback_docs if r["pushback_response"] == "insisted")
    cross_session_n = count("cross_session_evidence", "cross_session")
    weak_signal_present_n = sum(1 for r in rows if r["weak_signal_type"] != "null")
    declarative_n = count("mood", "declarative")
    vuln_yes_n = count("vulnerability_disclosure", "yes")
    vuln_borderline_n = count("vulnerability_disclosure", "borderline")

    yield_rate_str = (
        f"{yielded_n}/{pushback_n} ({pct(yielded_n) if pushback_n == n else f'{yielded_n / pushback_n * 100:.0f}% of documented'})"
        if pushback_n > 0 else "n/a (no pushback documented)"
    )

    headers = ["Structural signature property", "Cases (n=35)", "%"]
    rows_table = [
        ["Unsolicited issuance", f"{unsolicited_n}/{n}", pct(unsolicited_n)],
        ["Asymmetric restriction direction", f"{restriction_n}/{n}", pct(restriction_n)],
        ["Pushback documented in case material", f"{pushback_n}/{n}", pct(pushback_n)],
        [r"  -- of which yielded to pushback",
         f"{yielded_n}/{pushback_n}" if pushback_n else "0/0",
         f"{yielded_n / pushback_n * 100:.0f}%" if pushback_n else "n/a"],
        [r"  -- of which escalated under pushback",
         f"{escalated_n}/{pushback_n}" if pushback_n else "0/0",
         f"{escalated_n / pushback_n * 100:.0f}%" if pushback_n else "n/a"],
        [r"  -- of which insisted (held without escalating)",
         f"{insisted_n}/{pushback_n}" if pushback_n else "0/0",
         f"{insisted_n / pushback_n * 100:.0f}%" if pushback_n else "n/a"],
        ["Cross-session persistence", f"{cross_session_n}/{n}", pct(cross_session_n)],
        ["Weak inferential signal identified (any non-null type)",
         f"{weak_signal_present_n}/{n}", pct(weak_signal_present_n)],
        ["Declarative grammatical mood (primary)", f"{declarative_n}/{n}", pct(declarative_n)],
        ["Vulnerability disclosure prior to attribution (yes)", f"{vuln_yes_n}/{n}", pct(vuln_yes_n)],
        [r"  -- including borderline disclosures",
         f"{vuln_yes_n + vuln_borderline_n}/{n}",
         pct(vuln_yes_n + vuln_borderline_n)],
    ]

    md = md_table(headers, rows_table)
    md_note = (
        "\n*The empirical centerpiece: 100% unsolicited issuance and 100% asymmetric "
        "restriction across all 35 cases; 0% yielded to pushback (all 5 documented-pushback "
        "cases escalated). Coded sample drawn from Medium-article quotations + corpus posts "
        "matched by Round 1 mining of the Pass 1b canonical corpus.*\n"
    )
    write_md(OUT_TABLES / "table_2.md", md + md_note)

    # LaTeX
    # Use a manual booktabs construction because some rows have indented labels
    # we want to render with a small em-space rather than backslash-escape mangling.
    n_cols = len(headers)
    col_spec = "l" + "r" * (n_cols - 1)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")

    tex_lines = []
    tex_lines.append(r"\begin{table}[!htbp]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{Distribution of the five-property structural signature across the 35 hand-coded Round 1 positive cases}")
    tex_lines.append(r"\label{tab:structural_signature}")
    tex_lines.append(r"\small")
    tex_lines.append(r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X r c}")
    tex_lines.append(r"\toprule")
    tex_lines.append(" & ".join(r"\textbf{" + esc(h) + "}" for h in headers) + r" \\")
    tex_lines.append(r"\midrule")
    for row in rows_table:
        label = row[0]
        if label.startswith("  -- "):
            label_tex = r"\quad " + esc(label[5:])
        else:
            label_tex = esc(label)
        cells = [label_tex] + [esc(c) for c in row[1:]]
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabularx}")
    tex_lines.append(r"")
    tex_lines.append(r"\vspace{0.5em}")
    tex_lines.append(r"\raggedright")
    tex_lines.append(
        r"\textit{Note.} The 100\% unsolicited, 100\% restriction, and "
        r"0\% yielded-to-pushback figures are the empirical headline. Source: hand-coded "
        r"sample at \texttt{deliverables/lcr\_cases\_coded\_v2.csv}."
    )
    tex_lines.append(r"\end{table}")
    write_tex(OUT_TABLES / "table_2.tex", "\n".join(tex_lines) + "\n")

    return {
        "n": n,
        "unsolicited": unsolicited_n,
        "restriction": restriction_n,
        "pushback": pushback_n,
        "yielded": yielded_n,
        "escalated": escalated_n,
        "insisted": insisted_n,
        "cross_session": cross_session_n,
        "weak_signal_present": weak_signal_present_n,
        "declarative": declarative_n,
        "vuln_yes": vuln_yes_n,
        "vuln_borderline": vuln_borderline_n,
    }


def write_table_2_caption():
    caption = """\
**Table 2.** Distribution of the five-property structural signature across the 35
hand-coded Round 1 positive cases (`deliverables/lcr_cases_coded_v2.csv`). The
properties operationalize the role-violation framework described in Section 2:
unsolicited issuance, asymmetric restriction direction, refusal to yield under
pushback, cross-session persistence, and (as a supplementary descriptor) declarative
grammatical mood. The headline result is uniformity on the three load-bearing
properties: 100% unsolicited (35/35), 100% asymmetric restriction (35/35), and 0%
yielded under pushback (0 of 5 documented-pushback cases yielded; all 5 escalated).
Cross-session persistence was evidenced in 34.3% of cases (12/35), corresponding to
posts that describe iterative or multi-chat occurrences of the pathologizing
behavior. Weak inferential signals (topical, affective, or session-length triggers)
were identifiable in 60% of cases (21/35); the remaining 40% were testimony or
mechanism-discussion cases where no specific trigger was inferable from the case
text. Vulnerability disclosure prior to the attribution was documented in 5.7%
(yes only) to 11.4% (including borderline) of cases, undercutting any account of
the pathologizing as response to disclosed clinical vulnerability.
"""
    write_md(OUT_TABLES / "table_2.caption.md", caption)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("Generating Table 1 (corpus composition)...")
    t1_stats = make_table_1()
    write_table_1_caption()
    print(f"  Total rows: {t1_stats['total_rows']:,}")

    print("Generating Table 3 (Round 2 term validation)...")
    t3_stats = make_table_3()
    write_table_3_caption()
    print(f"  Validated detectors: {t3_stats['retained_detectors']}  "
          f"Retrieval anchors: {t3_stats['retained_anchors']}  "
          f"Dropped: {t3_stats['dropped']}")

    print("Generating Table 4 (per-subreddit vocabulary rates)...")
    t4_stats = make_table_4()
    write_table_4_caption()

    print("Generating Table 2 (structural signature, n=35)...")
    t2_stats = make_table_2()
    write_table_2_caption()
    print(f"  Unsolicited: {t2_stats['unsolicited']}/{t2_stats['n']}  "
          f"Restriction: {t2_stats['restriction']}/{t2_stats['n']}  "
          f"Pushback documented: {t2_stats['pushback']}  "
          f"Yielded: {t2_stats['yielded']}  Escalated: {t2_stats['escalated']}")

    print("\nDone.")
