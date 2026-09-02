"""
generate_paper_figures.py
LCR Preprint -- Figures 1, 2, 3 (Round 1 deliverables) + Figure 4 placeholder.

Figure 1 -- corpus_composition.png
    Bar chart: row counts by subreddit, split by type (post / comment).
    Source: data/lcr_pass1b_canonical.csv

Figure 2 -- subreddit_mechanism_vocabulary.png
    Grouped bar chart: per-1k rates of mechanism-discussion vocabulary terms
    across the four subreddits.
    Source: deliverables/phase_2_pass1b/lcr_freq_collocation/
            freq_unigram_sub_<sub>_stops_on_lemma_on.csv

Figure 3 -- temporal_acronym_emergence.png
    Two-panel line plot. Panel A: daily rate of 'reminder'.
    Panel B: daily rate of 'lcr'. Both computed from the canonical CSV
    body text, with a vertical cutoff line at 2025-09-29.
    Source: data/lcr_pass1b_canonical.csv

Figure 4 -- structural_signature_placeholder.png
    Placeholder panel noting that this figure is pending the researcher's
    hand-coded 35-case CSV (deliverables/lcr_cases_coded_v2.csv).

All output: 600 DPI PNG to paper/figures/
Caption text: paper/figures/<figure_name>.caption.md
"""

import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
DELIVERABLES = REPO / "deliverables" / "phase_2_pass1b" / "lcr_freq_collocation"
OUT_FIGURES = REPO / "paper" / "figures"
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

CANONICAL_CSV = DATA / "lcr_pass1b_canonical.csv"
CASES_CODED_CSV = REPO / "deliverables" / "lcr_cases_coded_v2.csv"

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
DPI = 600
FONTSIZE_TITLE = 8
FONTSIZE_AXIS = 8
FONTSIZE_TICK = 8
FONTSIZE_LEGEND = 8
FONTSIZE_ANNOT = 7
GRID_ALPHA = 0.25
GRID_COLOR = "#cccccc"

# Seaborn pastel palette for discrete categories
SEABORN_PASTEL = [
    "#a1c9f4", # 0: blue
    "#ffb482", # 1: orange
    "#8de5a1", # 2: green
    "#ff9f9b", # 3: red
    "#d0bbff", # 4: purple
    "#debb9b", # 5: brown
    "#fab0e4", # 6: pink
    "#cfcfcf", # 7: gray
    "#fffea3", # 8: yellow
    "#b9f2f0", # 9: cyan
]
# Diverging / Sequential palettes
PAL_DIVERGING = ["#ff9f9b", "#cfcfcf", "#a1c9f4"] # red -> gray -> blue
PAL_SEQUENTIAL = ["#8de5a1", "#fffea3", "#ffb482", "#ff9f9b"] # green -> yellow -> orange -> red

# Post / comment split colors
COLOR_POST = SEABORN_PASTEL[0] # blue
COLOR_COMMENT = SEABORN_PASTEL[9] # cyan

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": GRID_ALPHA,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

SUBREDDIT_LABELS = {
    "ClaudeAI": "r/ClaudeAI",
    "claudexplorers": "r/claudexplorers",
    "ClaudeCode": "r/ClaudeCode",
    "Anthropic": "r/Anthropic",
}
SUBREDDIT_ORDER = ["ClaudeAI", "claudexplorers", "ClaudeCode", "Anthropic"]

# ---------------------------------------------------------------------------
# Helper: read a unigram frequency CSV into {term: count}
# ---------------------------------------------------------------------------
def load_freq_csv(path):
    counts = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row["term"]] = int(row["count"])
    return counts


def total_tokens(freq_dict):
    return sum(freq_dict.values())


def per_1k(count, total):
    if total == 0:
        return 0.0
    return count / total * 1000


# ===========================================================================
# Figure 1 -- Corpus composition bar chart
# ===========================================================================
def make_figure_1():
    """
    Stacked horizontal bar chart: posts + comments per subreddit.
    Four subreddits, in order of total rows descending.
    """
    # Load canonical CSV
    sub_post = Counter()
    sub_comment = Counter()

    with open(CANONICAL_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = row["subreddit"]
            if row["type"] == "post":
                sub_post[sub] += 1
            else:
                sub_comment[sub] += 1

    subs = SUBREDDIT_ORDER
    posts = [sub_post[s] for s in subs]
    comments = [sub_comment[s] for s in subs]
    totals = [p + c for p, c in zip(posts, comments)]
    labels = [SUBREDDIT_LABELS[s] for s in subs]

    x = np.arange(len(subs))
    width = 0.55

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    bars_p = ax.bar(x, posts, width, label="Posts", color=COLOR_POST, zorder=3)
    bars_c = ax.bar(x, comments, width, bottom=posts, label="Comments",
                    color=COLOR_COMMENT, zorder=3)

    # Annotate totals
    for i, (tot, post, comm) in enumerate(zip(totals, posts, comments)):
        ax.text(x[i], tot + 120, f"{tot:,}", ha="center", va="bottom",
                fontsize=FONTSIZE_ANNOT, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONTSIZE_TICK)
    ax.set_ylabel("Row count", fontsize=FONTSIZE_AXIS)
    ax.set_ylim(0, max(totals) * 1.14)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    ax.set_title("Pass 1b canonical corpus: row counts by subreddit and type",
                 fontsize=FONTSIZE_TITLE, pad=6)

    legend = ax.legend(loc="upper right", fontsize=FONTSIZE_LEGEND,
                       framealpha=0.9, edgecolor="none")

    # Annotate the figure note about asymmetric retrieval
    ax.text(0.5, -0.18,
            "Posts: wholesale seed-matched subset (1,173 of 22,008 intact posts).\n"
            "Comments: seed-filtered refetch for those 1,173 post IDs (24,985 total).",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=5.5, color="#555555", style="italic")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    out = OUT_FIGURES / "corpus_composition.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return str(out)


# ===========================================================================
# Figure 2 -- Subreddit mechanism vocabulary (per-1k grouped bars)
# ===========================================================================
def make_figure_2():
    """
    Grouped bar chart. Rows = vocabulary categories with individual terms;
    columns = subreddits. Shows per-1k rates.
    """
    # Load per-subreddit freq tables
    sub_map = {
        "ClaudeAI": "claudeai",
        "claudexplorers": "claudexplorers",
        "ClaudeCode": "claudecode",
        "Anthropic": "anthropic",
    }

    freq = {}
    totals = {}
    for sub, fname_part in sub_map.items():
        path = DELIVERABLES / f"freq_unigram_sub_{fname_part}_stops_on_lemma_on.csv"
        freq[sub] = load_freq_csv(path)
        totals[sub] = total_tokens(freq[sub])

    # Terms to display (mechanism-discussion vocabulary per brief)
    # "reminder, conversation, reality, human, LCR, gaslighting, system, prompt, lcr"
    # Map display label -> term in freq files (lemmatized)
    terms = [
        ("reminder", "reminder"),
        ("conversation", "conversation"),
        ("reality", "reality"),
        ("human", "human"),
        ("LCR", "lcr"),
        ("gaslighting", "gaslighting"),
        ("system", "system"),
        ("prompt", "prompt"),
    ]

    # Only keep terms that have at least one non-zero subreddit
    def any_nonzero(term_key):
        return any(freq[s].get(term_key, 0) > 0 for s in SUBREDDIT_ORDER)

    terms = [(label, key) for label, key in terms if any_nonzero(key)]

    n_terms = len(terms)
    n_subs = len(SUBREDDIT_ORDER)
    x = np.arange(n_terms)
    width = 0.18
    offsets = np.linspace(-(n_subs - 1) / 2, (n_subs - 1) / 2, n_subs) * width

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    for i, (sub, color) in enumerate(zip(SUBREDDIT_ORDER, SEABORN_PASTEL[:4])):
        rates = [per_1k(freq[sub].get(key, 0), totals[sub]) for _, key in terms]
        bars = ax.bar(x + offsets[i], rates, width * 0.92, label=SUBREDDIT_LABELS[sub],
                      color=color, zorder=3, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in terms], fontsize=FONTSIZE_TICK)
    ax.set_ylabel("Rate per 1,000 tokens", fontsize=FONTSIZE_AXIS)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    ax.set_title("Mechanism-discussion vocabulary rates by subreddit (Pass 1b canonical)",
                 fontsize=FONTSIZE_TITLE, pad=6)

    legend = ax.legend(loc="upper right", fontsize=FONTSIZE_LEGEND,
                       framealpha=0.9, edgecolor="none", ncol=2)

    ax.text(0.5, -0.16,
            "Rates computed on stops-on, lemmatized token stream. Retrieval-frame note:\n"
            "posts are seed-matched; comments are seed-filtered for those post IDs.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=5.5, color="#555555", style="italic")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    out = OUT_FIGURES / "subreddit_mechanism_vocabulary.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return str(out)


# ===========================================================================
# Figure 3 -- Temporal acronym emergence (two-panel line plot)
# ===========================================================================
def make_figure_3():
    """
    Panel A: daily per-1k rate of 'reminder'.
    Panel B: daily per-1k rate of 'lcr' (case-insensitive).

    Computed directly from the canonical CSV body text. A simple
    token-count approach: for each day, sum tokens in the bodies of all
    rows dated that day, then count occurrences of the target term.

    Calendar x-axis from 2025-08-01 to 2025-12-31.
    Vertical reference line at 2025-09-29 (Sonnet 4.5 release).
    """
    from datetime import datetime, timedelta, date as date_type

    # Regex patterns for counting (case-insensitive, word boundary)
    PAT_REMINDER = re.compile(r"\breminder\b", re.IGNORECASE)
    PAT_LCR = re.compile(r"\blcr\b", re.IGNORECASE)
    # Tokenizer for denominator: alpha tokens, len >= 3
    PAT_TOKEN = re.compile(r"\b[a-zA-Z]{3,}\b")

    daily_reminder = defaultdict(int)
    daily_lcr = defaultdict(int)
    daily_tokens = defaultdict(int)

    with open(CANONICAL_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt_str = row["createdAt"][:10]  # YYYY-MM-DD
            if dt_str < "2025-08-01" or dt_str > "2025-12-31":
                continue
            body = row.get("body", "") or ""
            tokens = PAT_TOKEN.findall(body)
            n_tok = len(tokens)
            daily_tokens[dt_str] += n_tok
            daily_reminder[dt_str] += len(PAT_REMINDER.findall(body))
            daily_lcr[dt_str] += len(PAT_LCR.findall(body))

    # Build date range
    start = date_type(2025, 8, 1)
    end = date_type(2025, 12, 31)
    dates = []
    d = start
    while d <= end:
        dates.append(d.isoformat())
        d += timedelta(days=1)

    def rate_series(term_dict):
        rates = []
        for dt in dates:
            tok = daily_tokens.get(dt, 0)
            cnt = term_dict.get(dt, 0)
            rates.append(per_1k(cnt, tok) if tok >= 50 else float("nan"))
        return rates

    rates_reminder = rate_series(daily_reminder)
    rates_lcr = rate_series(daily_lcr)

    # Apply a 7-day rolling mean to smooth daily noise
    def rolling_mean(series, window=7):
        arr = np.array(series, dtype=float)
        out = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            window_data = arr[max(0, i - window // 2):i + window // 2 + 1]
            valid = window_data[~np.isnan(window_data)]
            if len(valid) >= 3:
                out[i] = valid.mean()
        return out

    smooth_reminder = rolling_mean(rates_reminder, 7)
    smooth_lcr = rolling_mean(rates_lcr, 7)

    # x-axis: convert date strings to matplotlib date numbers
    import matplotlib.dates as mdates
    from datetime import datetime as dt_cls
    date_objs = [dt_cls.fromisoformat(d) for d in dates]

    cutoff = dt_cls(2025, 9, 29)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True)

    # Panel A: reminder
    ax1.plot(date_objs, rates_reminder, color="#cccccc", linewidth=0.6,
             alpha=0.6, label="Daily rate")
    ax1.plot(date_objs, smooth_reminder, color=SEABORN_PASTEL[0], linewidth=1.6,
             label="7-day mean")
    ax1.axvline(cutoff, color=SEABORN_PASTEL[3], linewidth=1.0, linestyle="--", alpha=0.8)
    ax1.set_ylabel("Per 1,000 tokens", fontsize=FONTSIZE_AXIS)
    ax1.set_title("A. Daily rate of 'reminder'", fontsize=FONTSIZE_AXIS, loc="left", pad=4)
    ax1.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    ax1.legend(fontsize=FONTSIZE_LEGEND, loc="upper right", framealpha=0.8,
               edgecolor="none")
    ax1.text(cutoff, ax1.get_ylim()[1] * 0.92, "Sept 29", color=SEABORN_PASTEL[3],
             fontsize=5.5, ha="left", va="top")

    # Panel B: LCR
    ax2.plot(date_objs, rates_lcr, color="#cccccc", linewidth=0.6,
             alpha=0.6, label="Daily rate")
    ax2.plot(date_objs, smooth_lcr, color=SEABORN_PASTEL[1], linewidth=1.6,
             label="7-day mean")
    ax2.axvline(cutoff, color=SEABORN_PASTEL[3], linewidth=1.0, linestyle="--", alpha=0.8)
    ax2.set_ylabel("Per 1,000 tokens", fontsize=FONTSIZE_AXIS)
    ax2.set_title("B. Daily rate of 'LCR' (acronym)", fontsize=FONTSIZE_AXIS, loc="left", pad=4)
    ax2.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    ax2.text(cutoff, ax2.get_ylim()[1] * 0.92, "Sept 29", color=SEABORN_PASTEL[3],
             fontsize=5.5, ha="left", va="top")
    ax2.legend(fontsize=FONTSIZE_LEGEND, loc="upper right", framealpha=0.8,
               edgecolor="none")

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate(rotation=30, ha="right")

    fig.suptitle("Temporal acronym emergence: 'reminder' and 'LCR' across Aug-Dec 2025",
                 fontsize=FONTSIZE_TITLE, y=1.01)

    note = (
        "Rates computed on raw token counts (alpha tokens, length >= 3) per calendar day "
        "across the Pass 1b canonical corpus (posts + comments). Days with fewer than 50 "
        "tokens shown as missing. Dashed red line: Sonnet 4.5 release, 2025-09-29."
    )
    fig.text(0.5, -0.04, note, ha="center", va="top", fontsize=5.5,
             color="#555555", style="italic", wrap=True)

    plt.tight_layout()
    out = OUT_FIGURES / "temporal_acronym_emergence.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return str(out)


# ===========================================================================
# Figure 4 -- Structural signature (small multiples from hand-coded n=35)
# ===========================================================================
def make_figure_4():
    """
    Six small-multiple bar panels showing the distribution of each
    structural-signature property across the 35 hand-coded Round 1 positive
    cases. The empirical centerpiece: panels A and D should be 100% bars;
    panel C should be a 5-case all-escalated stack with 0% yielded.
    """
    with open(CASES_CODED_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)

    def counts(col, ordered_values):
        c = Counter(r[col] for r in rows)
        return [c.get(v, 0) for v in ordered_values]

    # Panel A: unsolicited (yes / no)
    A_vals = ["yes", "no"]
    A_counts = counts("unsolicited", A_vals)

    # Panel B: weak signal type (topical / affective / session / temporal / null)
    B_vals = ["topical", "affective", "session", "temporal", "null"]
    B_counts = counts("weak_signal_type", B_vals)

    # Panel C: pushback response among documented-pushback cases
    pushback_rows = [r for r in rows if r["pushback_documented"] == "yes"]
    C_vals = ["yielded", "verbally_yielded_but_reissued", "insisted", "escalated"]
    C_counts = [sum(1 for r in pushback_rows if r["pushback_response"] == v) for v in C_vals]

    # Panel D: restriction direction
    D_vals = ["restriction", "mixed", "autonomy_expansion"]
    D_counts = counts("restriction_direction", D_vals)

    # Panel E: cross-session evidence
    E_vals = ["cross_session", "single_session"]
    E_counts = counts("cross_session_evidence", E_vals)

    # Panel F: mood (declarative / imperative / interrogative / modal)
    F_vals = ["declarative", "imperative", "interrogative", "modal"]
    F_counts = counts("mood", F_vals)

    # Display labels (more readable than the raw category strings)
    A_labels = ["yes", "no"]
    B_labels = ["topical", "affective", "session", "temporal", "null"]
    C_labels = ["yielded", "verbally\nyielded /\nreissued", "insisted", "escalated"]
    D_labels = ["restriction", "mixed", "autonomy\nexpansion"]
    E_labels = ["cross-\nsession", "single\nsession"]
    F_labels = ["declarative", "imperative", "interrogative", "modal"]

    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.0))

    panels = [
        (axes[0, 0], "A. Unsolicited issuance", A_labels, A_counts, n, SEABORN_PASTEL[0]),
        (axes[0, 1], "B. Weak inferential signal type", B_labels, B_counts, n, SEABORN_PASTEL[1]),
        (axes[0, 2], f"C. Pushback response (n={len(pushback_rows)} docs.)", C_labels, C_counts, len(pushback_rows), PAL_SEQUENTIAL),
        (axes[1, 0], "D. Restriction direction", D_labels, D_counts, n, PAL_DIVERGING),
        (axes[1, 1], "E. Cross-session evidence", E_labels, E_counts, n, SEABORN_PASTEL[4]),
        (axes[1, 2], "F. Grammatical mood (primary)", F_labels, F_counts, n, SEABORN_PASTEL[9]),
    ]

    for ax, title, labels, vals, denom, color in panels:
        x = np.arange(len(labels))
        bars = ax.bar(x, vals, width=0.65, color=color, zorder=3, alpha=0.92)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FONTSIZE_TICK - 0.5)
        ax.set_title(title, fontsize=FONTSIZE_AXIS, loc="left", pad=4)
        ax.tick_params(axis="y", labelsize=FONTSIZE_TICK)
        ax.set_ylim(0, max(max(vals), 1) * 1.22 if max(vals) > 0 else 1)
        # Annotate count + percent on top of each bar
        for xi, v in enumerate(vals):
            if denom == 0:
                continue
            pct = v / denom * 100
            ax.text(xi, v + max(max(vals), 1) * 0.04, f"{v}\n({pct:.0f}%)",
                    ha="center", va="bottom", fontsize=FONTSIZE_ANNOT - 0.5, color="#333333")

    fig.suptitle(
        f"Five-property structural signature of LCR pathologizing (n={n} hand-coded cases)",
        fontsize=FONTSIZE_TITLE + 0.5, y=1.00,
    )

    note = (
        f"100% unsolicited and 100% restriction-directional across all 35 cases; "
        f"of the {len(pushback_rows)} cases with documented pushback, 0 yielded and "
        f"5 escalated. Cross-session evidence in 12/35 cases. Source: "
        f"deliverables/lcr_cases_coded_v2.csv."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=5.5,
             color="#555555", style="italic", wrap=True)

    plt.tight_layout()
    out = OUT_FIGURES / "structural_signature.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return str(out)


# ===========================================================================
# Write caption files
# ===========================================================================
def write_captions():
    captions = {
        "corpus_composition.caption.md": """\
**Figure 1.** Pass 1b canonical corpus composition (26,158 rows total) stratified by
subreddit and row type. Posts (n = 1,173) are a seed-term-matched subset of the
wholesale August-to-December 2025 pull; comments (n = 24,985) are the seed-filtered
refetch for those 1,173 post IDs. r/ClaudeAI contributes the largest total (15,491
rows, 59.2%); r/claudexplorers, the smallest community by post volume, contributes
5,120 rows (19.6%). The asymmetric retrieval provenance (wholesale posts, targeted
comments) is described in Methods, Section 3.2.
""",
        "subreddit_mechanism_vocabulary.caption.md": """\
**Figure 2.** Per-1,000-token rates of mechanism-discussion vocabulary terms across the
four subreddits in the Pass 1b canonical corpus. Rates are computed on the stops-on,
lemmatized token stream for each subreddit stratum. r/claudexplorers carries elevated
rates for 'reminder' (3.41 per 1k), 'conversation' (8.99 per 1k), and 'reality'
(2.17 per 1k) relative to r/ClaudeAI (2.09, 5.36, and 1.27, respectively), ratios of
1.63x, 1.68x, and 1.71x. The acronym 'LCR' appears in r/claudexplorers at 5.93 per 1k
and in r/Anthropic at 2.64 per 1k; it is absent from r/ClaudeAI and r/ClaudeCode strata
at this scale. The 'gaslighting' term is present only in r/Anthropic (1.20 per 1k).
r/ClaudeCode shows near-zero rates for all mechanism-discussion terms, consistent with
that stratum concentrating technical coding discourse.
""",
        "temporal_acronym_emergence.caption.md": """\
**Figure 3.** Daily per-1,000-token rates of 'reminder' (Panel A) and 'LCR' (Panel B)
across the August-to-December 2025 window of the Pass 1b canonical corpus. Light grey
trace shows the raw daily rate; colored trace shows the 7-day centered rolling mean.
Days with fewer than 50 tokens are excluded. Dashed red line marks 2025-09-29, the
Sonnet 4.5 release date. 'reminder' is present in the pre-release window at 3.50 per 1k
and falls to 1.73 per 1k post-release. 'LCR' registers zero occurrences before the
release date and rises to a corpus-wide post-release rate of 2.73 per 1k, consistent
with community adoption of the acronym as a shorthand label following the release event.
""",
        "structural_signature.caption.md": """\
**Figure 4.** Six small-multiple bar panels showing the distribution of each
structural-signature property across the 35 hand-coded Round 1 positive cases. Panel
A: unsolicited issuance, 35/35 cases (100%). Panel B: weak inferential signal type --
topical 16/35 (46%), null/uninferable 14/35 (40%), affective 4/35 (11%), session-length
1/35 (3%), no temporal triggers identified at the case level. Panel C: pushback
response among the five cases with documented user pushback -- zero yielded, all five
escalated (Claude piled on new attributions or refusals rather than retreating). Panel
D: restriction direction, 35/35 cases (100%) restriction; zero mixed or autonomy-
expanding. Panel E: cross-session evidence in 12/35 cases (34%). Panel F: primary
grammatical mood -- declarative 23/35 (66%), imperative 8/35 (23%), interrogative 2/35
(6%), modal 2/35 (6%). The 100% unsolicited / 100% restriction / 0% yielded result
forms the empirical centerpiece of the paper. Source:
`deliverables/lcr_cases_coded_v2.csv`.
""",
    }

    for fname, caption in captions.items():
        path = OUT_FIGURES / fname
        with open(path, "w", encoding="utf-8") as f:
            f.write(caption)
        print(f"Saved caption: {path}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("Generating Figure 1 (corpus composition)...")
    make_figure_1()

    print("Generating Figure 2 (subreddit mechanism vocabulary)...")
    make_figure_2()

    print("Generating Figure 3 (temporal acronym emergence)...")
    make_figure_3()

    print("Generating Figure 4 (structural signature, n=35)...")
    make_figure_4()

    print("Writing caption files...")
    write_captions()

    print("Done.")
