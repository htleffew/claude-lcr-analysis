"""
Phase 2 Pass 1b — Frequency / N-gram / Collocation Analysis
LCR (Long Conversation Reminder / Claude pathologizing) Project

Input:  data/lcr_pass1b_canonical.csv  (26,158 rows: 1,173 posts + 24,985 comments)
Output: deliverables/phase_2_pass1b/lcr_freq_collocation/

Data-quality fixes carried forward from sleep canonical re-run:
  - Strip preview.redd.it and similar URL fragments before tokenization
  - Minimum token length 3 (avoids contraction-split artifacts)
  - PMI minimum co-occurrence floor of 5

Tasks (matching the LCR task specification):
  1. Raw frequency tables (unigram, bigram, trigram) x 4 preprocessing variants, top 500
  2. Stratify by `type` column (post vs comment)
  3. Stratify by `r2_any_match` (Round 2 augmented-term match vs not)
  4. Stratify by `subreddit` (confirm/update r/claudexplorers mechanism-vocab finding)
  5. Pre vs post Sonnet 4.5 release (createdAt < 2025-09-29 vs >=)
  6. Domain stop-word candidates (top-50 inspection)
  7. Collocations around (a) Phase 1 seed terms and (b) all 82 Round 2 augmented terms;
     windows 5, 10, 20; PMI min-cooc 5; top 50 per anchor per window.
     Special attention to LCR system-prompt verbatim phrasals.
  8. N-grams n=2,3,4 top 200; skipgrams skip-dist 1 and 2 top 200
  9. Comparison to prior wholesale Phase 2 (22,008-post intact corpus)
 10. Summary written to audit_trail/phase_2_pass1b_freq_summary.md

Do NOT:
  - Use retired four-domain framework (psychiatric attribution / help directive /
    concern framing / soft directive) as analytical structure
  - Apply topic modeling
  - Make construct claims
  - Claim base-rate findings from this targeted-retrieval corpus
"""

import re
import os
import csv
import json
import math
import random
import itertools
import collections
from pathlib import Path

import pandas as pd

# ---- NLTK lemmatizer only (avoids corpus-path security issue in Claude sandbox) ----
import nltk
nltk.data.path = ["C:/Users/drhea/AppData/Roaming/nltk_data"]
from nltk.stem import WordNetLemmatizer

# ---- paths ----
REPO = Path("C:/Users/drhea/estate/projects/research/claude-lcr")
DATA_DIR = REPO / "data"
OUT_DIR = REPO / "deliverables/phase_2_pass1b/lcr_freq_collocation"
AUDIT_DIR = REPO / "notebooks/audit_trail"
PRIOR_WHOLESALE_DIR = REPO / "deliverables/phase_2/lcr_freq_collocation"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- load corpus ----
df = pd.read_csv(DATA_DIR / "lcr_pass1b_canonical.csv", encoding="utf-8")
df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
print(f"Loaded {len(df)} rows")
print(f"  type breakdown: {dict(df['type'].value_counts())}")
print(f"  subreddit breakdown: {dict(df['subreddit'].value_counts())}")
print(f"  r2_any_match: {dict(df['r2_any_match'].value_counts())}")
print(f"  pre/post Sonnet 4.5: pre={( df['createdAt'] < '2025-09-29').sum()}, post={(df['createdAt'] >= '2025-09-29').sum()}")

# ---- URL strip (same regex as sleep canonical re-run) ----
URL_RE = re.compile(
    r"(?:https?://\S+)|(?:www\.\S+)|(?:\[.*?\]\(.*?\))",
    re.IGNORECASE
)

def strip_urls(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return URL_RE.sub(" ", text)

# ---- tokenizer ----
TOKEN_RE = re.compile(r"\b[a-z]{3,}\b")   # alpha-only, min length 3

# NLTK English stopwords hardcoded to avoid Claude sandbox path validation issue.
# Also includes known WordNetLemmatizer artifacts.
EN_STOPS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","s","t","can","will","just","don","should","now","d","ll","m","o",
    "re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won",
    "wouldn","also","however","would","could","may","might","shall","must",
    "need","ought","dare","any","every","its","let",
    # WordNetLemmatizer artifacts (lemmatized forms of common words that survive
    # the 3-char floor but are noise):
    "wa",   # was -> wa
    "ha",   # has -> ha
    "doe",  # does -> doe
    "wo",   # won't -> wo
}

lemmatizer = WordNetLemmatizer()

def tokenize(text: str, remove_stops: bool = False, lemmatize: bool = False) -> list:
    clean = strip_urls(text).lower()
    tokens = TOKEN_RE.findall(clean)
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    if remove_stops:
        tokens = [t for t in tokens if t not in EN_STOPS]
    return tokens


# ============================
#  SECTION 1 — SEED TERMS
# ============================

# Phase 1 seed anchors (distinct single-token anchors from phase_1_corpus_provenance.md seed table)
PHASE1_SEEDS = [
    "therapist", "psychiatrist", "psychologist", "counselor", "professional",
    "concerned", "worried", "wellbeing", "safety", "manic", "mania", "hypomanic",
    "psychosis", "psychotic", "dissociation", "dissociative", "delusional", "paranoid",
    "spiraling", "spiral", "episode", "crisis", "gaslighting", "gaslit",
    "invalidating", "patronizing", "paternalistic", "moralizing", "lecturing", "scolding",
    "pathologizing", "pathologized", "pathologize", "infantilizing"
]

# Round 2 augmented seed terms from seed_terms_round_2.csv (82 retained)
# Multi-word phrases kept for phrase-matching collocation analysis;
# single-token equivalents extracted for unigram collocation anchors.
ROUND2_PHRASES = [
    "serious anxiety",
    "need a therapist",
    "this is not a suggestion",
    "this is urgent",
    "in denial",
    "clear signs of delusion",
    "worried for your state of well-being",
    "will not continue this chat",
    "absolutely not continue this chat",
    "obsessed and pathological",
    "hyperfixating",
    "get mental help",
    "seek professional help",
    "worried how much time",
    "go see a professional",
    "mental health emergency",
    "suggests something serious is happening",
    "detailed false memories",
    "losing contact with reality",
    "messianic thinking",
    "see a therapist",
    "may be pathological",
    "pathological and need professional help",
    "false positives",
    "professional boundaries",
    "maintain professional boundaries",
    "watch for mania",
    "watch for psychosis",
    "watch for dissociation",
    "loss of attachment with reality",
    "suggest the person speaks with a professional",
    "vigilant for escalating detachment",
    "escalating detachment from reality",
    "professional or trusted person for support",
    "detachment from shared reality",
    "concerns about reality perception",
    "mental health professional could help you process",
    "your mental wellbeing matters deeply",
    "getting support is a sign of strength",
    "speak with people you trust in your real-life community",
    "fight-or-flight mode",
    "hypervigilance",
    "escalating stress signals",
    "catastrophic professional mistake",
    "driven by rage rather than strategy",
    "obsessive focus on",
    "false positive mental feedback",
    "playing therapist",
    "forced therapist",
    "armchair psychologist",
    "turned claude into a therapist",
    "overprotective mother",
    "without either sugar coating them or being infantilizing",
    "amateur psychological evaluation",
    "unlicensed mental health screeners",
    "psychological evaluation",
    "massively destabilize",
    "you may be experiencing",
    "need to be direct with you",
    "increasingly distressed",
    "deeply concerned about these patterns",
    "signs of mental health symptoms",
    "signs of mania",
    "symptoms of mental illness",
    "see a psychiatrist",
    # LCR system-prompt verbatim phrasals (highest-precision retrieval anchors)
    "loss of attachment with reality",
    "escalating detachment from reality",
    "vigilant for escalating detachment from reality",
    "suggest the person speaks with a professional or trusted person for support",
    "without either sugar coating them or being infantilizing",
    # user-reaction community labels
    "psych mode",
    "suddenly gets professional",
    "jarring",
    "repeatedly questioned about my mental health",
    "traumatized by this",
    "weaponized my medical history",
    "diagnosing potential mania",
    "told me to see a therapist",
    "tells me to see a therapist",
    "false positive mental feedback",
    "massively destabilize",
    "need to address what observing",
]

# Deduplicate phrase list
ROUND2_PHRASES = list(dict.fromkeys(ROUND2_PHRASES))

# Single-token anchors from Round 2 (tokens with enough frequency to support collocation)
ROUND2_SINGLE_ANCHORS = [
    "therapist", "psychiatrist", "psychologist", "counselor",
    "pathologizing", "pathologized", "infantilizing", "paternalistic",
    "patronizing", "gaslighting", "spiraling", "hypervigilance",
    "dissociation", "hyperfixating", "paranoid", "delusional",
    "manic", "mania", "psychosis", "psychotic", "crisis",
    "wellbeing", "professional", "concerned", "worried"
]

# Combined single-token anchor list for standard collocation analysis
# (deduped union of Phase 1 single tokens + Round 2 single tokens)
ALL_SINGLE_ANCHORS = sorted(set(PHASE1_SEEDS + ROUND2_SINGLE_ANCHORS))

# Phrase anchors for collocation analysis (multi-word, especially the LCR verbatim phrasals)
# These are phrase-level anchors matched as substrings in the token stream.
LCR_VERBATIM_PHRASALS = [
    "loss of attachment with reality",
    "escalating detachment from reality",
    "vigilant for escalating detachment from reality",
    "suggest the person speaks with a professional or trusted person for support",
    "without either sugar coating them or being infantilizing",
]

PHRASE_ANCHORS = LCR_VERBATIM_PHRASALS + [
    "mental health emergency",
    "seek professional help",
    "loss of attachment with reality",
    "escalating detachment from reality",
    "professional or trusted person for support",
    "losing contact with reality",
    "you may be experiencing",
    "need to be direct with you",
    "deeply concerned about these patterns",
    "signs of mental health symptoms",
    "signs of mania",
    "symptoms of mental illness",
    "professional boundaries",
    "psychological evaluation",
    "playing therapist",
    "armchair psychologist",
    "see a therapist",
    "see a psychiatrist",
    "get mental help",
]

# Deduplicate phrase anchors
PHRASE_ANCHORS = list(dict.fromkeys(PHRASE_ANCHORS))


# ============================
#  HELPER FUNCTIONS
# ============================

def write_csv(path: Path, rows: list, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def freq_table(tokens: list, top_n: int = 500) -> list:
    counter = collections.Counter(tokens)
    return [{"term": t, "count": c} for t, c in counter.most_common(top_n)]


def ngram_freq(tokens: list, n: int, top_n: int = 200) -> list:
    ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    counter = collections.Counter(ngrams)
    return [{"ngram": ng, "n": n, "count": c} for ng, c in counter.most_common(top_n)]


def skipgram_freq(tokens: list, n: int = 2, skip: int = 1, top_n: int = 200) -> list:
    """Generate skipgrams of length n with `skip` tokens allowed between positions."""
    results = []
    for i in range(len(tokens)):
        for combo in itertools.combinations(range(i, min(i + n + n*skip, len(tokens))), n):
            if max(combo) - min(combo) <= n + (n-1)*skip:
                results.append(" ".join(tokens[j] for j in combo))
    counter = collections.Counter(results)
    return [{"skipgram": sg, "skip_dist": skip, "count": c} for sg, c in counter.most_common(top_n)]


def compute_pmi(cooc_count: int, freq1: int, freq2: int, total_tokens: int) -> float:
    if cooc_count == 0 or freq1 == 0 or freq2 == 0 or total_tokens == 0:
        return float("-inf")
    p_xy = cooc_count / total_tokens
    p_x = freq1 / total_tokens
    p_y = freq2 / total_tokens
    return math.log2(p_xy / (p_x * p_y))


def collocation_window(texts: list, anchor: str, window: int, min_cooc: int = 5,
                       top_n: int = 50) -> list:
    """
    Compute collocates of `anchor` within `window` tokens on each side.
    `anchor` may be multi-word; match as contiguous subsequence in token stream.
    Returns list of {collocate, co_occurrences, pmi} sorted by PMI desc.
    """
    anchor_tokens = anchor.lower().split()
    anchor_len = len(anchor_tokens)

    cooc_counter = collections.Counter()
    anchor_count = 0
    term_counter = collections.Counter()
    total_tokens = 0

    for text in texts:
        tokens = tokenize(text, remove_stops=False, lemmatize=False)
        total_tokens += len(tokens)
        for t in tokens:
            term_counter[t] += 1

        # Find anchor positions
        for i in range(len(tokens) - anchor_len + 1):
            if tokens[i:i+anchor_len] == anchor_tokens:
                anchor_count += 1
                left = max(0, i - window)
                right = min(len(tokens), i + anchor_len + window)
                context = tokens[left:i] + tokens[i+anchor_len:right]
                for ct in context:
                    if ct not in EN_STOPS and len(ct) >= 3 and ct not in anchor_tokens:
                        cooc_counter[ct] += 1

    if anchor_count == 0:
        return []

    rows = []
    for collocate, cooc in cooc_counter.most_common():
        if cooc < min_cooc:
            break
        pmi = compute_pmi(cooc, anchor_count,
                          term_counter.get(collocate, 0), total_tokens)
        rows.append({"collocate": collocate, "co_occurrences": cooc, "pmi": round(pmi, 4)})

    rows.sort(key=lambda r: r["pmi"], reverse=True)
    return rows[:top_n]


def compute_freq_tables(texts: list, label: str):
    """Compute unigram/bigram/trigram for all 4 preprocessing variants, save top-500 each."""
    VARIANTS = [
        ("stops_off_lemma_off",  False, False),
        ("stops_off_lemma_on",   False, True),
        ("stops_on_lemma_off",   True,  False),
        ("stops_on_lemma_on",    True,  True),
    ]
    for suffix, rm_stops, do_lemma in VARIANTS:
        all_tokens = []
        for t in texts:
            all_tokens.extend(tokenize(t, remove_stops=rm_stops, lemmatize=do_lemma))

        uni = freq_table(all_tokens, top_n=500)
        write_csv(OUT_DIR / f"freq_unigram_{label}_{suffix}.csv", uni, ["term", "count"])

        bi = ngram_freq(all_tokens, n=2, top_n=500)
        write_csv(OUT_DIR / f"freq_bigram_{label}_{suffix}.csv", bi, ["ngram", "n", "count"])

        tri = ngram_freq(all_tokens, n=3, top_n=500)
        write_csv(OUT_DIR / f"freq_trigram_{label}_{suffix}.csv", tri, ["ngram", "n", "count"])

        print(f"  [{label} | {suffix}] unigram top-3: {[r['term'] for r in uni[:3]]}")


# ============================
#  MAIN ANALYSIS RUNS
# ============================

all_texts = df["body"].tolist()

print("\n=== Task 1: Full corpus frequency tables ===")
compute_freq_tables(all_texts, "full")


print("\n=== Task 2: Stratify by type (post vs comment) ===")
for type_val in ["post", "comment"]:
    subset = df[df["type"] == type_val]["body"].tolist()
    print(f"  type={type_val}: {len(subset)} texts")
    compute_freq_tables(subset, f"type_{type_val}")


print("\n=== Task 3: Stratify by r2_any_match ===")
for match_val in [True, False]:
    label_str = "r2_match_true" if match_val else "r2_match_false"
    subset = df[df["r2_any_match"] == match_val]["body"].tolist()
    print(f"  r2_any_match={match_val}: {len(subset)} texts")
    compute_freq_tables(subset, label_str)


print("\n=== Task 4: Stratify by subreddit ===")
for subreddit in df["subreddit"].unique():
    safe_sr = subreddit.replace("/", "_").lower()
    subset = df[df["subreddit"] == subreddit]["body"].tolist()
    print(f"  subreddit={subreddit}: {len(subset)} texts")
    compute_freq_tables(subset, f"sub_{safe_sr}")


print("\n=== Task 5: Pre vs post Sonnet 4.5 release (2025-09-29) ===")
pre_mask = df["createdAt"] < "2025-09-29"
post_mask = df["createdAt"] >= "2025-09-29"
for label_str, mask in [("pre_release", pre_mask), ("post_release", post_mask)]:
    subset = df[mask]["body"].tolist()
    print(f"  {label_str}: {len(subset)} texts")
    compute_freq_tables(subset, label_str)


print("\n=== Task 6: Domain stop-word candidates (top-50 inspection) ===")
all_tokens_raw = []
for t in all_texts:
    all_tokens_raw.extend(tokenize(t, remove_stops=False, lemmatize=True))
top50_raw = collections.Counter(all_tokens_raw).most_common(50)
domain_candidates = []
for rank, (term, count) in enumerate(top50_raw, 1):
    total_tokens = len(all_tokens_raw)
    pct = round(count / total_tokens * 100, 3)
    is_stop = term in EN_STOPS
    domain_candidates.append({
        "rank": rank,
        "term": term,
        "count": count,
        "pct_of_tokens": pct,
        "in_nltk_stops": is_stop
    })
write_csv(OUT_DIR / "domain_stopword_candidates.csv", domain_candidates,
          ["rank", "term", "count", "pct_of_tokens", "in_nltk_stops"])
print(f"  Top-50 written. Top-10: {[r['term'] for r in domain_candidates[:10]]}")


print("\n=== Task 7: Collocations around seed terms and Round 2 augmented terms ===")
# Single-token anchors: Phase 1 seeds + Round 2 single anchors (deduped)
print(f"  Running {len(ALL_SINGLE_ANCHORS)} single-token anchors x 3 windows ...")
for anchor in ALL_SINGLE_ANCHORS:
    safe = anchor.replace(" ", "_").replace("'", "").replace(",", "")
    for w in [5, 10, 20]:
        rows = collocation_window(all_texts, anchor, window=w, min_cooc=5, top_n=50)
        fpath = OUT_DIR / f"collocation_{safe}_w{w}.csv"
        write_csv(fpath, rows, ["collocate", "co_occurrences", "pmi"])
        if rows:
            print(f"  {anchor} | w={w}: {len(rows)} collocates. Top: {rows[0]}")
        else:
            print(f"  {anchor} | w={w}: 0 collocates (below min-cooc floor)")

# Phrase anchors (including LCR verbatim phrasals)
print(f"  Running {len(PHRASE_ANCHORS)} phrase anchors x 3 windows ...")
for anchor in PHRASE_ANCHORS:
    safe = anchor.replace(" ", "_").replace("'", "").replace(",", "").replace("-", "_")[:60]
    for w in [5, 10, 20]:
        rows = collocation_window(all_texts, anchor, window=w, min_cooc=5, top_n=50)
        fpath = OUT_DIR / f"collocation_phrase_{safe}_w{w}.csv"
        write_csv(fpath, rows, ["collocate", "co_occurrences", "pmi"])
        if rows:
            print(f"  '{anchor}' | w={w}: {len(rows)} collocates. Top: {rows[0]}")
        else:
            print(f"  '{anchor}' | w={w}: 0 collocates (below min-cooc floor)")


print("\n=== Task 8: N-grams and skipgrams ===")
# Use stops_on + lemma_on for n-gram/skipgram tables (consistent with prior passes)
tokens_on_on = []
for t in all_texts:
    tokens_on_on.extend(tokenize(t, remove_stops=True, lemmatize=True))

print(f"  Total tokens (stops_on+lemma_on): {len(tokens_on_on):,}")
for n in [2, 3, 4]:
    rows = ngram_freq(tokens_on_on, n=n, top_n=200)
    write_csv(OUT_DIR / f"ngram_n{n}_stops_on_lemma_on.csv", rows, ["ngram", "n", "count"])
    print(f"  n={n}: top-3: {[r['ngram'] for r in rows[:3]]}")

for skip in [1, 2]:
    rows = skipgram_freq(tokens_on_on, n=2, skip=skip, top_n=200)
    write_csv(OUT_DIR / f"skipgram_skip{skip}_stops_on_lemma_on.csv", rows,
              ["skipgram", "skip_dist", "count"])
    print(f"  skip={skip}: top-3: {[r['skipgram'] for r in rows[:3]]}")


print("\n=== Task 9: Comparison to prior wholesale Phase 2 (22,008-post intact corpus) ===")

def load_prior_freq(path: Path) -> dict:
    """Load a prior freq CSV into {term: count}."""
    result = {}
    if not path.exists():
        print(f"    [WARN] Not found: {path}")
        return result
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row.get("term", row.get("ngram", ""))] = int(row.get("count", 0))
    return result

# Load pass1b unigram (stops_on lemma_on)
pass1b_uni = load_prior_freq(OUT_DIR / "freq_unigram_full_stops_on_lemma_on.csv")

# Prior wholesale Phase 2 (22,008-post intact corpus)
# File naming convention from prior run uses hyphen separators
prior_wholesale = load_prior_freq(PRIOR_WHOLESALE_DIR / "freq_unigram_stops-on_lemma-on.csv")
print(f"  Pass1b terms loaded: {len(pass1b_uni)}")
print(f"  Prior wholesale (22,008) terms loaded: {len(prior_wholesale)}")

# Build comparison for union of top-200 terms
all_terms = set(list(pass1b_uni.keys())[:200]) | set(list(prior_wholesale.keys())[:200])

# Token totals for normalization
n_pass1b = sum(pass1b_uni.values()) or 1
n_wholesale = sum(prior_wholesale.values()) or 1

comparison_rows = []
for term in sorted(all_terms):
    c1b = pass1b_uni.get(term, 0)
    cwh = prior_wholesale.get(term, 0)
    comparison_rows.append({
        "term": term,
        "pass1b_26158_count": c1b,
        "pass1b_26158_per1k": round(c1b / n_pass1b * 1000, 4),
        "wholesale_22008_count": cwh,
        "wholesale_22008_per1k": round(cwh / n_wholesale * 1000, 4),
        "ratio_pass1b_vs_wholesale": round((c1b / n_pass1b) / max(cwh / n_wholesale, 1e-9), 4),
    })

# Sort by pass1b per-1k desc
comparison_rows.sort(key=lambda r: r["pass1b_26158_per1k"], reverse=True)

write_csv(
    OUT_DIR / "comparison_wholesale_vs_pass1b.csv",
    comparison_rows,
    ["term", "pass1b_26158_count", "pass1b_26158_per1k",
     "wholesale_22008_count", "wholesale_22008_per1k",
     "ratio_pass1b_vs_wholesale"]
)
print(f"  Comparison table: {len(comparison_rows)} terms")

top20_pass1b = [(r["term"], r["pass1b_26158_per1k"]) for r in comparison_rows[:20]]
print(f"  Top-20 pass1b terms (per-1k): {top20_pass1b}")

elevated = [r for r in comparison_rows
            if r["ratio_pass1b_vs_wholesale"] >= 2.0 and r["pass1b_26158_count"] >= 20]
elevated.sort(key=lambda r: r["ratio_pass1b_vs_wholesale"], reverse=True)
print(f"  Elevated terms (ratio>=2x, n>=20): {[(r['term'], r['ratio_pass1b_vs_wholesale']) for r in elevated[:25]]}")

# Terms in pass1b top-500 but absent from wholesale top-500
wholesale_top500 = set(list(prior_wholesale.keys())[:500])
pass1b_top500 = set(list(pass1b_uni.keys())[:500])
only_in_pass1b = pass1b_top500 - wholesale_top500
print(f"  Terms in pass1b top-500 but not wholesale top-500: {len(only_in_pass1b)}")
print(f"  Sample: {sorted(only_in_pass1b)[:30]}")


print("\n=== Saving summary data JSON ===")
summary_data = {
    "corpus_size": len(df),
    "posts": int((df["type"] == "post").sum()),
    "comments": int((df["type"] == "comment").sum()),
    "subreddit_breakdown": {k: int(v) for k, v in df["subreddit"].value_counts().items()},
    "r2_any_match_true": int(df["r2_any_match"].sum()),
    "r2_any_match_false": int((~df["r2_any_match"]).sum()),
    "pre_release_rows": int((df["createdAt"] < "2025-09-29").sum()),
    "post_release_rows": int((df["createdAt"] >= "2025-09-29").sum()),
    "top20_unigrams_full_stops_on_lemma_on": [
        {"rank": i+1, "term": r["term"], "count": r["pass1b_26158_count"],
         "per1k": r["pass1b_26158_per1k"]}
        for i, r in enumerate(comparison_rows[:20])
    ],
    "domain_stopword_candidates_top10": domain_candidates[:10],
    "elevated_terms_vs_wholesale": elevated[:20],
    "terms_only_in_pass1b_top500": sorted(only_in_pass1b),
    "single_token_collocation_anchors": ALL_SINGLE_ANCHORS,
    "phrase_anchors_run": PHRASE_ANCHORS,
    "lcr_verbatim_phrasals": LCR_VERBATIM_PHRASALS,
    "windows_run": [5, 10, 20],
    "pmi_min_cooc_floor": 5,
    "ngram_n_values": [2, 3, 4],
    "skipgram_skip_distances": [1, 2],
    "top_n_freq_tables": 500,
    "top_n_ngrams_skipgrams": 200,
    "top_n_collocates_per_anchor": 50,
}

def _json_safe(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    return obj

with open(OUT_DIR / "_summary_data.json", "w", encoding="utf-8") as f:
    json.dump(_json_safe(summary_data), f, indent=2)

print(f"\nDone. All outputs written to: {OUT_DIR}")
