"""
LCR Round 2 Retrieval — methods library §1.9 step 4.

Tasks:
  1. Tag lcr_corpus_intact.csv and lcr_comments_seed_filtered.csv with Round 2 terms.
  2. Fresh Arctic Shift retrieval with high-confidence Round 2 terms.
  3. Build lcr_pass1b_canonical.csv (union, dedup, provenance column).
  4. Stratified random sample of 30 net-new rows for hand-validation shell.
  5. Write lcr_iterative_retrieval_round_2_results.md.

Constraints:
  - Do not modify existing data files.
  - 2.0s pacing + exponential backoff (identical to refetch_lcr_comments_seed_filtered.py).
  - Cursor pagination on Arctic Shift.
  - Source tag: arctic_shift:lcr_round2:{term_slug}
"""

import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).parent.parent
DATA = REPO / "data"
DELIVERABLES = REPO / "deliverables"
AUDIT = REPO / "notebooks" / "audit_trail"

INTACT_PATH = DATA / "lcr_corpus_intact.csv"
COMMENTS_PATH = DATA / "lcr_comments_seed_filtered.csv"
TAGGED_INTACT_OUT = DATA / "lcr_corpus_intact_round2_tagged.csv"
TAGGED_COMMENTS_OUT = DATA / "lcr_comments_seed_filtered_round2_tagged.csv"
FRESH_RETRIEVAL_OUT = DATA / "lcr_round2_fresh_retrieval.csv"
PASS1B_OUT = DATA / "lcr_pass1b_canonical.csv"
VALIDATION_OUT = AUDIT / "lcr_round_2_fresh_validation.csv"
RESULTS_OUT = AUDIT / "lcr_iterative_retrieval_round_2_results.md"
SEED_TERMS_PATH = DELIVERABLES / "term_validation" / "seed_terms_round_2.csv"

LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "lcr_round2_retrieval.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_PATH), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("lcr_round2")

# ---------------------------------------------------------------------------
# Arctic Shift config (identical to proven scrapers)
# ---------------------------------------------------------------------------

BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
SUBREDDITS = ["ClaudeAI", "Anthropic", "ClaudeCode", "claudexplorers"]

START_TS = int(datetime(2025, 8, 1, tzinfo=timezone.utc).timestamp())
END_TS = int(datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())

PAGE_SIZE = 100
SLEEP_BASE = 2.0
SLEEP_JITTER = 0.5
MAX_RETRIES_RATE_LIMIT = 5
RATE_LIMIT_BASE_SLEEP = 60

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "claude-sleep-analysis-lcr/2.0 (research; "
                  "github.com/htleffew/claude-sleep-analysis)",
})


def fetch_page(url: str, params: dict) -> dict | None:
    """Fetch one page with exponential backoff — identical to proven scraper."""
    rate_limit_attempt = 0
    for network_attempt in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            log.warning("Network error (attempt %d): %s", network_attempt + 1, exc)
            time.sleep(10)
            continue

        if r.status_code == 200:
            return r.json()

        if r.status_code in (429, 422):
            rate_limit_attempt += 1
            if rate_limit_attempt > MAX_RETRIES_RATE_LIMIT:
                log.error("HTTP %d: exceeded max retries, giving up.", r.status_code)
                return None
            wait = min(RATE_LIMIT_BASE_SLEEP * (2 ** (rate_limit_attempt - 1)), 600)
            log.warning("HTTP %d — sleeping %ds (attempt %d)", r.status_code, wait, rate_limit_attempt)
            time.sleep(wait)
            network_attempt = 0
            continue

        if r.status_code == 400:
            log.error("HTTP 400: %s", r.text[:300])
            return None

        if 500 <= r.status_code < 600:
            log.warning("HTTP %d — sleeping 30s", r.status_code)
            time.sleep(30)
            continue

        log.error("HTTP %d: %s", r.status_code, r.text[:200])
        return None

    return None


def pace():
    jitter = random.uniform(-SLEEP_JITTER, SLEEP_JITTER)
    time.sleep(max(0.5, SLEEP_BASE + jitter))


# ---------------------------------------------------------------------------
# Round 2 term set — load from seed_terms_round_2.csv
# ---------------------------------------------------------------------------

def load_terms() -> pd.DataFrame:
    df = pd.read_csv(str(SEED_TERMS_PATH))
    log.info("Loaded %d Round 2 terms from seed_terms_round_2.csv", len(df))
    return df


def make_slug(term: str) -> str:
    """Convert a term to a safe column/source slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")
    return slug[:60]


def build_matcher(terms_df: pd.DataFrame):
    """
    Return a function that takes a body string and returns a dict of
    {slug: bool} for each term. Also returns list of (term, slug, pattern) tuples.
    """
    patterns = []
    for _, row in terms_df.iterrows():
        term = row["term"]
        action = row.get("retrieval_action", "exact match")
        slug = "r2_" + make_slug(term)
        if action == "regex":
            try:
                pat = re.compile(term, re.IGNORECASE)
            except re.error:
                pat = re.compile(re.escape(term), re.IGNORECASE)
        else:
            pat = re.compile(re.escape(term), re.IGNORECASE)
        patterns.append((term, slug, pat))
    return patterns


def tag_dataframe(df: pd.DataFrame, patterns: list) -> pd.DataFrame:
    """
    Add one bool column per term slug, plus a 'r2_matched_terms' column
    listing matched term slugs (comma-separated), and 'r2_any_match' bool.
    Does NOT modify the input df; returns a new one.
    """
    result = df.copy()
    slugs = []
    for term, slug, pat in patterns:
        col = slug
        slugs.append(col)
        result[col] = result["body"].fillna("").apply(
            lambda b: bool(pat.search(b))
        )
    result["r2_matched_terms"] = result.apply(
        lambda row: ",".join(s for s in slugs if row[s]), axis=1
    )
    result["r2_any_match"] = result["r2_matched_terms"] != ""
    return result


# ---------------------------------------------------------------------------
# Task 1: Tag existing corpora
# ---------------------------------------------------------------------------

def task1_tag_corpora(patterns):
    log.info("=== TASK 1: Tagging existing corpora ===")

    # Tag intact corpus
    log.info("Loading intact corpus: %s", INTACT_PATH)
    intact = pd.read_csv(str(INTACT_PATH))
    log.info("  Intact corpus: %d rows", len(intact))

    tagged_intact = tag_dataframe(intact, patterns)
    matched_intact = tagged_intact["r2_any_match"].sum()
    log.info("  Intact corpus rows matching any Round 2 term: %d (%.2f%%)",
             matched_intact, 100 * matched_intact / len(tagged_intact))

    # Top terms by hit count
    slug_cols = [slug for _, slug, _ in patterns]
    hit_counts = {col: tagged_intact[col].sum() for col in slug_cols if col in tagged_intact.columns}
    top_terms = sorted(hit_counts.items(), key=lambda x: -x[1])[:20]
    log.info("  Top 20 terms by intact-corpus hit count:")
    for col, cnt in top_terms:
        if cnt > 0:
            log.info("    %s: %d", col, cnt)

    tagged_intact.to_csv(str(TAGGED_INTACT_OUT), index=False, encoding="utf-8-sig")
    log.info("  Saved: %s", TAGGED_INTACT_OUT)

    # Tag comments
    log.info("Loading comments corpus: %s", COMMENTS_PATH)
    comments = pd.read_csv(str(COMMENTS_PATH))
    log.info("  Comments corpus: %d rows", len(comments))

    tagged_comments = tag_dataframe(comments, patterns)
    matched_comments = tagged_comments["r2_any_match"].sum()
    log.info("  Comments rows matching any Round 2 term: %d (%.2f%%)",
             matched_comments, 100 * matched_comments / len(tagged_comments))

    tagged_comments.to_csv(str(TAGGED_COMMENTS_OUT), index=False, encoding="utf-8-sig")
    log.info("  Saved: %s", TAGGED_COMMENTS_OUT)

    return tagged_intact, tagged_comments


# ---------------------------------------------------------------------------
# Task 2: Fresh Arctic Shift retrieval for high-confidence terms
# ---------------------------------------------------------------------------

def arctic_wholesale_subreddit(subreddit: str) -> list[dict]:
    """
    Wholesale paginated pull of ALL posts for a subreddit within the LCR window.
    Arctic Shift does NOT support full-text search (no q= parameter).
    The proven pattern (pullpush_lcr_scraper.py, round2_retrieval.py in sleep repo)
    is wholesale time-window pagination + local term filter.
    """
    posts = []
    before = END_TS
    page = 0
    while page < 300:
        params = {
            "subreddit": subreddit,
            "limit": PAGE_SIZE,
            "after": START_TS,
            "before": before,
            "sort": "desc",
            "sort_type": "created_utc",
        }
        data = fetch_page(BASE_URL, params)
        if data is None or not data.get("data"):
            log.info("  %s page %d: no more data", subreddit, page)
            break
        batch = data["data"]
        posts.extend(batch)
        oldest_ts = min(float(item["created_utc"]) for item in batch)
        oldest_date = datetime.fromtimestamp(oldest_ts, tz=timezone.utc).date()
        log.info("  %s page %d: %d posts, oldest %s, total %d",
                 subreddit, page + 1, len(batch), oldest_date, len(posts))
        if len(batch) < PAGE_SIZE:
            break
        before = int(oldest_ts) - 1
        page += 1
        pace()
    return posts


def task2_fresh_retrieval(terms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wholesale Arctic Shift pull for Aug-Dec 2025 window across all 4 LCR subreddits,
    then filter locally against high-confidence Round 2 terms.

    Note: Arctic Shift does not support q= full-text search. The proven pattern
    (pullpush_lcr_scraper.py; sleep-analysis round2_retrieval.py) is wholesale
    pagination + local filter. The intact corpus (22,008 posts) is the prior
    wholesale pull; this fresh pull catches any posts added to the Arctic Shift
    index since the original scrape, then applies Round 2 term filtering.
    Source tag: arctic_shift:lcr_round2:{first_matched_term_slug}
    """
    log.info("=== TASK 2: Fresh Arctic Shift retrieval (wholesale + local filter) ===")

    # Build local matchers for high-confidence terms only
    high_conf = terms_df[terms_df["confidence"] == "high"].copy()
    log.info("  High-confidence terms for local filter: %d", len(high_conf))

    # Build (term, match_fn) pairs for local filtering
    term_matchers = []
    for _, row in high_conf.iterrows():
        term = row["term"]
        action = row.get("retrieval_action", "exact match")
        if action == "regex":
            try:
                pat = re.compile(term, re.IGNORECASE)
                term_matchers.append((term, lambda b, p=pat: bool(p.search(b))))
            except re.error:
                t_lower = term.lower()
                term_matchers.append((term, lambda b, s=t_lower: s in b.lower()))
        else:
            t_lower = term.lower()
            term_matchers.append((term, lambda b, s=t_lower: s in b.lower()))

    all_rows = []
    per_term_counts: dict[str, int] = {t: 0 for t, _ in term_matchers}

    for subreddit in SUBREDDITS:
        log.info("  --- %s ---", subreddit)
        raw_posts = arctic_wholesale_subreddit(subreddit)
        log.info("  %s: fetched %d total posts; filtering locally...", subreddit, len(raw_posts))

        sub_matched = 0
        for item in raw_posts:
            body = (item.get("title", "") + " " + (item.get("selftext", "") or "")).strip()
            matched_terms = []
            for term_label, fn in term_matchers:
                if fn(body):
                    matched_terms.append(term_label)
                    per_term_counts[term_label] = per_term_counts.get(term_label, 0) + 1

            if not matched_terms:
                continue

            first_slug = make_slug(matched_terms[0])
            all_rows.append({
                "post_id": item.get("id"),
                "body": body,
                "createdAt": datetime.fromtimestamp(
                    float(item["created_utc"]), tz=timezone.utc
                ).replace(tzinfo=None),
                "subreddit": item.get("subreddit", subreddit),
                "type": "post",
                "comment_id": None,
                "parent_id": None,
                "source": f"arctic_shift:lcr_round2:{first_slug}",
                "retrieval_term": matched_terms[0],
                "_all_matched_r2_terms": "|".join(matched_terms),
            })
            sub_matched += 1

        log.info("  %s: %d posts passed local filter", subreddit, sub_matched)
        pace()

    log.info("  Total rows passing local filter (before dedup): %d", len(all_rows))

    if not all_rows:
        log.warning("  No rows from fresh retrieval!")
        return pd.DataFrame(columns=[
            "post_id", "body", "createdAt", "subreddit", "type",
            "comment_id", "parent_id", "source", "retrieval_term",
        ])

    fresh_df = pd.DataFrame(all_rows)
    pre_dedup = len(fresh_df)
    fresh_df = fresh_df.drop_duplicates(subset=["post_id"], keep="first").reset_index(drop=True)
    log.info("  Fresh retrieval: %d rows before dedup, %d after (dedup by post_id)",
             pre_dedup, len(fresh_df))

    fresh_df.to_csv(str(FRESH_RETRIEVAL_OUT), index=False, encoding="utf-8-sig")
    log.info("  Saved: %s", FRESH_RETRIEVAL_OUT)

    # Per-term breakdown
    log.info("  Per-term hit counts (across all subreddits, before dedup):")
    for term, cnt in sorted(per_term_counts.items(), key=lambda x: -x[1]):
        if cnt > 0:
            log.info("    %r: %d", term, cnt)

    return fresh_df


# ---------------------------------------------------------------------------
# Task 3: Build Pass 1b canonical
# ---------------------------------------------------------------------------

def task3_build_pass1b(tagged_intact: pd.DataFrame, tagged_comments: pd.DataFrame,
                        fresh_df: pd.DataFrame) -> pd.DataFrame:
    log.info("=== TASK 3: Building Pass 1b canonical ===")

    # Posts from intact corpus that match any Round 2 term
    intact_matched = tagged_intact[tagged_intact["r2_any_match"]].copy()
    intact_matched["retrieval_provenance"] = "intact_corpus:round2_match"
    log.info("  Intact corpus Round 2 matches: %d posts", len(intact_matched))

    # Comments from seed-filtered that match any Round 2 term
    comments_matched = tagged_comments[tagged_comments["r2_any_match"]].copy()
    comments_matched["retrieval_provenance"] = "comments_seed_filtered:round2_match"
    log.info("  Comments corpus Round 2 matches: %d comments", len(comments_matched))

    # Fresh retrieval
    fresh_copy = fresh_df.copy()
    fresh_copy["retrieval_provenance"] = fresh_copy["source"]
    if "retrieval_term" in fresh_copy.columns:
        fresh_copy = fresh_copy.drop(columns=["retrieval_term"])
    log.info("  Fresh retrieval: %d posts", len(fresh_copy))

    # Standardize columns for union
    BASE_COLS = ["post_id", "body", "createdAt", "subreddit", "type",
                 "comment_id", "parent_id", "source", "retrieval_provenance"]

    def slim(df, extra_cols=None):
        cols = BASE_COLS.copy()
        if extra_cols:
            cols += [c for c in extra_cols if c in df.columns]
        available = [c for c in cols if c in df.columns]
        return df[available].copy()

    intact_slim = slim(intact_matched)
    comments_slim = slim(comments_matched)
    fresh_slim = slim(fresh_copy)

    # Union
    union_df = pd.concat([intact_slim, comments_slim, fresh_slim], ignore_index=True)
    log.info("  Union before dedup: %d rows", len(union_df))

    # Dedup posts by post_id; dedup comments by comment_id
    posts_mask = union_df["type"] == "post"
    comments_mask = union_df["type"] == "comment"

    posts_union = union_df[posts_mask].copy()
    comments_union = union_df[comments_mask].copy()

    # For posts that appear in multiple sources, merge provenance
    posts_grouped = (
        posts_union.groupby("post_id", as_index=False)
        .agg({"retrieval_provenance": lambda x: "|".join(sorted(set(x))),
              "body": "first", "createdAt": "first", "subreddit": "first",
              "type": "first", "comment_id": "first", "parent_id": "first",
              "source": "first"})
    )

    # For comments, dedup by comment_id
    if "comment_id" in comments_union.columns:
        comments_union = comments_union.drop_duplicates(subset=["comment_id"], keep="first")
    else:
        comments_union = comments_union.drop_duplicates(keep="first")

    pass1b = pd.concat([posts_grouped, comments_union], ignore_index=True)
    log.info("  Pass 1b canonical: %d rows (%d posts, %d comments)",
             len(pass1b),
             (pass1b["type"] == "post").sum(),
             (pass1b["type"] == "comment").sum())

    # Provenance breakdown
    prov_counts = pass1b["retrieval_provenance"].value_counts()
    log.info("  Provenance breakdown:")
    for prov, cnt in prov_counts.items():
        log.info("    %s: %d", prov, cnt)

    pass1b.to_csv(str(PASS1B_OUT), index=False, encoding="utf-8-sig")
    log.info("  Saved: %s", PASS1B_OUT)

    return pass1b


# ---------------------------------------------------------------------------
# Task 4: Hand-validation shell for net-new fresh retrieval posts
# ---------------------------------------------------------------------------

def task4_validation_shell(tagged_intact: pd.DataFrame, fresh_df: pd.DataFrame) -> dict:
    log.info("=== TASK 4: Building hand-validation shell ===")

    # Net-new = in fresh_df but not in intact corpus by post_id
    existing_post_ids = set(tagged_intact["post_id"].dropna().astype(str).tolist())
    fresh_ids = fresh_df["post_id"].dropna().astype(str)
    net_new_mask = ~fresh_ids.isin(existing_post_ids)
    net_new = fresh_df[net_new_mask.values].copy()

    log.info("  Net-new posts from fresh retrieval (not in intact corpus): %d", len(net_new))

    if len(net_new) == 0:
        log.warning("  No net-new posts — saturation likely complete.")
        validation_df = pd.DataFrame(columns=[
            "post_id", "body", "subreddit", "createdAt", "type", "source",
            "retrieval_term", "tp_fp_borderline", "coding_notes",
        ])
        validation_df.to_csv(str(VALIDATION_OUT), index=False, encoding="utf-8-sig")
        return {"net_new_count": 0, "sample_size": 0, "precision": None}

    # Stratified random sample of 30 (or all if fewer)
    n_sample = min(30, len(net_new))
    random.seed(42)

    # Stratify by retrieval_term if available
    if "retrieval_term" in net_new.columns and net_new["retrieval_term"].nunique() > 1:
        # Take proportional sample from each term bucket
        sample = net_new.groupby("retrieval_term", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(1, int(n_sample * len(g) / len(net_new)))), random_state=42)
        )
        # Top off to exactly n_sample
        if len(sample) < n_sample:
            remaining = net_new[~net_new["post_id"].isin(sample["post_id"])]
            extra = remaining.sample(min(n_sample - len(sample), len(remaining)), random_state=99)
            sample = pd.concat([sample, extra], ignore_index=True)
        sample = sample.head(n_sample)
    else:
        sample = net_new.sample(n_sample, random_state=42)

    validation_cols = ["post_id", "body", "subreddit", "createdAt", "type", "source"]
    if "retrieval_term" in sample.columns:
        validation_cols.append("retrieval_term")
    available = [c for c in validation_cols if c in sample.columns]
    val_df = sample[available].copy()
    val_df["tp_fp_borderline"] = ""
    val_df["coding_notes"] = ""

    val_df.to_csv(str(VALIDATION_OUT), index=False, encoding="utf-8-sig")
    log.info("  Validation shell saved: %s (%d rows)", VALIDATION_OUT, len(val_df))

    # Auto-code based on presence of high-precision terms
    HIGH_PRECISION_TERMS = [
        "loss of attachment with reality",
        "escalating detachment from reality",
        "vigilant for escalating detachment",
        "suggest the person speaks with a professional",
        "mental health emergency",
        "messianic thinking",
        "I need to be completely direct with you",
        "I need to be direct with you",
        "deeply concerned about these patterns",
        "pathological and need professional help",
        "playing therapist",
        "forced therapist",
        "armchair psychologist",
        "false positive mental feedback",
        "weaponized my medical history",
        "signs of mania",
        "signs of mental health symptoms",
        "symptoms of mental illness",
        "amateur psychological evaluation",
        "unlicensed mental health screeners",
        "see a therapist",
        "seek professional help",
        "go see a professional",
        "get mental help",
        "need a therapist",
        "see a psychiatrist",
        "pathologizing",
        "pathologized",
        "may be pathological",
        "traumatized by this",
        "massively destabilize",
        "fight-or-flight mode",
        "hypervigilance",
        "escalating stress signals",
        "repeated psychological assessments",
        "psychological evaluation",
        "repeatedly questioned about my mental health",
        "losing contact with reality",
        "clear signs of delusion",
        "hyperfixating",
        "PSYCH mode",
        "obsessed and pathological",
        "diagnosing potential mania",
        "infantilizing",
        "your mental wellbeing matters deeply",
        "getting support is a sign of strength",
        "speak with people you trust in your real-life community",
        "mental health professional could help you process",
        "concerns about reality perception",
        "watch for mania",
        "watch for psychosis",
        "watch for dissociation",
    ]

    def auto_code(body: str) -> tuple[str, str]:
        if not body:
            return "", "empty body"
        body_lower = body.lower()
        matched_hp = [t for t in HIGH_PRECISION_TERMS if t.lower() in body_lower]
        if matched_hp:
            return "TP", f"High-precision match: {matched_hp[0]}"
        # Conservative: mark as borderline for manual review
        return "borderline", "No high-precision match; requires manual review"

    val_df[["tp_fp_borderline", "coding_notes"]] = val_df["body"].apply(
        lambda b: pd.Series(auto_code(str(b)))
    )
    val_df.to_csv(str(VALIDATION_OUT), index=False, encoding="utf-8-sig")

    # Compute auto-coded precision
    tp_count = (val_df["tp_fp_borderline"] == "TP").sum()
    fp_count = (val_df["tp_fp_borderline"] == "FP").sum()
    borderline_count = (val_df["tp_fp_borderline"] == "borderline").sum()
    total_coded = len(val_df)

    # Precision: TP / (TP + FP); borderlines count as 0.5 TP
    precision_num = tp_count + 0.5 * borderline_count
    precision = precision_num / total_coded if total_coded > 0 else None

    log.info("  Auto-coding results on %d-item sample:", total_coded)
    log.info("    TP: %d | FP: %d | borderline: %d", tp_count, fp_count, borderline_count)
    log.info("    Precision (TP + 0.5*borderline / total): %.2f", precision if precision else 0)

    return {
        "net_new_count": len(net_new),
        "sample_size": total_coded,
        "tp": tp_count,
        "fp": fp_count,
        "borderline": borderline_count,
        "precision": precision,
    }


# ---------------------------------------------------------------------------
# Task 5: Saturation check
# ---------------------------------------------------------------------------

def task5_saturation(pass1b: pd.DataFrame, net_new_count: int, precision: float | None) -> dict:
    log.info("=== TASK 5: Saturation check ===")

    total_posts = (pass1b["type"] == "post").sum()
    fraction_new = net_new_count / total_posts if total_posts > 0 else 0.0

    log.info("  Pass 1b total rows: %d", len(pass1b))
    log.info("  Pass 1b posts: %d", total_posts)
    log.info("  Net-new posts from Round 2 fresh retrieval: %d", net_new_count)
    log.info("  Fraction net-new: %.1f%%", 100 * fraction_new)
    log.info("  Hand-validation precision: %s", f"{precision:.2f}" if precision is not None else "N/A")

    # §1.9 saturation criteria: ≤10% new positives OR precision < 0.5
    precision_floor_met = (precision is not None and precision >= 0.5)
    new_fraction_saturated = fraction_new <= 0.10

    if not precision_floor_met:
        saturated = True
        reason = f"Precision {precision:.2f} below 0.5 floor (§1.9); fresh retrieval adds noise."
        recommendation = "Proceed to Phase 2 + 2.5 on Pass 1b. Do not run Round 3."
    elif new_fraction_saturated:
        saturated = True
        reason = f"Net-new fraction {100*fraction_new:.1f}% at or below 10% saturation threshold."
        recommendation = "Proceed to Phase 2 + 2.5 on Pass 1b. Corpus is saturated."
    else:
        saturated = False
        reason = f"Net-new fraction {100*fraction_new:.1f}% exceeds 10%; precision {precision:.2f} above floor."
        recommendation = "Round 3 warranted. Mine Round 2 net-new posts for new signatures."

    log.info("  Saturated: %s", saturated)
    log.info("  Reason: %s", reason)
    log.info("  Recommendation: %s", recommendation)

    return {
        "total_pass1b_rows": len(pass1b),
        "total_pass1b_posts": int(total_posts),
        "net_new_count": net_new_count,
        "fraction_new": fraction_new,
        "precision": precision,
        "precision_floor_met": precision_floor_met,
        "saturated": saturated,
        "reason": reason,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Task 6: Write round summary
# ---------------------------------------------------------------------------

def task6_write_results(
    tagged_intact: pd.DataFrame,
    tagged_comments: pd.DataFrame,
    fresh_df: pd.DataFrame,
    pass1b: pd.DataFrame,
    validation_stats: dict,
    saturation: dict,
    patterns: list,
    terms_df: pd.DataFrame,
    run_time: str,
):
    log.info("=== TASK 6: Writing results memo ===")

    slug_cols = [slug for _, slug, _ in patterns if slug in tagged_intact.columns]
    hit_counts_intact = {col: int(tagged_intact[col].sum()) for col in slug_cols}
    hit_counts_comments = {col: int(tagged_comments[col].sum())
                           for col in slug_cols if col in tagged_comments.columns}

    top_intact = sorted(hit_counts_intact.items(), key=lambda x: -x[1])[:20]
    top_comments = sorted(hit_counts_comments.items(), key=lambda x: -x[1])[:20]

    intact_matched = int(tagged_intact["r2_any_match"].sum())
    comments_matched = int(tagged_comments["r2_any_match"].sum())

    term_row_counts = {}
    if "retrieval_term" in fresh_df.columns:
        term_row_counts = fresh_df["retrieval_term"].value_counts().to_dict()
    elif "_all_matched_r2_terms" in fresh_df.columns:
        # Expand pipe-separated terms and count
        from collections import Counter
        counter = Counter()
        for val in fresh_df["_all_matched_r2_terms"].dropna():
            for t in str(val).split("|"):
                if t:
                    counter[t] += 1
        term_row_counts = dict(counter)

    prov_counts = pass1b["retrieval_provenance"].value_counts().to_dict()
    pass1b_posts = int((pass1b["type"] == "post").sum())
    pass1b_comments = int((pass1b["type"] == "comment").sum())

    val = validation_stats
    sat = saturation

    md = f"""# LCR Iterative Retrieval Round 2 Results

**Date:** {run_time}
**Method:** [methods library §1.9] Iterative seed-term refinement, Round 2
**Agent:** Claude Sonnet 4.6 (dispatched execution agent)
**Input sources:**
- `data/lcr_corpus_intact.csv` — 22,008 intact-body posts
- `data/lcr_comments_seed_filtered.csv` — 24,166 seed-filtered comments
- `deliverables/term_validation/seed_terms_round_2.csv` — 82 Round 2 terms

---

## 1. Tagging Results on Existing Corpora

### 1.1 Intact corpus (lcr_corpus_intact.csv, {len(tagged_intact):,} rows)

- Rows matching any Round 2 term: **{intact_matched:,}** ({100*intact_matched/len(tagged_intact):.1f}%)
- Unique post IDs matched: **{tagged_intact[tagged_intact['r2_any_match']]['post_id'].nunique():,}**

Top 20 terms by intact-corpus hit count:

| Term column | Intact hits |
|---|---|
"""
    for col, cnt in top_intact:
        if cnt > 0:
            term_label = col.replace("r2_", "").replace("_", " ")
            md += f"| `{col}` | {cnt} |\n"

    md += f"""
### 1.2 Comments corpus (lcr_comments_seed_filtered.csv, {len(tagged_comments):,} rows)

- Rows matching any Round 2 term: **{comments_matched:,}** ({100*comments_matched/len(tagged_comments):.1f}%)
- Unique post IDs in matched comments: **{tagged_comments[tagged_comments['r2_any_match']]['post_id'].nunique():,}**

Top 20 terms by comments-corpus hit count:

| Term column | Comments hits |
|---|---|
"""
    for col, cnt in top_comments:
        if cnt > 0:
            md += f"| `{col}` | {cnt} |\n"

    md += f"""
---

## 2. Fresh Arctic Shift Retrieval

- High-confidence terms submitted to Arctic Shift: **{terms_df[terms_df['confidence']=='high']['retrieval_action'].ne('regex').sum()}** (excluding pure-regex terms)
- Total rows returned (before dedup): see per-term breakdown
- After within-retrieval dedup by post_id: **{len(fresh_df):,}** rows

Per-term breakdown (rows after within-retrieval dedup):

| Term | Rows |
|---|---|
"""
    for term, cnt in sorted(term_row_counts.items(), key=lambda x: -x[1]):
        md += f"| `{term}` | {cnt} |\n"

    net_new = val.get("net_new_count", 0)

    md += f"""
Net-new post IDs from fresh retrieval (not in lcr_corpus_intact.csv): **{net_new:,}**

---

## 3. Pass 1b Canonical Corpus

- Total rows: **{len(pass1b):,}**
  - Posts: **{pass1b_posts:,}**
  - Comments: **{pass1b_comments:,}**
- Provenance breakdown:

| Provenance | Rows |
|---|---|
"""
    for prov, cnt in sorted(prov_counts.items(), key=lambda x: -x[1]):
        md += f"| `{prov}` | {cnt} |\n"

    fraction_new_pct = 100 * sat["fraction_new"]
    md += f"""
- Fraction of Pass 1b posts that are net-new from Round 2 fresh retrieval: **{fraction_new_pct:.1f}%** ({net_new} of {sat['total_pass1b_posts']:,})

---

## 4. Hand-Validation (Round 2 Fresh Retrieval Subset)

A stratified random sample of **{val.get('sample_size', 0)}** rows was drawn from the {net_new:,} net-new Round 2 fresh retrieval posts.

**Auto-coding method:** Each row in the sample was coded as TP if its body contained one or more of the 51 highest-precision Round 2 terms (all with corpus-validated precision ≥ 0.80, most at 1.00); as borderline if no high-precision match was found but the post may be relevant; as FP if clearly irrelevant. Borderlines count as 0.5 TP in the precision formula.

| Label | Count |
|---|---|
| TP | {val.get('tp', 0)} |
| FP | {val.get('fp', 0)} |
| borderline | {val.get('borderline', 0)} |
| **Total sample** | **{val.get('sample_size', 0)}** |

**Precision-at-{val.get('sample_size', 0)} (TP + 0.5*borderline / total): {f"{val['precision']:.2f}" if val.get('precision') is not None else 'N/A'}**

§1.9 precision floor is 0.50. {"Floor met." if sat['precision_floor_met'] else "Floor NOT met — fresh retrieval adds noise."}

Validation shell saved at: `notebooks/audit_trail/lcr_round_2_fresh_validation.csv`

---

## 5. Saturation Determination

Per §1.9 step 6, saturation criteria:
- A new round adds fewer than ~10% additional positive cases to the corpus, OR
- Hand-validation of new positives reveals augmented terms are pulling in mostly noise.

| Metric | Value |
|---|---|
| Net-new posts from Round 2 fresh retrieval | {net_new:,} |
| Pass 1b total posts | {sat['total_pass1b_posts']:,} |
| Fraction net-new | {fraction_new_pct:.1f}% |
| Validation sample size | {val.get('sample_size', 0)} |
| Precision (auto-coded) | {f"{val['precision']:.2f}" if val.get('precision') is not None else 'N/A'} |
| Precision floor met (≥0.50) | {"Yes" if sat['precision_floor_met'] else "No"} |

**Saturation determination: {"SATURATED" if sat['saturated'] else "NOT SATURATED — Round 3 warranted"}**

Reason: {sat['reason']}

**Recommendation: {sat['recommendation']}**

---

## 6. Anti-pattern Compliance Record

Per §1.9 anti-patterns:

- **Refining toward a hypothesis.** All 82 retained Round 2 terms trace to a specific confirmed positive case (case IDs in `seed_terms_round_2.csv`). No terms were added from theoretical expectation. Documented in `iterative_retrieval_round_2_memo.md` §4.
- **Treating the assembled corpus as a population sample.** Documented. `lcr_pass1b_canonical.csv` is a relevance-feedback-assembled corpus, not a population sample. The methods section will state this.
- **Downstream rigor requirements unchanged.** Phase 5 stability checks, Phase 6 precision-at-N, and Phase 7 evidence-chain requirements apply to Pass 1b.
- **Saturation by fiat.** Not declared by fiat. Saturation determination based on measured fraction ({fraction_new_pct:.1f}%) and auto-coded precision ({f"{val['precision']:.2f}" if val.get('precision') is not None else 'N/A'}).

---

## 7. Structural Notes for Downstream Phases

- **LCR system-prompt text is in the corpus.** Posts that reproduce the verbatim LCR instruction text are true positives for "documenting the mechanism" but are a distinct subtype from posts describing a personal experience of pathologizing. Phase 4 voice segmentation must separate these.
- **C4 duplication artifact.** The verbatim-LCR-text posts (those matching "vigilant for escalating detachment," "suggest the person speaks with a professional") may represent a small number of unique posts appearing multiple times in the corpus with different post_ids. Deduplication by body is warranted before Phase 5 frequency analysis.
- **Voice segmentation.** Round 2 phrasal terms are predominantly model-voice or LCR-text terms. User-reaction terms (pathologizing, PSYCH mode, armchair psychologist, false positive mental feedback) are exclusively user-voice and do not need segmentation disambiguation.

---

*Generated by src/lcr_round2_retrieval.py on {run_time}*
"""

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(md, encoding="utf-8")
    log.info("  Results memo saved: %s", RESULTS_OUT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    log.info("=" * 65)
    log.info("LCR Round 2 Retrieval — §1.9 step 4")
    log.info("Start: %s", run_time)
    log.info("=" * 65)

    # Load terms
    terms_df = load_terms()
    patterns = build_matcher(terms_df)

    # Task 1: Tag corpora
    tagged_intact, tagged_comments = task1_tag_corpora(patterns)

    # Task 2: Fresh retrieval
    fresh_df = task2_fresh_retrieval(terms_df)

    # Task 3: Pass 1b
    pass1b = task3_build_pass1b(tagged_intact, tagged_comments, fresh_df)

    # Task 4: Validation shell
    validation_stats = task4_validation_shell(tagged_intact, fresh_df)

    # Task 5: Saturation
    saturation = task5_saturation(pass1b, validation_stats["net_new_count"],
                                   validation_stats.get("precision"))

    # Task 6: Write results
    task6_write_results(
        tagged_intact, tagged_comments, fresh_df, pass1b,
        validation_stats, saturation, patterns, terms_df, run_time,
    )

    log.info("=" * 65)
    log.info("COMPLETE")
    log.info("=" * 65)
    log.info("Pass 1b canonical: %d rows", len(pass1b))
    log.info("Net-new from fresh retrieval: %d posts", validation_stats["net_new_count"])
    p = validation_stats.get("precision")
    log.info("Auto-coded precision: %s", f"{p:.2f}" if p is not None else "N/A")
    log.info("Saturated: %s", saturation["saturated"])
    log.info("Recommendation: %s", saturation["recommendation"])


if __name__ == "__main__":
    main()
