"""
LCR search-filtered comment refetch — Arctic Shift API, cursor-based pagination.

METHODOLOGICAL CONTEXT (2026-05-17 pivot):
  The wholesale comment refetch v2 (src/refetch_lcr_comments_wholesale_v2.py) ran cleanly
  but its purpose was retired. The methodologically appropriate corpus is search-filtered:
  comments for posts that match at least one LCR seed term from the Phase 1 provenance
  record, not all comments for all posts.

  Rationale (community_reported_llm_behavior_method.md §C.1 targeted retrieval mode):
    Comments are general discussion; the phenomenon's community-response signal concentrates
    around phenomenon-matched posts. Wholesale comment retrieval was estimated at 22h+ runtime
    for material with marginal phenomenological value. The pivot was decided after the v2
    wholesale run demonstrated clean operation but excessive runtime.

  Retrieval-frame bias: the corpus contains only comments on posts whose authors used
  seed-term vocabulary. Users who experienced the phenomenon but described it outside the
  seed-term vocabulary are absent from the comment corpus. This bias is named here and
  managed via [method §1.9 iterative seed-term refinement] if downstream Phase 5-6 work
  reveals corpus sparsity.

  Partial wholesale output preserved for provenance at:
    data/lcr_comments_wholesale_partial_aborted.csv

BUGS FIXED FROM v1 (wholesale, refetch_lcr_comments_wholesale.py):
  1. v1 used offset=page*PAGE_SIZE pagination — not accepted by Arctic Shift (HTTP 400).
  2. v1 pacing (1.5 s flat) triggered HTTP 422 "Timeout" errors.

INHERITED FIXES (identical to v2):
  - Cursor-based pagination: before=oldest_utc - 1 per page.
  - Only documented Arctic Shift parameters: link_id, limit, after, before, sort, sort_type.
  - Exponential backoff on 429/422; hard fail on 400 unknown-param.
  - Conservative pacing: 2.0 s base + random jitter in [-0.5, +0.5] s.
  - Long pause every 250 post_ids.

DIFFERENCE FROM v2:
  - Iterates only over post_ids that matched at least one LCR seed term.
  - Loads post_ids from data/lcr_corpus_intact_seed_tagged.csv (matched_terms not null/empty).
  - Output file: data/lcr_comments_seed_filtered.csv (distinct from wholesale output).
  - Source field: 'arctic_shift:comments:seed_filtered_v1'.

Usage:
    python src/refetch_lcr_comments_seed_filtered.py           # full run
    python src/refetch_lcr_comments_seed_filtered.py --test    # 5 diverse post_ids only

    # Override test IDs from command line (optional):
    python src/refetch_lcr_comments_seed_filtered.py --test --test-ids 1o286pp 1pv0gt7 1pwiqm1 1nzip5c 1np2wc4

Schema (data/lcr_comments_seed_filtered.csv):
    post_id, body, createdAt (naive UTC datetime), subreddit, type='comment',
    comment_id, parent_id, source='arctic_shift:comments:seed_filtered_v1'

Provenance:
    Tagged corpus: data/lcr_corpus_intact_seed_tagged.csv (22,008 intact posts, 1,173 matched).
    Wholesale partial preserved: data/lcr_comments_wholesale_partial_aborted.csv (provenance only).
    Methodological pivot documented in: notebooks/audit_trail/lcr_comment_refetch_job.md
"""

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TAGGED_CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "lcr_corpus_intact_seed_tagged.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "lcr_comments_seed_filtered.csv"
)
TEST_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "lcr_comments_seed_filtered_test.csv"
)
LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "logs", "lcr_comment_refetch_seed_filtered.log"
)

COMMENT_BASE_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"

# Pacing: 2.0 s base + random jitter in [-0.5, +0.5] — identical to v2
SLEEP_BETWEEN_REQUESTS = 2.0
SLEEP_JITTER = 0.5

# Long pause every N post_ids to let the server breathe
LONG_PAUSE_EVERY = 250
LONG_PAUSE_SECONDS = 30

PAGE_SIZE = 100          # Arctic Shift max items per request
FLUSH_EVERY = 25         # append to disk every N post_ids

# Retry settings — identical to v2
MAX_RETRIES_RATE_LIMIT = 5    # 429 / 422
MAX_RETRIES_SERVER = 3        # 5xx
MAX_RETRIES_NETWORK = 3       # requests exceptions
RATE_LIMIT_BASE_SLEEP = 60    # seconds (doubles each attempt, capped at 600)
SERVER_ERROR_SLEEP = 30       # seconds on 5xx
NETWORK_ERROR_SLEEP = 10      # seconds on requests exception

# Smoke-test threshold for median comments per post (warn if too low)
MEDIAN_WARN_THRESHOLD = 0.5

# Source tag for provenance
SOURCE_TAG = "arctic_shift:comments:seed_filtered_v1"

# Default test post_ids — 5 spanning different seed-term categories
# (LCR, gaslighting, therapist, psychosis, manic/multi-term)
DEFAULT_TEST_IDS = ["1o286pp", "1pv0gt7", "1pwiqm1", "1nzip5c", "1np2wc4"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("lcr_comments_seed_filtered")

# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "claude-sleep-analysis-lcr/2.0 (research; "
                  "github.com/htleffew/claude-sleep-analysis)",
})


# ---------------------------------------------------------------------------
# Core fetch helper with full retry/backoff (identical to v2)
# ---------------------------------------------------------------------------

def fetch_page(url: str, params: dict) -> dict | None:
    """
    Fetch one API page with tiered retry/backoff.

    Returns the parsed JSON dict on success, or None on permanent failure.
    Raises SystemExit on HTTP 400 with an unknown-parameter message (hard fail
    so a bad parameter set is never silently swallowed).
    """
    rate_limit_attempt = 0

    for network_attempt in range(MAX_RETRIES_NETWORK):
        try:
            r = SESSION.get(url, params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            wait = NETWORK_ERROR_SLEEP
            log.warning("Network error (attempt %d/%d): %s — sleeping %ds",
                        network_attempt + 1, MAX_RETRIES_NETWORK, exc, wait)
            time.sleep(wait)
            continue

        if r.status_code == 200:
            return r.json()

        if r.status_code in (429, 422):
            rate_limit_attempt += 1
            if rate_limit_attempt > MAX_RETRIES_RATE_LIMIT:
                log.error("HTTP %d: exceeded max retries (%d). Giving up on this page.",
                          r.status_code, MAX_RETRIES_RATE_LIMIT)
                return None
            wait = min(RATE_LIMIT_BASE_SLEEP * (2 ** (rate_limit_attempt - 1)), 600)
            log.warning("HTTP %d: %s — sleeping %ds (rate-limit attempt %d/%d)",
                        r.status_code, r.text[:120], wait,
                        rate_limit_attempt, MAX_RETRIES_RATE_LIMIT)
            time.sleep(wait)
            # Do NOT count this toward the network attempt loop; retry immediately
            network_attempt = 0  # reset network counter; rate-limit has its own
            continue

        if r.status_code == 400:
            body_text = r.text[:300]
            if "Unknown query parameter" in body_text or "unknown" in body_text.lower():
                log.critical(
                    "HTTP 400 — unknown/invalid parameter detected in request.\n"
                    "  URL    : %s\n"
                    "  Params : %s\n"
                    "  Response: %s\n"
                    "This is a programming error. Aborting to avoid silent data loss.",
                    url, params, body_text,
                )
                sys.exit(1)
            log.error("HTTP 400 (non-parameter): %s — skipping page.", body_text)
            return None

        if 500 <= r.status_code < 600:
            log.warning("HTTP %d — sleeping %ds before retry.",
                        r.status_code, SERVER_ERROR_SLEEP)
            time.sleep(SERVER_ERROR_SLEEP)
            continue

        # Any other unexpected status
        log.error("HTTP %d: %s — skipping page.", r.status_code, r.text[:200])
        return None

    log.error("All retries exhausted for params=%s", params)
    return None


# ---------------------------------------------------------------------------
# Per-post cursor-based comment pagination (identical to v2)
# ---------------------------------------------------------------------------

def fetch_comments_for_post(post_id: str) -> list[dict]:
    """
    Fetch ALL comments for a single post_id using Arctic Shift cursor pagination.

    Strategy (mirrors scrape_subreddit_posts in pullpush_lcr_scraper.py):
      - First request: sort=desc, sort_type=created_utc, no before/after.
        Gets the most-recent PAGE_SIZE comments.
      - If full page returned: grab oldest created_utc, set before=oldest-1,
        repeat until response has fewer than PAGE_SIZE items.
      - Only documented parameters are used: link_id, limit, sort, sort_type, before.
    """
    comments = []
    before_ts = None
    page_num = 0

    while True:
        params: dict = {
            "link_id": post_id,
            "limit": PAGE_SIZE,
            "sort": "desc",
            "sort_type": "created_utc",
        }
        if before_ts is not None:
            params["before"] = before_ts

        data = fetch_page(COMMENT_BASE_URL, params)

        if data is None:
            log.warning("  post_id=%s page=%d: fetch returned None, stopping pagination.",
                        post_id, page_num)
            break

        batch = data.get("data", []) or []

        if not batch:
            break  # clean empty page — done

        for item in batch:
            created_utc = item.get("created_utc")
            if created_utc is not None:
                created_at = datetime.fromtimestamp(
                    float(created_utc), tz=timezone.utc
                ).replace(tzinfo=None)
            else:
                created_at = None

            comments.append({
                "post_id": post_id,
                "body": item.get("body", ""),
                "createdAt": created_at,
                "subreddit": item.get("subreddit", ""),
                "type": "comment",
                "comment_id": item.get("id"),
                "parent_id": item.get("parent_id"),
                "source": SOURCE_TAG,
            })

        if len(batch) < PAGE_SIZE:
            # Last page (partial) — pagination complete
            break

        # Set cursor to one second before the oldest item in this batch
        oldest_ts = min(
            float(item["created_utc"])
            for item in batch
            if item.get("created_utc") is not None
        )
        before_ts = int(oldest_ts) - 1

        page_num += 1

        # Paced inter-page sleep (not the full between-post sleep)
        jitter = random.uniform(-SLEEP_JITTER, SLEEP_JITTER)
        time.sleep(max(0.5, SLEEP_BETWEEN_REQUESTS + jitter))

    return comments


# ---------------------------------------------------------------------------
# Incremental CSV writer
# ---------------------------------------------------------------------------

SCHEMA_COLUMNS = [
    "post_id", "body", "createdAt", "subreddit",
    "type", "comment_id", "parent_id", "source",
]


def append_to_csv(rows: list[dict], output_path: str, is_first_write: bool):
    """Append a batch of rows to the output CSV."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    if is_first_write:
        df.to_csv(output_path, index=False, encoding="utf-8-sig", mode="w")
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig",
                  mode="a", header=False)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def eta_string(elapsed_s: float, done: int, total: int) -> str:
    if done == 0:
        return "unknown"
    rate = elapsed_s / done
    remaining_s = rate * (total - done)
    hours, rem = divmod(int(remaining_s), 3600)
    minutes = rem // 60
    return f"{hours}h {minutes}m"


def smoke_test_comment_counts(comment_counts: list[int], posts_done: int):
    """Log a warning if median comment count is suspiciously low."""
    if not comment_counts:
        return
    median_val = sorted(comment_counts)[len(comment_counts) // 2]
    min_val = min(comment_counts)
    max_val = max(comment_counts)
    log.info("  [smoke] after %d posts — comments per post: median=%d min=%d max=%d",
             posts_done, median_val, min_val, max_val)
    if median_val < MEDIAN_WARN_THRESHOLD:
        log.warning("  [smoke] WARNING: median comments/post %.2f is anomalously low. "
                    "Possible pagination or API issue.", median_val)


# ---------------------------------------------------------------------------
# Main run logic
# ---------------------------------------------------------------------------

def run(post_ids: list[str], output_path: str):
    total = len(post_ids)
    log.info("=" * 65)
    log.info("LCR search-filtered comment refetch (seed_filtered_v1)")
    log.info("Post IDs to process : %d (seed-term-matched only)", total)
    log.info("Output              : %s", output_path)
    log.info("Source tag          : %s", SOURCE_TAG)
    log.info("Pacing              : %.1fs base + ±%.1f jitter between requests",
             SLEEP_BETWEEN_REQUESTS, SLEEP_JITTER)
    log.info("Long pause          : %ds every %d post_ids",
             LONG_PAUSE_SECONDS, LONG_PAUSE_EVERY)
    log.info("Flush interval      : every %d post_ids", FLUSH_EVERY)
    log.info("Start time          : %s", datetime.now())
    # ETA estimate: 2.0 s/request * total posts + long pauses
    n_long_pauses = total // LONG_PAUSE_EVERY
    estimated_s = (SLEEP_BETWEEN_REQUESTS * total) + (n_long_pauses * LONG_PAUSE_SECONDS)
    log.info("Estimated runtime   : ~%.1fh (2.0s pacing + %d long pauses, single-comment posts)",
             estimated_s / 3600, n_long_pauses)
    log.info("=" * 65)

    start_time = time.time()
    running_comments: list[dict] = []
    total_written = 0
    is_first_write = True
    comment_counts_buffer: list[int] = []

    for idx, pid in enumerate(post_ids):
        pid_str = str(pid)
        comments = fetch_comments_for_post(pid_str)
        running_comments.extend(comments)
        comment_counts_buffer.append(len(comments))

        posts_done = idx + 1

        # Smoke test every 100 posts
        if posts_done % 100 == 0:
            elapsed = time.time() - start_time
            eta = eta_string(elapsed, posts_done, total)
            log.info("[%d/%d] comments so far: %d | elapsed: %ds | ETA: %s",
                     posts_done, total,
                     total_written + len(running_comments),
                     int(elapsed), eta)
            smoke_test_comment_counts(comment_counts_buffer, posts_done)
            comment_counts_buffer = []

        # Flush to disk every FLUSH_EVERY post_ids and at the end
        if posts_done % FLUSH_EVERY == 0 or posts_done == total:
            append_to_csv(running_comments, output_path, is_first_write)
            total_written += len(running_comments)
            running_comments = []
            is_first_write = False

        # Long pause every LONG_PAUSE_EVERY posts
        if posts_done % LONG_PAUSE_EVERY == 0 and posts_done < total:
            log.info("  [long pause] %ds after %d posts...",
                     LONG_PAUSE_SECONDS, posts_done)
            time.sleep(LONG_PAUSE_SECONDS)

        # Paced sleep between post_ids (skip after the last one)
        if posts_done < total:
            jitter = random.uniform(-SLEEP_JITTER, SLEEP_JITTER)
            time.sleep(max(0.5, SLEEP_BETWEEN_REQUESTS + jitter))

    # Final summary
    elapsed_total = time.time() - start_time
    log.info("")
    log.info("=" * 65)
    log.info("REFETCH COMPLETE")
    log.info("=" * 65)
    log.info("Total comments collected : %d", total_written)
    log.info("Total elapsed            : %ds (%.2fh)", int(elapsed_total), elapsed_total / 3600)

    if total_written > 0 and os.path.exists(output_path):
        try:
            out_df = pd.read_csv(output_path, parse_dates=["createdAt"])
            dates = out_df["createdAt"].dropna()
            if not dates.empty:
                log.info("Date range               : %s to %s",
                         dates.min().date(), dates.max().date())
            log.info("Subreddit breakdown:")
            for sub, count in out_df["subreddit"].value_counts().items():
                log.info("  %s: %d", sub, count)
        except Exception as exc:
            log.warning("(Could not load output for summary: %s)", exc)
    else:
        log.warning("No comments were collected.")

    log.info("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "LCR search-filtered comment refetch via Arctic Shift (cursor pagination). "
            "Iterates only over post_ids that matched at least one LCR seed term."
        )
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run against 5 diverse seed-matched post_ids only (schema + pagination validation).",
    )
    parser.add_argument(
        "--test-ids",
        nargs="*",
        default=None,
        help="Override the default test post_ids. Used with --test.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load seed-matched post_ids from tagged corpus
    # -----------------------------------------------------------------------
    log.info("Loading tagged corpus from %s ...", TAGGED_CORPUS_PATH)
    tagged_df = pd.read_csv(TAGGED_CORPUS_PATH)
    total_corpus = len(tagged_df)

    # Seed-matched: matched_terms is not null and not empty string
    has_match = tagged_df["matched_terms"].notna() & (
        tagged_df["matched_terms"].astype(str).str.strip() != ""
    )
    matched_df = tagged_df[has_match]
    all_post_ids = matched_df["post_id"].dropna().unique().tolist()

    log.info("Total intact posts in corpus  : %d", total_corpus)
    log.info("Seed-term-matched post_ids    : %d (%.1f%% of intact corpus)",
             len(all_post_ids), 100 * len(all_post_ids) / total_corpus if total_corpus else 0)
    log.info("Tagged corpus path            : %s", TAGGED_CORPUS_PATH)

    if args.test:
        test_ids = args.test_ids if args.test_ids else DEFAULT_TEST_IDS
        output_path = TEST_OUTPUT_PATH
        log.info("\n*** TEST MODE: processing %d post_ids ***", len(test_ids))
        log.info("Post IDs: %s", test_ids)
        log.info("Output  : %s", output_path)
        run(test_ids, output_path)

        # Print schema verification
        if os.path.exists(output_path):
            test_df = pd.read_csv(output_path)
            log.info("\nTest output: %d rows, %d columns", len(test_df), len(test_df.columns))
            log.info("Columns: %s", list(test_df.columns))
            counts_by_post = test_df.groupby("post_id").size()
            log.info("Comments per post_id:\n%s", counts_by_post.to_string())
            pd.set_option("display.max_colwidth", 80)
            pd.set_option("display.max_columns", 10)
            log.info("\nFirst 3 rows:\n%s", test_df.head(3).to_string(index=False))
            # Source tag check
            sources = test_df["source"].unique()
            log.info("Source tags: %s", sources)
        else:
            log.warning("Test output file not found — zero comments returned from API.")
        return

    # -----------------------------------------------------------------------
    # Full run — write to seed-filtered output (new file, do NOT touch wholesale)
    # -----------------------------------------------------------------------
    output_path = OUTPUT_PATH
    if os.path.exists(output_path):
        log.warning(
            "Output file already exists: %s\n"
            "  This run will OVERWRITE it. Starting fresh.",
            output_path,
        )
        os.remove(output_path)

    run(all_post_ids, output_path)


if __name__ == "__main__":
    main()
