"""
LCR-era corpus scraper using Arctic Shift API.

Targets the late-2025 Sonnet 4.5 era when the Long Conversation Reminder
pathologizing payload was at its peak in community discourse. Pulls posts
from r/ClaudeAI, r/Anthropic, r/ClaudeCode, and r/claudexplorers across
August 1 to December 31, 2025.

Arctic Shift (arctic-shift.photon-reddit.com) is an academic Reddit
preservation project that has continued coverage where Pullpush.io shows
gaps (Pullpush is missing approximately May 2025 through January 2026 at
the time of this scrape). API endpoints are Pushshift-compatible.

Output: data/lcr_corpus.csv with the same column structure as the main
corpus, so downstream analyzers can run unchanged.
"""

import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

DATA_DIR = "data/"
os.makedirs(DATA_DIR, exist_ok=True)

SUBREDDITS = ["ClaudeAI", "Anthropic", "ClaudeCode", "claudexplorers"]

# Window: Aug 1, 2025 -> Dec 31, 2025 (inclusive)
START_TS = int(datetime(2025, 8, 1, tzinfo=timezone.utc).timestamp())
END_TS = int(datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())

# Pullpush returns at most 100 items per request
PAGE_SIZE = 100
SLEEP_BETWEEN_REQUESTS = 1.5  # be polite, avoid hammering
MAX_PAGES_PER_SUBREDDIT = 500  # cap as safety

BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
COMMENT_BASE_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "claude-sleep-analysis-lcr/1.0 (research; "
                  "github.com/htleffew/claude-sleep-analysis)",
})


def fetch_page(url, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  429 rate limit. Sleeping {wait}s...")
                time.sleep(wait)
                continue
            print(f"  HTTP {r.status_code}: {r.text[:200]}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  request error (attempt {attempt+1}): {e}")
            time.sleep(10)
    return None


def scrape_subreddit_posts(subreddit):
    """Walk backward through history fetching posts."""
    print(f"\n--- {subreddit} ---")
    posts = []
    before = END_TS
    page = 0
    while page < MAX_PAGES_PER_SUBREDDIT:
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
            print(f"  page {page}: no more data, stopping")
            break
        batch = data["data"]
        for item in batch:
            posts.append({
                "post_id": item.get("id"),
                "body": (item.get("title", "") + " " + (item.get("selftext", "") or "")),
                "createdAt": datetime.fromtimestamp(
                    item["created_utc"], tz=timezone.utc
                ).replace(tzinfo=None),
                "subreddit": item.get("subreddit", subreddit),
                "type": "post",
                "comment_id": None,
                "parent_id": None,
                "source": f"pullpush:{subreddit}:posts",
            })
        oldest_ts = min(item["created_utc"] for item in batch)
        oldest_date = datetime.fromtimestamp(oldest_ts, tz=timezone.utc).date()
        print(f"  page {page+1}: {len(batch)} posts, oldest {oldest_date}")
        if len(batch) < PAGE_SIZE:
            break
        before = oldest_ts - 1
        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    print(f"  collected {len(posts)} posts from {subreddit}")
    return posts


def scrape_subreddit_comments(subreddit, post_ids):
    """Fetch comments for specific post_ids via Pullpush comment search."""
    print(f"\n--- {subreddit} comments (link_id-scoped) ---")
    comments = []
    # Pullpush supports search by link_id (the parent submission id)
    for i, pid in enumerate(post_ids):
        params = {
            "link_id": pid,
            "limit": 50,
            "after": START_TS,
            "before": END_TS,
        }
        data = fetch_page(COMMENT_BASE_URL, params)
        if data is None:
            continue
        batch = data.get("data", []) or []
        for item in batch:
            comments.append({
                "post_id": pid,
                "body": item.get("body", ""),
                "createdAt": datetime.fromtimestamp(
                    item["created_utc"], tz=timezone.utc
                ).replace(tzinfo=None),
                "subreddit": item.get("subreddit", subreddit),
                "type": "comment",
                "comment_id": item.get("id"),
                "parent_id": item.get("parent_id"),
                "source": f"pullpush:{subreddit}:comments",
            })
        if (i + 1) % 25 == 0:
            print(f"  fetched comments for {i+1}/{len(post_ids)} posts; "
                  f"running total {len(comments)} comments")
        time.sleep(SLEEP_BETWEEN_REQUESTS * 0.5)
    print(f"  collected {len(comments)} comments total")
    return comments


# Prefilter: keep posts that mention Claude or LCR-relevant lexicon
LCR_PREFILTER_TERMS = [
    "claude", "sonnet", "opus", "anthropic", "llm",
    "spiral", "spiraling", "manic", "mania", "psychosis", "psychotic",
    "dissociation", "dissociative", "episode", "hypomanic", "hypomania",
    "seek help", "professional help", "crisis line", "mental health",
    "concerned about you", "your wellbeing", "lecturing", "lectured",
    "moralizing", "pathologizing", "diagnos", "therapist",
    "long conversation reminder", "lcr", "system prompt",
    "paternalistic", "patronizing", "scolding", "berating",
    "rest", "sleep", "tired", "exhausted", "tomorrow", "tonight",
]


def passes_prefilter(body):
    if not body:
        return False
    body_lower = body.lower()
    return any(term in body_lower for term in LCR_PREFILTER_TERMS)


def main():
    start_run = datetime.now()
    print(f"LCR scrape start: {start_run}")
    print(f"Window: {datetime.fromtimestamp(START_TS, tz=timezone.utc).date()} "
          f"to {datetime.fromtimestamp(END_TS, tz=timezone.utc).date()}")

    all_posts = []
    for sub in SUBREDDITS:
        sub_posts = scrape_subreddit_posts(sub)
        all_posts.extend(sub_posts)
        # Save snapshot per subreddit so partial progress isn't lost
        snapshot_path = os.path.join(DATA_DIR, f"lcr_posts_{sub}.csv")
        pd.DataFrame(sub_posts).to_csv(snapshot_path, index=False)
        print(f"  saved snapshot: {snapshot_path}")
        time.sleep(SLEEP_BETWEEN_REQUESTS * 2)

    posts_df = pd.DataFrame(all_posts)
    print(f"\nTotal posts collected: {len(posts_df)}")
    if posts_df.empty:
        print("No posts to process.")
        return

    # Prefilter
    posts_df["passes_filter"] = posts_df["body"].apply(passes_prefilter)
    relevant_pids = posts_df.loc[posts_df["passes_filter"], "post_id"].dropna().unique().tolist()
    print(f"Posts passing LCR/Claude prefilter: {len(relevant_pids)}")

    # Fetch comments for relevant posts
    all_comments = []
    for sub in SUBREDDITS:
        sub_pids = posts_df.loc[
            (posts_df["subreddit"] == sub) & posts_df["passes_filter"],
            "post_id",
        ].dropna().unique().tolist()
        if not sub_pids:
            continue
        sub_comments = scrape_subreddit_comments(sub, sub_pids)
        all_comments.extend(sub_comments)
        time.sleep(SLEEP_BETWEEN_REQUESTS * 2)

    comments_df = pd.DataFrame(all_comments)
    print(f"\nTotal comments collected: {len(comments_df)}")

    # Combine, dedupe by (post_id+type, comment_id)
    posts_df = posts_df.drop(columns=["passes_filter"])
    full_df = pd.concat([posts_df, comments_df], ignore_index=True)
    full_df = full_df[full_df["body"].str.len().fillna(0) > 30].reset_index(drop=True)
    full_df = full_df.drop_duplicates(
        subset=["type", "comment_id", "post_id", "body"], keep="first"
    ).reset_index(drop=True)
    print(f"After dedup + min-length filter: {len(full_df)} rows")

    out_path = os.path.join(DATA_DIR, "lcr_corpus.csv")
    full_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved LCR corpus: {out_path}")

    # Report coverage
    if not full_df.empty:
        df_dates = pd.to_datetime(full_df["createdAt"])
        print(f"\nLCR corpus coverage:")
        print(f"  Date range: {df_dates.min().date()} to {df_dates.max().date()}")
        print(f"  Subreddit breakdown:")
        for sub, count in full_df["subreddit"].value_counts().items():
            print(f"    {sub}: {count}")
        print(f"  Total rows: {len(full_df)}")
    print(f"\nElapsed: {datetime.now() - start_run}")


if __name__ == "__main__":
    main()
