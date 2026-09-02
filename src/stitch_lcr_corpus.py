"""Stitch the per-subreddit LCR post snapshots into a combined corpus."""

import glob
import html
import os
import re

import pandas as pd

DATA_DIR = "data/"

files = sorted(glob.glob(os.path.join(DATA_DIR, "lcr_posts_*.csv")))
print(f"Found {len(files)} subreddit snapshots")

dfs = []
for f in files:
    df = pd.read_csv(f)
    print(f"  {f}: {len(df)} rows")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined["body"] = combined["body"].astype(str).apply(
    lambda x: html.unescape(re.sub(r"\\", "", x))
)
combined = combined[combined["body"].str.len() > 30].reset_index(drop=True)
combined = combined.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
print(f"After clean+dedup: {len(combined)} posts")

out = os.path.join(DATA_DIR, "lcr_corpus.csv")
combined.to_csv(out, index=False, encoding="utf-8-sig")
print(f"Saved {out}")

dates = pd.to_datetime(combined["createdAt"])
print(f"\nDate range: {dates.min().date()} to {dates.max().date()}")
print(f"Pre-Sept-29-2025 (Sonnet 4.5 release): {(dates < pd.Timestamp('2025-09-29')).sum()}")
print(f"Post-Sept-29-2025: {(dates >= pd.Timestamp('2025-09-29')).sum()}")
print(f"\nSubreddit breakdown:")
print(combined["subreddit"].value_counts().to_string())
