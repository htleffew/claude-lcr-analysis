"""
LCR stub-post exclusion per researcher decision 2026-05-17.

Reddit removed the body of 29.3% of posts in the LCR corpus, leaving body =
title + '[removed]'. These stubs contribute artifact tokens ('removed', 'by
moderator', 'removed by moderator') to frequency tables.

Per the Phase 2 checkpoint of 2026-05-17, the researcher selected "Exclude
entirely": drop stubs, work with the intact-body subset (~21,986 posts).

This script:
- Reads data/lcr_corpus.csv (the wholesale Pass 1a corpus from Arctic Shift)
- Flags stubs by detecting '[removed]', '[deleted]', or
  'removed by moderator' as the post body content
- Writes:
  - data/lcr_corpus_full_with_stub_flag.csv : full corpus with is_stub column
    (preserves the original for provenance and possible later analysis)
  - data/lcr_corpus_intact.csv : the canonical operating corpus going forward,
    stubs excluded
- Reports per-subreddit stub rate and final corpus size.

The intact-body corpus is what subsequent Phase 2/2.5/3+ work consumes.
"""

import re

import pandas as pd

DATA_DIR = "data/"

SRC = DATA_DIR + "lcr_corpus.csv"
FULL_FLAGGED = DATA_DIR + "lcr_corpus_full_with_stub_flag.csv"
INTACT = DATA_DIR + "lcr_corpus_intact.csv"

STUB_PATTERNS = [
    r"\[removed\]",
    r"\[deleted\]",
    r"removed by moderator",
    r"deleted by user",
]
STUB_RE = re.compile("|".join(STUB_PATTERNS), re.IGNORECASE)


def is_stub(body: str) -> bool:
    """Return True if the post body is a stub artifact rather than real content."""
    if not isinstance(body, str):
        return True
    text = body.strip()
    if len(text) == 0:
        return True
    if STUB_RE.search(text):
        # Heuristic: if 70% or more of the body is the stub marker plus title,
        # consider it a stub. This catches title + '[removed]' but not posts
        # that quote '[removed]' in passing.
        marker_match = STUB_RE.search(text)
        if marker_match:
            non_marker_len = len(text) - (marker_match.end() - marker_match.start())
            if non_marker_len < 200:
                return True
    return False


def main():
    df = pd.read_csv(SRC)
    print(f"Loaded {len(df):,} rows from {SRC}")

    df["is_stub"] = df["body"].apply(is_stub)

    n_total = len(df)
    n_stub = int(df["is_stub"].sum())
    n_intact = n_total - n_stub

    print(f"\nStub detection:")
    print(f"  Total posts: {n_total:,}")
    print(f"  Stubs: {n_stub:,} ({100 * n_stub / n_total:.1f}%)")
    print(f"  Intact: {n_intact:,} ({100 * n_intact / n_total:.1f}%)")

    print(f"\nStub rate by subreddit:")
    by_sub = df.groupby("subreddit")["is_stub"].agg(["sum", "count"])
    by_sub["rate"] = by_sub["sum"] / by_sub["count"]
    by_sub = by_sub.sort_values("rate", ascending=False)
    for sub, row in by_sub.iterrows():
        print(
            f"  {sub:20s} stubs={int(row['sum']):5,}  "
            f"total={int(row['count']):5,}  rate={row['rate'] * 100:5.1f}%"
        )

    df.to_csv(FULL_FLAGGED, index=False, encoding="utf-8-sig")
    print(f"\nSaved full corpus with is_stub flag: {FULL_FLAGGED}")

    df_intact = df[~df["is_stub"]].drop(columns=["is_stub"]).reset_index(drop=True)
    df_intact.to_csv(INTACT, index=False, encoding="utf-8-sig")
    print(f"Saved intact-body operating corpus: {INTACT} ({len(df_intact):,} rows)")

    if not df_intact.empty:
        dates = pd.to_datetime(df_intact["createdAt"])
        print(f"\nIntact corpus coverage:")
        print(f"  Date range: {dates.min().date()} to {dates.max().date()}")
        print(f"  Pre-Sep-29-2025: {(dates < pd.Timestamp('2025-09-29')).sum():,}")
        print(f"  Post-Sep-29-2025: {(dates >= pd.Timestamp('2025-09-29')).sum():,}")


if __name__ == "__main__":
    main()
