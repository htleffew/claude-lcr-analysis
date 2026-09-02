"""Targeted corpus search for yield-to-pushback candidates (contrast-class work, 2026-06).

Searches the intact LCR corpus for posts/comments in which a user describes the model
*retreating* from a concern/pathologizing/overprotective framing after user pushback.
Output: deliverables/term_validation/yield_search_candidates.csv for hand review.

Rationale (paper section 4.1 / 6.x): the hand-coded sample reports 0 yielded / 5 escalated
among documented-pushback cases. A corpus-wide targeted search for yield narratives tests
whether yielding is recoverable from the corpus at all. Both outcomes are reportable.
"""

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "lcr_corpus_intact.csv"
OUT = ROOT / "deliverables" / "term_validation" / "yield_search_candidates.csv"

csv.field_size_limit(10_000_000)

# Yield-narrative patterns: model retreats from concern/pathologizing framing after pushback.
YIELD_PATTERNS = [
    r"(it|he|she|claude)\s+(went back and\s+)?(reviewed|reread|re-read).{0,60}(agreed|admitted|apologi[sz]ed)",
    r"(told|asked)\s+(it|claude|him|her)\b.{0,80}(patroni[sz]|overprotect|condescend|pathologi[sz]|gaslight|therapy|therapist|mental health|concern)",
    r"(apologi[sz]ed|backed (off|down)|dropped|stopped|relented|conceded|softened|let it go|moved on)\b.{0,120}(concern|patroni[sz]|overprotect|pathologi[sz]|mental health|therap|wellness|psychiatric|wellbeing|well-being)",
    r"(concern|patroni[sz]\w*|overprotect\w*|pathologi[sz]\w*|mental health|therap\w*|psychiatric)\b.{0,120}(apologi[sz]ed|backed (off|down)|dropped (it|the)|stopped (doing|suggesting|saying)|relented|conceded|softened)",
    r"acknowledged\s+(it|he|she)?\s*(might be|was)\s+wrong",
    r"adjust(ed)?\s+(its\s+|his\s+|her\s+)?(tone|approach|behavior).{0,80}(overprotect|patroni[sz]|concern|pushback|told)",
]

COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in YIELD_PATTERNS]


def main() -> None:
    rows_out = []
    with open(CORPUS, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        text_cols = [c for c in fields if c and c.lower() in
                     ("body", "selftext", "text", "content", "title")]
        id_cols = [c for c in fields if c and c.lower() in
                   ("id", "post_id", "comment_id", "name", "link_id", "parent_id")]
        meta_cols = [c for c in fields if c and c.lower() in
                     ("subreddit", "created_utc", "created", "row_type", "type", "provenance")]
        if not text_cols:
            # Fall back: scan all string columns
            text_cols = fields
        for row in reader:
            text = " ".join(str(row.get(c, "") or "") for c in text_cols)
            if len(text) < 40:
                continue
            for i, pat in enumerate(COMPILED):
                m = pat.search(text)
                if m:
                    start = max(0, m.start() - 200)
                    end = min(len(text), m.end() + 200)
                    rows_out.append({
                        "pattern_id": i,
                        "match": m.group(0)[:200],
                        "context": text[start:end].replace("\n", " "),
                        **{c: row.get(c, "") for c in id_cols + meta_cols},
                    })
                    break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if rows_out:
        with open(OUT, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
    print(f"columns: {fields}")
    print(f"candidates: {len(rows_out)} -> {OUT}")
    by_pat = {}
    for r in rows_out:
        by_pat[r["pattern_id"]] = by_pat.get(r["pattern_id"], 0) + 1
    print(f"by pattern: {by_pat}")


if __name__ == "__main__":
    sys.exit(main())
