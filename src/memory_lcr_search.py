"""Search the corpus for cross-conversation memory mentions co-occurring with LCR/pathologizing
context (memory-confound check, 2026-06-12).

Question: do user reports indicate the memory feature (or past-chat search) supplied
cross-session information that bolstered the LCR's diagnostic determinations?
Output: deliverables/term_validation/memory_lcr_candidates.csv for hand review.
"""

import csv
import re

csv.field_size_limit(10_000_000)

CORPUS = "data/lcr_corpus_intact.csv"
OUT = "deliverables/term_validation/memory_lcr_candidates.csv"

MEMORY = re.compile(
    r"(memory feature|memories|cross[- ]chat|across (chats|conversations|sessions)|"
    r"past (chats|conversations|sessions)|previous (chat|conversation|session)s?|"
    r"other (chats|conversations|sessions)|earlier (chat|conversation|session)s?|"
    r"reference chats|search(ed)? (my |the )?(past|previous|other) (chats|conversations)|"
    r"remember(ed|s)? (what|that|things|details|information|me telling)|"
    r"brought up (something|things|what) (I|from)|incognito)",
    re.IGNORECASE,
)

LCR_CONTEXT = re.compile(
    r"(LCR|long conversation reminder|pathologiz|therapist|psychiatric|"
    r"mental health|diagnos|concern(ed|ing)? (about|for) (you|my|the user)|"
    r"professional help|wellbeing|well-being|mania|psychosis|dissociation|"
    r"detachment from reality|see a professional|speak with a professional)",
    re.IGNORECASE,
)

rows_out = []
with open(CORPUS, encoding="utf-8", errors="replace", newline="") as f:
    r = csv.DictReader(f)
    pid = r.fieldnames[0]
    for row in r:
        body = row.get("body", "") or ""
        if len(body) < 60:
            continue
        m1 = MEMORY.search(body)
        if not m1:
            continue
        m2 = LCR_CONTEXT.search(body)
        if not m2:
            continue
        # proximity: require the two matches within 1500 chars to cut topical noise
        if abs(m1.start() - m2.start()) > 1500:
            continue
        lo = max(0, min(m1.start(), m2.start()) - 150)
        hi = min(len(body), max(m1.end(), m2.end()) + 150)
        rows_out.append({
            "post_id": row[pid],
            "subreddit": row.get("subreddit", ""),
            "type": row.get("type", ""),
            "createdAt": row.get("createdAt", ""),
            "memory_match": m1.group(0)[:80],
            "lcr_match": m2.group(0)[:80],
            "context": body[lo:hi].replace("\n", " "),
        })

with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else
                       ["post_id", "subreddit", "type", "createdAt", "memory_match", "lcr_match", "context"])
    w.writeheader()
    w.writerows(rows_out)

print(f"candidates: {len(rows_out)} -> {OUT}")
for x in rows_out[:25]:
    print(x["post_id"], "|", x["createdAt"][:10], "|", x["memory_match"], "<->", x["lcr_match"])
