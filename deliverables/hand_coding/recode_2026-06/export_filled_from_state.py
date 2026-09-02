"""Build recode_instrument_blinded_FILLED.csv from the hub autosave state.

The dashboard autosaves its full state to the estate hub, which persists it
as recode_state_autosave.json next to the instrument. This script turns that
state into the FILLED.csv the analysis step expects, so finishing a round
needs no browser download/upload: code on any device, then run this here.

Run: py -3.13 export_filled_from_state.py [--partial]
"""
import csv
import json
import sys
from pathlib import Path

from build_recode_dashboard import SCHEMA

HERE = Path(__file__).resolve().parent
STATE = HERE / "recode_state_autosave.json"
INSTRUMENT = HERE / "recode_instrument_blinded.csv"
OUT = HERE / "recode_instrument_blinded_FILLED.csv"

FIELD_KEYS = [f["key"] for f in SCHEMA]


def composed_notes(c):
    """Mirror the dashboard's composedNotes: free notes, then one segment per
    unclear-flagged field, then the discussion flag."""
    parts = []
    if c.get("recode_notes"):
        parts.append(c["recode_notes"])
    for k, v in (c.get("_unsure") or {}).items():
        parts.append(k + "=unclear" + (": " + v if v else ""))
    if c.get("_discuss"):
        parts.append("flagged_for_discussion")
    return "; ".join(parts)


def cell(h, row, c):
    if h in ("recode_id", "source_csv", "post_id"):
        return row[h]
    if h == "recode_notes":
        return composed_notes(c)
    v = c.get(h, "")
    return "|".join(v) if isinstance(v, list) else v


def main():
    if not STATE.exists():
        sys.exit("no %s yet: nothing has synced from the dashboard" % STATE.name)
    payload = json.loads(STATE.read_text(encoding="utf-8"))
    state = payload.get("state") or {}
    with open(INSTRUMENT, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [dict(zip(header, r)) for r in reader if r and r[0].strip()]
    missing = []
    for r in rows:
        c = state.get(r["recode_id"], {})
        n = sum(1 for k in FIELD_KEYS if c.get(k))
        if n < len(FIELD_KEYS):
            missing.append("%s (%d/%d)" % (r["recode_id"], n, len(FIELD_KEYS)))
    if missing and "--partial" not in sys.argv:
        sys.exit("incomplete, not writing (use --partial to force):\n  "
                 + "\n  ".join(missing))
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            c = state.get(r["recode_id"], {})
            w.writerow([cell(h, r, c) for h in header])
    print("wrote %s (%d cases%s)" % (OUT, len(rows), ", PARTIAL" if missing else ""))


if __name__ == "__main__":
    main()
