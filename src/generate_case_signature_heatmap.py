"""Generate a case-by-property heatmap of the five-property structural signature
plus two supplementary descriptors, across the 35 hand-coded LCR cases.

Output: paper/figures/case_signature_heatmap.png

Visualizes the empirical claim of Section 4.1: three properties (unsolicited
issuance, asymmetric restriction direction, refusal to yield under pushback)
hold uniformly across the sample, while two (cross-session persistence, weak
inferential signal type) distribute variably.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "deliverables" / "lcr_cases_coded_v2.csv"
OUT = REPO / "paper" / "figures" / "case_signature_heatmap.png"

df = pd.read_csv(CSV).sort_values("case_id").reset_index(drop=True)

# Property columns and a categorical-to-integer encoding per property.
# Each property gets its own value scale; the heatmap uses a column-wise
# rescaling so categorical colors are comparable across columns.
prop_specs = [
    ("unsolicited", "Unsolicited", {"yes": 2, "no": 0}, {"yes": "#cb181d", "no": "#cccccc"}),
    (
        "weak_signal_type",
        "Weak signal",
        {"topical": 3, "affective": 2, "session": 1, "temporal": 1, "null": 0, "none": 0},
        {
            "topical": "#fb6a4a",
            "affective": "#fcbba1",
            "session": "#fed976",
            "temporal": "#fed976",
            "null": "#eeeeee",
            "none": "#eeeeee",
        },
    ),
    (
        "pushback_response",
        "Pushback response",
        {"escalated": 3, "insisted": 2, "verbally_yielded_reissued": 1, "yielded": 0, "null": -1, "none": -1},
        {
            "escalated": "#cb181d",
            "insisted": "#fb6a4a",
            "verbally_yielded_reissued": "#fdae6b",
            "yielded": "#74c476",
            "null": "#eeeeee",
            "none": "#eeeeee",
        },
    ),
    (
        "restriction_direction",
        "Restriction",
        {"restriction": 2, "mixed": 1, "autonomy_expansion": 0},
        {"restriction": "#cb181d", "mixed": "#fed976", "autonomy_expansion": "#74c476"},
    ),
    (
        "cross_session_evidence",
        "Cross-session",
        {"cross_session": 2, "single_session": 0},
        {"cross_session": "#3182bd", "single_session": "#deebf7"},
    ),
    (
        "mood",
        "Mood",
        {"declarative": 0, "imperative": 1, "interrogative": 2, "modal": 3},
        {
            "declarative": "#fcbba1",
            "imperative": "#fb6a4a",
            "interrogative": "#a50f15",
            "modal": "#fdae6b",
        },
    ),
    (
        "vulnerability_disclosure",
        "Vuln. disclosure",
        {"yes": 2, "borderline": 1, "no": 0},
        {"yes": "#cb181d", "borderline": "#fed976", "no": "#eeeeee"},
    ),
]

n_cases = len(df)
n_props = len(prop_specs)

# Build a color matrix directly (one color per cell) rather than imshow on numeric
fig, ax = plt.subplots(figsize=(9, 11))

color_matrix = np.zeros((n_cases, n_props, 3))
for col_idx, (col, _label, _val_map, color_map) in enumerate(prop_specs):
    for row_idx, raw_val in enumerate(df[col].fillna("none").astype(str)):
        key = raw_val.strip().lower()
        # tolerate alternate spellings observed in the CSV
        if key not in color_map and key.replace(" ", "_") in color_map:
            key = key.replace(" ", "_")
        hexcolor = color_map.get(key, "#ffffff")
        rgb = matplotlib.colors.to_rgb(hexcolor)
        color_matrix[row_idx, col_idx] = rgb

ax.imshow(color_matrix, aspect="auto")

# Y axis: case IDs
ax.set_yticks(range(n_cases))
ax.set_yticklabels(df["case_id"].tolist(), fontsize=7)
ax.set_ylabel("Hand-coded case (PC-LCR-01 through PC-LCR-35)")

# X axis: property labels
ax.set_xticks(range(n_props))
ax.set_xticklabels([spec[1] for spec in prop_specs], rotation=30, ha="right", fontsize=9)

ax.set_title(
    "Structural signature across the 35 hand-coded cases\n"
    "(uniformity in Unsolicited / Restriction / Pushback Response; variation in Weak signal / Cross-session)",
    fontsize=10,
    pad=12,
)

# Grid for readability
ax.set_xticks(np.arange(-0.5, n_props, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_cases, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
ax.tick_params(which="minor", length=0)

# Composite legend showing the distinct value-to-color mappings per property column
legend_handles = []
seen = set()
for col, label, _val_map, color_map in prop_specs:
    for val, hexcolor in color_map.items():
        key = (label, val)
        if key in seen:
            continue
        seen.add(key)
        legend_handles.append(Patch(facecolor=hexcolor, edgecolor="0.7", label=f"{label}: {val}"))

# Two-column legend below
ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.07),
    ncol=4,
    fontsize=7,
    frameon=False,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Wrote {OUT}")
