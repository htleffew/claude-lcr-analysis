import json, numpy as np
from pathlib import Path
import pandas as pd

OUT_DIR = Path(r"C:/Users/drhea/estate/projects/research/claude-lcr\deliverables\phase_2_5_sense_discovery\professional")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

kwic_df = pd.read_csv(OUT_DIR / "kwic_full_professional.csv")
synfeat_df = pd.read_csv(OUT_DIR / "syntactic_features_per_cluster_professional.csv")
stab_df = pd.read_csv(OUT_DIR / "stability_crosstable_professional.csv")

cross_mask = stab_df.apply(
    lambda r: r["config_a"].split("_")[0] != r["config_b"].split("_")[0]
    and r["config_a"].split("_")[1] == r["config_b"].split("_")[1], axis=1)
mean_cross = stab_df.loc[cross_mask, "ari"].mean()

within_mask = stab_df.apply(
    lambda r: r["config_a"].split("_")[0] == r["config_b"].split("_")[0]
    and r["config_a"] != r["config_b"], axis=1)
mean_within = stab_df.loc[within_mask, "ari"].mean()

summary = {
    "seed": "professional",
    "total_occ": int(len(kwic_df)),
    "total_posts": int(kwic_df["post_index"].nunique()),
    "ph_pm5_count": int(kwic_df["has_professional_help_pm5"].sum()),
    "ph_pm5_frac": round(float(kwic_df["has_professional_help_pm5"].mean()), 4),
    "mean_cross_ari": round(float(mean_cross), 4),
    "mean_within_ari": round(float(mean_within), 4),
    "canonical_clusters": [int(c) for c in sorted(synfeat_df[synfeat_df["cluster_id"] != -1]["cluster_id"].tolist())],
    "config_stats": [
        ["minilm_mcs5", 17, 0.652],
        ["minilm_mcs10", 6, 0.720],
        ["minilm_mcs20", 3, 0.790],
        ["mpnet_mcs5", 18, 0.594],
        ["mpnet_mcs10", 5, 0.769],
        ["mpnet_mcs20", 2, 0.773],
    ],
}

(OUT_DIR / "sense_discovery_summary_data.json").write_text(
    json.dumps(summary, indent=2, cls=NpEncoder), encoding="utf-8"
)
print("Summary JSON saved.")
print("Occurrences:", summary["total_occ"])
print("Posts:", summary["total_posts"])
print("PH pm5 count:", summary["ph_pm5_count"])
print("PH pm5 frac:", summary["ph_pm5_frac"])
print("Mean cross ARI:", summary["mean_cross_ari"])
print("Mean within ARI:", summary["mean_within_ari"])
