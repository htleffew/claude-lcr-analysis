"""Generate a 2D UMAP projection of the 485 KWIC contexts of bare *professional*,
colored by HDBSCAN cluster assignment (MiniLM-L6-v2 embeddings, mcs=10).

Output: paper/figures/sense_discovery_umap.png

Cluster colors are grouped by semantic register:
  - User-vocabulary registers (Clusters 0, 1, 2): blue family
  - LCR-payload registers     (Clusters 3, 4, 5): red family
  - Unclustered noise         (Cluster -1):        light gray
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

REPO = Path(__file__).resolve().parent.parent
SENSE_DIR = REPO / "deliverables" / "phase_2_5_sense_discovery" / "professional"
OUT = REPO / "paper" / "figures" / "sense_discovery_umap.png"

embeddings = np.load(SENSE_DIR / "embeddings_professional_minilm.npy")
clusters = pd.read_csv(SENSE_DIR / "clusters_professional_minilm_mcs10.csv")

# Order of rows in embeddings matches order in the clusters CSV
cluster_ids = clusters["cluster_id"].to_numpy()

# UMAP reproducibly to 2D
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
coords = reducer.fit_transform(embeddings)

# Color scheme: user-vocabulary blue family, LCR-payload red family, noise gray
cluster_label = {
    -1: "Unclustered (noise)",
    0: "Cluster 0: professional contexts (user-vocabulary)",
    1: "Cluster 1: professional devs (user-vocabulary)",
    2: "Cluster 2: professional workflow (user-vocabulary)",
    3: "Cluster 3: professional oversight (LCR-payload)",
    4: "Cluster 4: verbatim LCR system-prompt (LCR-payload)",
    5: "Cluster 5: LCR-bound 'professional' (LCR-payload)",
}
cluster_color = {
    -1: "#cfcfcf",
    0: "#9ecae1",
    1: "#6baed6",
    2: "#3182bd",
    3: "#fcbba1",
    4: "#fb6a4a",
    5: "#cb181d",
}

fig, ax = plt.subplots(figsize=(8, 7))
for cid in sorted(cluster_color.keys()):
    mask = cluster_ids == cid
    if mask.any():
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=cluster_color[cid],
            s=22 if cid != -1 else 8,
            alpha=0.75 if cid != -1 else 0.35,
            edgecolors="none",
            label=cluster_label[cid],
        )

ax.set_xlabel("UMAP dimension 1")
ax.set_ylabel("UMAP dimension 2")
ax.set_title(
    "Sense-discovery of 'professional' (n = 485 KWIC contexts, MiniLM-L6-v2, mcs = 10)"
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend below the plot to keep the panel uncluttered
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    fontsize=8,
    frameon=False,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Wrote {OUT}")
