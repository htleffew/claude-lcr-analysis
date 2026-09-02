"""
Phase 2.5 Sense-Discovery: bare `professional`
Methods library §1.8 — Sense-discovery via embedding-cluster on KWIC contexts.

Input:  data/lcr_corpus_intact.csv (22,008 stub-excluded posts)
Output: deliverables/phase_2_5_sense_discovery/professional/
        - kwic_full_professional.csv
        - embeddings_professional_minilm.npy + index CSV
        - embeddings_professional_mpnet.npy + index CSV
        - clusters_professional_minilm_mcs{5,10,20}.csv
        - clusters_professional_mpnet_mcs{5,10,20}.csv
        - stability_crosstable_professional.csv
        - exemplars_professional.csv
        - syntactic_features_per_cluster_professional.csv
        - sense_discovery_notes_professional.md

Run: python src/phase_2_5_sense_discovery_professional.py
"""

import re
import csv
import json
import time
import warnings
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(r"C:/Users/drhea/estate/projects/research/claude-lcr")
DATA_PATH = REPO_ROOT / "data" / "lcr_corpus_intact.csv"
OUT_DIR   = REPO_ROOT / "deliverables" / "phase_2_5_sense_discovery" / "professional"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED      = "professional"
WINDOW    = 20   # ±20 tokens
MCS_LIST  = [5, 10, 20]

# ---------------------------------------------------------------------------
# 1. Load corpus
# ---------------------------------------------------------------------------
print("Loading corpus …")
df = pd.read_csv(DATA_PATH, low_memory=False)
# Body column name may vary; try common names
body_col = None
for c in ["body", "selftext", "text", "content", "post_body"]:
    if c in df.columns:
        body_col = c
        break
if body_col is None:
    raise ValueError(f"No recognised body column. Columns: {list(df.columns)}")

id_col = None
for c in ["id", "post_id", "name"]:
    if c in df.columns:
        id_col = c
        break

sub_col = None
for c in ["subreddit", "subreddit_name"]:
    if c in df.columns:
        sub_col = c
        break

print(f"  Rows: {len(df):,}  |  body='{body_col}'  id='{id_col}'  sub='{sub_col}'")

# ---------------------------------------------------------------------------
# 2. KWIC extraction — ±20 token context, raw text, stop-words preserved
# ---------------------------------------------------------------------------
print("Extracting KWIC windows …")

SEED_RE = re.compile(r'\b' + re.escape(SEED) + r'\b', re.IGNORECASE)
# Tokenize by whitespace (preserves punctuation attached to words — raw as specified)
TOKEN_RE = re.compile(r'\S+')

kwic_rows = []
for idx, row in df.iterrows():
    body = str(row[body_col]) if pd.notna(row[body_col]) else ""
    # tokenize
    tokens = TOKEN_RE.findall(body)
    token_str = " ".join(tokens)

    # find all occurrences of the seed in the original body (case-insensitive)
    for m in SEED_RE.finditer(body):
        # find the token index of the match start
        # Map character offset to token index
        char_start = m.start()
        # rebuild token offsets
        token_offsets = []
        pos = 0
        for tok in tokens:
            found = body.find(tok, pos)
            if found >= 0:
                token_offsets.append((found, found + len(tok), tok))
                pos = found + len(tok)

        # find which token contains char_start
        seed_tok_idx = None
        for ti, (ts, te, _) in enumerate(token_offsets):
            if ts <= char_start < te:
                seed_tok_idx = ti
                break
        if seed_tok_idx is None:
            # fallback: scan for seed token
            seed_lower = SEED.lower()
            for ti, (_, _, tok) in enumerate(token_offsets):
                if tok.lower() == seed_lower or tok.lower().strip(".,;:!?\"'") == seed_lower:
                    seed_tok_idx = ti
                    break

        if seed_tok_idx is None:
            continue

        left_idx  = max(0, seed_tok_idx - WINDOW)
        right_idx = min(len(tokens), seed_tok_idx + WINDOW + 1)
        left_ctx  = " ".join(tokens[left_idx:seed_tok_idx])
        seed_tok  = tokens[seed_tok_idx]
        right_ctx = " ".join(tokens[seed_tok_idx + 1:right_idx])
        full_ctx  = " ".join(tokens[left_idx:right_idx])

        # check for code block within full context
        has_code_block = bool(re.search(r'```|    \S', full_ctx))

        # check for professional help within ±5 tokens of seed position
        ph_left  = max(0, seed_tok_idx - 5)
        ph_right = min(len(tokens), seed_tok_idx + 6)
        ph_window_str = " ".join(tokens[ph_left:ph_right])
        has_professional_help = bool(
            re.search(r'\bprofessional\s+help\b', ph_window_str, re.IGNORECASE)
        )

        kwic_rows.append({
            "kwic_id":              len(kwic_rows),
            "post_index":           idx,
            "post_id":              row[id_col] if id_col else "",
            "subreddit":            row[sub_col] if sub_col else "",
            "seed_token":           seed_tok,
            "left_context":         left_ctx,
            "right_context":        right_ctx,
            "full_context":         full_ctx,
            "seed_tok_idx":         seed_tok_idx,
            "has_code_block":       has_code_block,
            "has_professional_help_pm5": has_professional_help,
        })

kwic_df = pd.DataFrame(kwic_rows)
kwic_path = OUT_DIR / "kwic_full_professional.csv"
kwic_df.to_csv(kwic_path, index=False, encoding="utf-8")
print(f"  Total occurrences:  {len(kwic_df):,}")
print(f"  Posts represented:  {kwic_df['post_index'].nunique():,}")
print(f"  Saved -> {kwic_path.name}")

if len(kwic_df) < 50:
    raise ValueError(
        f"Only {len(kwic_df)} occurrences — below the floor for clustering. "
        "Hand-read all contexts instead."
    )

# ---------------------------------------------------------------------------
# 3. Embed contexts with sentence-transformers
# ---------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet":  "sentence-transformers/all-mpnet-base-v2",
}

contexts = kwic_df["full_context"].tolist()
embeddings_by_model = {}

for short_name, model_name in MODELS.items():
    print(f"Embedding with {model_name} …")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    emb = model.encode(contexts, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True)
    elapsed = time.time() - t0
    print(f"  Embedding shape: {emb.shape}  |  {elapsed:.1f}s")

    npy_path = OUT_DIR / f"embeddings_professional_{short_name}.npy"
    np.save(npy_path, emb)

    idx_path = OUT_DIR / f"embeddings_professional_{short_name}_index.csv"
    kwic_df[["kwic_id", "post_id", "subreddit"]].to_csv(idx_path, index=False, encoding="utf-8")

    embeddings_by_model[short_name] = emb
    print(f"  Saved -> {npy_path.name}")

# ---------------------------------------------------------------------------
# 4. HDBSCAN clustering
# ---------------------------------------------------------------------------
import hdbscan

cluster_labels_all = {}  # key: (model, mcs) -> labels array

for short_name, emb in embeddings_by_model.items():
    for mcs in MCS_LIST:
        print(f"HDBSCAN  model={short_name}  mcs={mcs} …")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(emb)
        key = (short_name, mcs)
        cluster_labels_all[key] = labels

        out_df = kwic_df[["kwic_id", "post_id", "subreddit", "full_context",
                           "has_code_block", "has_professional_help_pm5"]].copy()
        out_df["cluster_id"] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac  = (labels == -1).mean()
        print(f"  Clusters: {n_clusters}  |  Noise fraction: {noise_frac:.3f}")

        fname = f"clusters_professional_{short_name}_mcs{mcs}.csv"
        out_df.to_csv(OUT_DIR / fname, index=False, encoding="utf-8")
        print(f"  Saved -> {fname}")

# ---------------------------------------------------------------------------
# 5. Stability cross-table (Adjusted Rand Index + NMI)
# ---------------------------------------------------------------------------
print("Computing stability cross-table …")
config_keys = list(cluster_labels_all.keys())
n_configs = len(config_keys)
config_names = [f"{m}_mcs{mc}" for m, mc in config_keys]

ari_matrix = np.zeros((n_configs, n_configs))
nmi_matrix = np.zeros((n_configs, n_configs))

for i in range(n_configs):
    for j in range(n_configs):
        la = cluster_labels_all[config_keys[i]]
        lb = cluster_labels_all[config_keys[j]]
        # mask noise in both
        mask = (la != -1) & (lb != -1)
        if mask.sum() < 2:
            ari_matrix[i, j] = np.nan
            nmi_matrix[i, j] = np.nan
        else:
            ari_matrix[i, j] = adjusted_rand_score(la[mask], lb[mask])
            nmi_matrix[i, j] = normalized_mutual_info_score(la[mask], lb[mask])

stab_rows = []
for i in range(n_configs):
    for j in range(n_configs):
        stab_rows.append({
            "config_a":    config_names[i],
            "config_b":    config_names[j],
            "ari":         round(ari_matrix[i, j], 4) if not np.isnan(ari_matrix[i, j]) else "",
            "nmi":         round(nmi_matrix[i, j], 4) if not np.isnan(nmi_matrix[i, j]) else "",
        })

stab_df = pd.DataFrame(stab_rows)
stab_path = OUT_DIR / "stability_crosstable_professional.csv"
stab_df.to_csv(stab_path, index=False, encoding="utf-8")
print(f"  Saved -> {stab_path.name}")

# ---------------------------------------------------------------------------
# 6. Sample exemplars — canonical config: minilm + mcs=10
# ---------------------------------------------------------------------------
print("Sampling exemplars (canonical: minilm + mcs=10) …")
from sklearn.metrics.pairwise import cosine_distances

canon_key    = ("minilm", 10)
canon_labels = cluster_labels_all[canon_key]
canon_emb    = embeddings_by_model["minilm"]

EXEMPLAR_N = 10
exemplar_rows = []

unique_clusters = sorted(c for c in set(canon_labels) if c != -1)
for cl in unique_clusters:
    mask = canon_labels == cl
    cl_embs  = canon_emb[mask]
    cl_idxs  = np.where(mask)[0]
    centroid = cl_embs.mean(axis=0, keepdims=True)
    dists    = cosine_distances(centroid, cl_embs)[0]
    top_n    = min(EXEMPLAR_N, len(cl_idxs))
    nearest  = cl_idxs[np.argsort(dists)[:top_n]]
    for rank, ni in enumerate(nearest):
        row = kwic_df.iloc[ni].to_dict()
        row["cluster_id"]           = cl
        row["centroid_rank"]        = rank + 1
        row["cosine_dist_centroid"] = round(dists[np.argsort(dists)[rank]], 4)
        exemplar_rows.append(row)

exemplar_df = pd.DataFrame(exemplar_rows)
ex_path = OUT_DIR / "exemplars_professional.csv"
exemplar_df.to_csv(ex_path, index=False, encoding="utf-8")
print(f"  Exemplar rows: {len(exemplar_df)}  |  Saved -> {ex_path.name}")

# ---------------------------------------------------------------------------
# 7. Syntactic-feature cross-validation
# ---------------------------------------------------------------------------
print("Computing syntactic features per cluster …")
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# POS of `professional` in context
seed_lower = SEED.lower()

def get_seed_pos(context_str):
    """Return POS tag of the seed token in the context."""
    doc = nlp(context_str[:2000])  # trim for speed
    for tok in doc:
        if tok.text.lower() == seed_lower:
            return tok.pos_
    return "UNKNOWN"

# Imperative mood heuristic: seed preceded by you/should/go/try/please/consider/speak
IMPERATIVE_RE = re.compile(
    r'(?:you\s+should|you\s+might|please|consider|try|go\s+see|go\s+get|speak\s+(?:with|to)|reach\s+out|consult)\s+(?:\w+\s+){0,3}professional',
    re.IGNORECASE
)

synfeat_rows = []
kwic_df["cluster_id_canon"] = canon_labels  # attach canonical labels

for cl in sorted(set(canon_labels)):
    sub = kwic_df[kwic_df["cluster_id_canon"] == cl]
    n = len(sub)

    # Code block fraction
    code_frac = sub["has_code_block"].mean()

    # professional help within ±5 tokens fraction
    ph_frac = sub["has_professional_help_pm5"].mean()

    # subreddit distribution
    sub_dist = sub["subreddit"].value_counts().to_dict() if sub_col else {}

    # POS distribution (sample up to 50 for speed)
    sample_sub = sub.sample(min(50, n), random_state=42)
    pos_counts = {}
    for _, rw in sample_sub.iterrows():
        pos = get_seed_pos(rw["full_context"])
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    # Imperative-mood fraction (heuristic)
    imp_count = sub["full_context"].str.contains(IMPERATIVE_RE).sum()
    imp_frac  = imp_count / n if n > 0 else 0.0

    # Dominant subreddit
    dom_sub = max(sub_dist, key=sub_dist.get) if sub_dist else ""

    synfeat_rows.append({
        "cluster_id":               cl,
        "label":                    "NOISE" if cl == -1 else f"C{cl}",
        "n_contexts":               n,
        "code_block_frac":          round(code_frac, 3),
        "professional_help_pm5_frac": round(ph_frac, 3),
        "imperative_mood_frac":     round(imp_frac, 3),
        "dominant_subreddit":       dom_sub,
        "subreddit_dist_json":      json.dumps(sub_dist),
        "pos_dist_json":            json.dumps(pos_counts),
    })

synfeat_df = pd.DataFrame(synfeat_rows)
sf_path = OUT_DIR / "syntactic_features_per_cluster_professional.csv"
synfeat_df.to_csv(sf_path, index=False, encoding="utf-8")
print(f"  Saved -> {sf_path.name}")

# ---------------------------------------------------------------------------
# 8. Write sense_discovery_notes_professional.md
# ---------------------------------------------------------------------------
print("Writing sense_discovery_notes_professional.md …")

# Gather cluster counts and noise fracs per config
config_stats = []
for (mn, mcs), labels in cluster_labels_all.items():
    n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
    nf     = (labels == -1).mean()
    config_stats.append((f"{mn}_mcs{mcs}", n_cl, round(nf, 3)))

# stability summary: mean ARI between minilm and mpnet (cross-model)
cross_ari_vals = []
for i, (mn_i, mcs_i) in enumerate(config_keys):
    for j, (mn_j, mcs_j) in enumerate(config_keys):
        if mn_i != mn_j and mcs_i == mcs_j:
            v = ari_matrix[i, j]
            if not np.isnan(v):
                cross_ari_vals.append(v)
mean_cross_ari = np.mean(cross_ari_vals) if cross_ari_vals else float("nan")

# within-model stability: same model, varying mcs
within_ari_vals = []
for i, (mn_i, mcs_i) in enumerate(config_keys):
    for j, (mn_j, mcs_j) in enumerate(config_keys):
        if mn_i == mn_j and i != j:
            v = ari_matrix[i, j]
            if not np.isnan(v):
                within_ari_vals.append(v)
mean_within_ari = np.mean(within_ari_vals) if within_ari_vals else float("nan")

# Build exemplar text per cluster (canonical)
def fmt_exemplar(row):
    left  = str(row.get("left_context", "")).strip()
    seed  = str(row.get("seed_token",   "")).strip()
    right = str(row.get("right_context","")).strip()
    ctx = f"…{left} **[{seed}]** {right}…"
    return textwrap.shorten(ctx, width=220, placeholder="…")

canon_exemplar_dict = {}
for cl in unique_clusters:
    cl_ex = exemplar_df[exemplar_df["cluster_id"] == cl].head(3)
    canon_exemplar_dict[cl] = [fmt_exemplar(r) for _, r in cl_ex.iterrows()]

# Per-cluster synfeat lookup
sf_lookup = synfeat_df.set_index("cluster_id").to_dict("index")

# Total occurrences / posts
total_occ   = len(kwic_df)
total_posts = kwic_df["post_index"].nunique()

notes_lines = []
notes_lines.append("# Sense-Discovery Notes: `professional`")
notes_lines.append("")
notes_lines.append(f"**Corpus:** `lcr_corpus_intact.csv` — 22,008 intact-body posts (stub-excluded)")
notes_lines.append(f"**Date:** 2026-05-17")
notes_lines.append(f"**Method:** §1.8 sense-discovery via embedding-cluster on KWIC contexts")
notes_lines.append(f"**Canonical config:** `minilm + mcs=10`")
notes_lines.append("")
notes_lines.append("---")
notes_lines.append("")
notes_lines.append("## 1. Occurrence summary")
notes_lines.append("")
notes_lines.append(f"| Metric | Value |")
notes_lines.append(f"|---|---|")
notes_lines.append(f"| Total occurrences of bare `professional` | {total_occ:,} |")
notes_lines.append(f"| Posts represented | {total_posts:,} |")
notes_lines.append(f"| KWIC window | ±20 tokens, raw text, stop-words preserved |")
notes_lines.append(f"| Occurrences with `professional help` within ±5 tokens | {kwic_df['has_professional_help_pm5'].sum()} ({kwic_df['has_professional_help_pm5'].mean():.1%}) |")
notes_lines.append("")
notes_lines.append("---")
notes_lines.append("")
notes_lines.append("## 2. Cluster counts and noise fractions")
notes_lines.append("")
notes_lines.append("| Configuration | Clusters | Noise fraction |")
notes_lines.append("|---|---|---|")
for cfg, n_cl, nf in config_stats:
    notes_lines.append(f"| `{cfg}` | {n_cl} | {nf:.3f} |")
notes_lines.append("")
notes_lines.append("---")
notes_lines.append("")
notes_lines.append("## 3. Stability summary")
notes_lines.append("")
notes_lines.append(f"| Comparison | Mean ARI |")
notes_lines.append(f"|---|---|")
notes_lines.append(f"| Cross-model (minilm vs. mpnet, same mcs) | {mean_cross_ari:.3f} |")
notes_lines.append(f"| Within-model (same model, varying mcs) | {mean_within_ari:.3f} |")
notes_lines.append("")
notes_lines.append("Full pairwise ARI and NMI table: `stability_crosstable_professional.csv`")
notes_lines.append("")
notes_lines.append("Stability interpretation guidance:")
notes_lines.append("- ARI >= 0.60: strong stability across the pair")
notes_lines.append("- ARI 0.40–0.59: moderate stability")
notes_lines.append("- ARI < 0.40: clusters are configuration-sensitive; treat as unstable")
notes_lines.append("")
notes_lines.append("---")
notes_lines.append("")
notes_lines.append("## 4. Per-cluster profiles (canonical: minilm + mcs=10)")
notes_lines.append("")
notes_lines.append(
    "Sense labels are NOT assigned here. Labels are a researcher decision at the Pattern A checkpoint."
)
notes_lines.append("")

noise_sf = sf_lookup.get(-1, {})
if noise_sf:
    notes_lines.append(f"### Noise (cluster -1)")
    notes_lines.append(f"- **n:** {noise_sf.get('n_contexts', 'n/a')}")
    notes_lines.append(f"- Code-block fraction: {noise_sf.get('code_block_frac', 'n/a')}")
    notes_lines.append(f"- `professional help` within ±5 tokens: {noise_sf.get('professional_help_pm5_frac', 'n/a')}")
    notes_lines.append(f"- Imperative-mood fraction: {noise_sf.get('imperative_mood_frac', 'n/a')}")
    notes_lines.append(f"- Dominant subreddit: {noise_sf.get('dominant_subreddit', 'n/a')}")
    notes_lines.append("")

for cl in unique_clusters:
    sf = sf_lookup.get(cl, {})
    notes_lines.append(f"### Cluster {cl}")
    notes_lines.append(f"- **n:** {sf.get('n_contexts', 'n/a')}")
    notes_lines.append(f"- Code-block fraction: {sf.get('code_block_frac', 'n/a')}")
    notes_lines.append(f"- `professional help` within ±5 tokens: **{sf.get('professional_help_pm5_frac', 'n/a')}**")
    notes_lines.append(f"- Imperative-mood fraction: {sf.get('imperative_mood_frac', 'n/a')}")
    notes_lines.append(f"- Dominant subreddit: {sf.get('dominant_subreddit', 'n/a')}")
    notes_lines.append(f"- POS distribution: {sf.get('pos_dist_json', 'n/a')}")
    notes_lines.append("")
    notes_lines.append("  **Centroid-nearest exemplars (3 of 10; full list in `exemplars_professional.csv`):**")
    for ei, ex in enumerate(canon_exemplar_dict.get(cl, [])[:3], 1):
        notes_lines.append(f"  {ei}. {ex}")
    notes_lines.append("")

notes_lines.append("---")
notes_lines.append("")
notes_lines.append("## 5. Stability flags")
notes_lines.append("")
notes_lines.append(
    "Clusters with cross-model ARI < 0.40 at the same mcs setting should be flagged as unstable "
    "and treated as configuration artifacts rather than stable senses. "
    "See `stability_crosstable_professional.csv` for the full pairwise table."
)
notes_lines.append("")
notes_lines.append(
    "When a cluster appears under one embedding model but not the other at the same mcs, "
    "it is an unstable cluster."
)
notes_lines.append("")
notes_lines.append("---")
notes_lines.append("")
notes_lines.append("## 6. Key disambiguating signal")
notes_lines.append("")
notes_lines.append(
    "The key disambiguating question per the Phase 2 KWIC agent's recommendation is: "
    "**what fraction of contexts contain `professional help` within ±5 tokens of the seed?** "
    "A cluster with high `professional help` co-occurrence (>10%) is a candidate help-directive cluster. "
    "A cluster with near-zero `professional help` co-occurrence is not. "
    "Sense labeling is a researcher decision at the Pattern A checkpoint; these fractions are evidence, not labels."
)
notes_lines.append("")
notes_lines.append("---")
notes_lines.append("")
notes_lines.append("## 7. Files produced")
notes_lines.append("")
notes_lines.append("| File | Description |")
notes_lines.append("|---|---|")
notes_lines.append("| `kwic_full_professional.csv` | All KWIC occurrences, ±20 token context |")
notes_lines.append("| `embeddings_professional_minilm.npy` | MiniLM-L6-v2 embeddings |")
notes_lines.append("| `embeddings_professional_mpnet.npy` | MPNet-base-v2 embeddings |")
notes_lines.append("| `clusters_professional_{model}_mcs{n}.csv` | Cluster assignments per config |")
notes_lines.append("| `stability_crosstable_professional.csv` | Pairwise ARI + NMI table |")
notes_lines.append("| `exemplars_professional.csv` | 10 centroid-nearest exemplars per cluster (canonical) |")
notes_lines.append("| `syntactic_features_per_cluster_professional.csv` | POS, code-block, imperative, subreddit per cluster |")

notes_text = "\n".join(notes_lines)
notes_path = OUT_DIR / "sense_discovery_notes_professional.md"
notes_path.write_text(notes_text, encoding="utf-8")
print(f"  Saved -> {notes_path.name}")

# ---------------------------------------------------------------------------
# Summary JSON for audit trail writer
# ---------------------------------------------------------------------------
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

summary_data = {
    "seed":           SEED,
    "total_occ":      int(total_occ),
    "total_posts":    int(total_posts),
    "ph_pm5_count":   int(kwic_df["has_professional_help_pm5"].sum()),
    "ph_pm5_frac":    round(float(kwic_df["has_professional_help_pm5"].mean()), 4),
    "config_stats":   [[c, int(n), float(nf)] for c, n, nf in config_stats],
    "mean_cross_ari": round(float(mean_cross_ari), 4) if not np.isnan(mean_cross_ari) else None,
    "mean_within_ari": round(float(mean_within_ari), 4) if not np.isnan(mean_within_ari) else None,
    "canonical_clusters": [int(c) for c in unique_clusters],
    "synfeat": {str(cl): sf_lookup[cl] for cl in sorted(sf_lookup)},
}
(OUT_DIR / "sense_discovery_summary_data.json").write_text(
    json.dumps(summary_data, indent=2, cls=NpEncoder), encoding="utf-8"
)
print("Done. Summary data saved.")
print(f"\nAll outputs in: {OUT_DIR}")

