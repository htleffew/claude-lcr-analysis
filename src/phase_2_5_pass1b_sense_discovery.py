"""
Phase 2.5 Pass 1b Sense-Discovery Script
Implements §1.8 (methods_library.md) for all 5 LCR polysemous seeds:
  professional, episode, concerned, worried, mental

Corpus: lcr_pass1b_canonical.csv (26,158 rows: 1,173 posts + 24,985 comments)
Output: deliverables/phase_2_5_pass1b_sense_discovery/

Per-seed procedure:
  1. Extract KWIC ±20 tokens -> kwic_full_{seed}.csv
  2. Embed with all-MiniLM-L6-v2 -> embeddings
  3. HDBSCAN at mcs=5,10,20 -> cluster assignments
  4. Re-embed with all-mpnet-base-v2 -> cluster again
  5. Stability cross-table (ARI/NMI)
  6. 10 nearest-centroid exemplars per cluster (canonical: MiniLM+mcs=10)
  7. Syntactic features per cluster (POS, code-block, imperative, subreddit,
     type, retrieval_provenance, r2_any_match);
     for 'professional': professional help bigram within ±5 tokens
  8. sense_discovery_notes_{seed}.md (no labels)
"""

import csv
import re
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CORPUS_PATH = r"C:/Users/drhea/estate/projects/research/claude-lcr\data\lcr_pass1b_canonical.csv"
OUT_DIR     = r"C:/Users/drhea/estate/projects/research/claude-lcr\deliverables\phase_2_5_pass1b_sense_discovery"
MODELS      = {
    "minilm": "all-MiniLM-L6-v2",
    "mpnet":  "all-mpnet-base-v2",
}
MCS_VALUES  = [5, 10, 20]
CANONICAL_MODEL = "minilm"
CANONICAL_MCS   = 10
SEEDS = ["professional", "episode", "concerned", "worried", "mental"]

# ---------------------------------------------------------------------------
# Data-quality fixes (proven on sleep canonical)
# ---------------------------------------------------------------------------
URL_RE   = re.compile(r'https?://\S+|www\.\S+')
WS_RE    = re.compile(r'\s+')
MIN_TOK  = 3   # minimum token length to include

def clean_text(text):
    """Strip URLs, collapse whitespace."""
    text = URL_RE.sub(' ', text)
    text = WS_RE.sub(' ', text)
    return text.strip()

def tokenize(text):
    """Simple whitespace tokenize; keep tokens with len >= MIN_TOK."""
    return [t for t in text.split() if len(t) >= MIN_TOK]

# ---------------------------------------------------------------------------
# KWIC extraction
# ---------------------------------------------------------------------------
WINDOW = 20   # ±20 tokens

def extract_kwic(rows, seed):
    """
    Extract KWIC windows for seed (word-boundary, case-insensitive).
    Returns list of dicts with: row_id, post_id, type, retrieval_provenance,
    r2_any_match, subreddit, full_context, token_pos
    """
    pattern = re.compile(r'\b' + re.escape(seed) + r'\b', re.IGNORECASE)
    kwic_rows = []
    for i, row in enumerate(rows):
        body = clean_text(row.get('body', ''))
        tokens = body.split()  # raw whitespace split for positional indexing
        # find all match positions in the joined token string
        for m in pattern.finditer(body):
            # Map character offset to token index
            char_start = m.start()
            pos = len(body[:char_start].split()) - 1  # approx token index
            # build window
            start = max(0, pos - WINDOW)
            end   = min(len(tokens), pos + WINDOW + 1)
            context = ' '.join(tokens[start:end])
            kwic_rows.append({
                'row_id':               i,
                'post_id':              row.get('post_id', row.get('﻿post_id', '')),
                'type':                 row.get('type', ''),
                'retrieval_provenance': row.get('retrieval_provenance', ''),
                'r2_any_match':         row.get('r2_any_match', ''),
                'subreddit':            row.get('subreddit', ''),
                'full_context':         context,
                'token_pos':            pos,
                'body_len':             len(tokens),
            })
    return kwic_rows

# ---------------------------------------------------------------------------
# Syntactic feature helpers
# ---------------------------------------------------------------------------
CODE_BLOCK_RE = re.compile(r'```|^\s{4}', re.MULTILINE)
IMPERATIVE_RE = re.compile(
    r'\b(you should|you need to|please|consider|try|go see|seek|speak with|talk to|reach out|consult)\b',
    re.IGNORECASE
)

def has_code_block(text):
    return bool(CODE_BLOCK_RE.search(text))

def has_imperative(text):
    return bool(IMPERATIVE_RE.search(text))

def has_professional_help_pm5(context, seed_token_pos, window=5):
    """Check if 'professional help' bigram appears within ±5 tokens of seed position."""
    tokens = context.split()
    start = max(0, seed_token_pos - window) if seed_token_pos is not None else 0
    end   = min(len(tokens), (seed_token_pos or 0) + window + 1)
    snippet = ' '.join(tokens[start:end])
    return bool(re.search(r'professional\s+help', snippet, re.IGNORECASE))

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def cluster_embeddings(embeddings, mcs):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(embeddings)
    return labels

# ---------------------------------------------------------------------------
# Stability cross-table
# ---------------------------------------------------------------------------
def build_stability_table(label_dict):
    """
    label_dict: {config_name: np.array of labels}
    Returns DataFrame of pairwise ARI and NMI.
    """
    configs = list(label_dict.keys())
    records = []
    for i in range(len(configs)):
        for j in range(i+1, len(configs)):
            a, b = configs[i], configs[j]
            la, lb = label_dict[a], label_dict[b]
            ari = adjusted_rand_score(la, lb)
            nmi = normalized_mutual_info_score(la, lb, average_method='arithmetic')
            records.append({'config_a': a, 'config_b': b, 'ari': round(ari,4), 'nmi': round(nmi,4)})
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Exemplars: nearest to cluster centroid
# ---------------------------------------------------------------------------
def get_exemplars(embeddings, labels, n=10):
    """
    For each unique cluster (excluding noise=-1), return indices of n nearest-centroid exemplars.
    """
    unique_clusters = sorted(set(labels) - {-1})
    exemplar_indices = {}
    for c in unique_clusters:
        mask = (labels == c)
        idxs = np.where(mask)[0]
        centroid = embeddings[idxs].mean(axis=0)
        dists = np.linalg.norm(embeddings[idxs] - centroid, axis=1)
        nearest = idxs[np.argsort(dists)[:n]]
        exemplar_indices[c] = nearest.tolist()
    return exemplar_indices

# ---------------------------------------------------------------------------
# Per-seed analysis
# ---------------------------------------------------------------------------
def analyze_seed(seed, rows, models_loaded):
    print(f"\n{'='*60}")
    print(f"SEED: {seed}")
    print(f"{'='*60}")

    seed_dir = os.path.join(OUT_DIR, seed)
    os.makedirs(seed_dir, exist_ok=True)

    # Step 1: KWIC extraction
    kwic_rows = extract_kwic(rows, seed)
    if len(kwic_rows) == 0:
        print(f"  No occurrences found for '{seed}'.")
        return None

    print(f"  KWIC occurrences: {len(kwic_rows)}")

    # Save KWIC
    kwic_df = pd.DataFrame(kwic_rows)
    kwic_path = os.path.join(seed_dir, f"kwic_full_{seed}.csv")
    kwic_df.to_csv(kwic_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {kwic_path}")

    contexts = kwic_df['full_context'].tolist()

    # Steps 2-4: Embed + cluster for both models
    label_dict = {}
    embeddings_dict = {}

    for model_key, model in models_loaded.items():
        print(f"  Embedding with {model_key}...")
        embs = model.encode(contexts, batch_size=64, show_progress_bar=False)
        embeddings_dict[model_key] = embs

        # Save embeddings
        emb_path = os.path.join(seed_dir, f"embeddings_{seed}_{model_key}.npy")
        np.save(emb_path, embs)

        # Save index
        idx_df = kwic_df[['row_id','post_id','type','retrieval_provenance','r2_any_match','subreddit']].copy()
        idx_df['emb_row'] = range(len(idx_df))
        idx_df.to_csv(os.path.join(seed_dir, f"embeddings_{seed}_{model_key}_index.csv"), index=False, encoding='utf-8-sig')

        for mcs in MCS_VALUES:
            config = f"{model_key}_mcs{mcs}"
            labels = cluster_embeddings(embs, mcs)
            label_dict[config] = labels

            cluster_df = kwic_df.copy()
            cluster_df['cluster_id'] = labels
            cluster_df.to_csv(
                os.path.join(seed_dir, f"clusters_{seed}_{model_key}_mcs{mcs}.csv"),
                index=False, encoding='utf-8-sig'
            )

            n_clusters = len(set(labels) - {-1})
            noise_frac = (labels == -1).sum() / len(labels)
            print(f"    {config}: {n_clusters} clusters, noise={noise_frac:.3f}")

    # Step 5: Stability cross-table
    stab_df = build_stability_table(label_dict)
    stab_path = os.path.join(seed_dir, f"stability_crosstable_{seed}.csv")
    stab_df.to_csv(stab_path, index=False, encoding='utf-8-sig')
    print(f"  Stability cross-table saved.")

    # Compute summary stability stats
    cross_model_pairs = stab_df[
        stab_df.apply(lambda r: r['config_a'].split('_mcs')[0] != r['config_b'].split('_mcs')[0], axis=1)
    ]
    within_model_pairs = stab_df[
        stab_df.apply(lambda r: r['config_a'].split('_mcs')[0] == r['config_b'].split('_mcs')[0], axis=1)
    ]
    mean_cross_ari  = cross_model_pairs['ari'].mean() if len(cross_model_pairs) else float('nan')
    mean_within_ari = within_model_pairs['ari'].mean() if len(within_model_pairs) else float('nan')
    # Canonical matched-mcs cross-model ARI (mcs=10 only)
    canon_row = stab_df[
        stab_df['config_a'].str.contains('mcs10') & stab_df['config_b'].str.contains('mcs10')
    ]
    canon_ari = canon_row['ari'].values[0] if len(canon_row) else float('nan')

    print(f"  Mean cross-model ARI: {mean_cross_ari:.3f}")
    print(f"  Mean within-model ARI: {mean_within_ari:.3f}")
    print(f"  Canonical (mcs=10) cross-model ARI: {canon_ari:.3f}")

    # Step 6: Exemplars under canonical config
    canon_config = f"{CANONICAL_MODEL}_mcs{CANONICAL_MCS}"
    canon_labels = label_dict[canon_config]
    canon_embs   = embeddings_dict[CANONICAL_MODEL]
    exemplar_idxs = get_exemplars(canon_embs, canon_labels, n=10)

    exemplar_records = []
    for c, idxs in exemplar_idxs.items():
        for rank, idx in enumerate(idxs):
            rec = kwic_df.iloc[idx].to_dict()
            rec['cluster_id'] = c
            rec['exemplar_rank'] = rank
            exemplar_records.append(rec)
    exemplar_df = pd.DataFrame(exemplar_records)
    exemplar_df.to_csv(
        os.path.join(seed_dir, f"exemplars_{seed}.csv"),
        index=False, encoding='utf-8-sig'
    )
    print(f"  Exemplars saved.")

    # Step 7: Syntactic features per cluster
    cluster_labels_series = pd.Series(canon_labels, name='cluster_id_canon')
    kwic_df_aug = kwic_df.copy()
    kwic_df_aug['cluster_id_canon'] = canon_labels
    kwic_df_aug['has_code_block']   = kwic_df_aug['full_context'].apply(has_code_block)
    kwic_df_aug['has_imperative']   = kwic_df_aug['full_context'].apply(has_imperative)
    if seed == 'professional':
        kwic_df_aug['has_prof_help_pm5'] = kwic_df_aug.apply(
            lambda r: has_professional_help_pm5(r['full_context'], r.get('token_pos'), window=5),
            axis=1
        )

    # Cluster-level feature table
    feat_records = []
    all_cluster_ids = sorted(set(canon_labels))
    for c in all_cluster_ids:
        mask = kwic_df_aug['cluster_id_canon'] == c
        subset = kwic_df_aug[mask]
        n = len(subset)
        rec = {
            'cluster_id': c,
            'n': n,
            'code_block_frac':  subset['has_code_block'].mean(),
            'imperative_frac':  subset['has_imperative'].mean(),
            'frac_post':        (subset['type']=='post').mean(),
            'frac_comment':     (subset['type']=='comment').mean(),
            'r2_match_frac':    (subset['r2_any_match']=='True').mean(),
            'dominant_subreddit': subset['subreddit'].value_counts().idxmax() if n > 0 else '',
            'subreddit_dist':   subset['subreddit'].value_counts().to_dict(),
            'top_prov':         subset['retrieval_provenance'].value_counts().idxmax() if n > 0 else '',
        }
        if seed == 'professional':
            rec['prof_help_pm5_frac'] = subset['has_prof_help_pm5'].mean()
        feat_records.append(rec)

    feat_df = pd.DataFrame(feat_records)
    feat_df.to_csv(
        os.path.join(seed_dir, f"syntactic_features_per_cluster_{seed}.csv"),
        index=False, encoding='utf-8-sig'
    )
    print(f"  Syntactic features saved.")

    # Cluster summary for notes
    n_clusters_canon = len(set(canon_labels) - {-1})
    noise_frac_canon = (canon_labels == -1).sum() / len(canon_labels)

    return {
        'seed': seed,
        'n_kwic': len(kwic_rows),
        'n_clusters_canon': n_clusters_canon,
        'noise_frac_canon': noise_frac_canon,
        'mean_cross_ari': mean_cross_ari,
        'mean_within_ari': mean_within_ari,
        'canon_mcs10_cross_ari': canon_ari,
        'feat_df': feat_df,
        'kwic_df': kwic_df_aug,
        'exemplar_df': exemplar_df,
        'label_dict': label_dict,
        'stab_df': stab_df,
        'seed_dir': seed_dir,
    }

# ---------------------------------------------------------------------------
# Per-seed notes writer
# ---------------------------------------------------------------------------
def write_notes(result):
    seed     = result['seed']
    feat_df  = result['feat_df']
    kwic_df  = result['kwic_df']
    ex_df    = result['exemplar_df']
    seed_dir = result['seed_dir']

    n_kwic   = result['n_kwic']
    n_clust  = result['n_clusters_canon']
    nf       = result['noise_frac_canon']
    cross_ari= result['mean_cross_ari']
    within_ari=result['mean_within_ari']
    canon_ari= result['canon_mcs10_cross_ari']

    # §1.8 verdict
    # Criteria to WORK: canon_ari >= 0.40, n_clusters >= 2
    # FALSIFIED: canon_ari < 0.20 or n_clusters <= 1
    # INCONCLUSIVE: otherwise
    if n_clust <= 1:
        verdict = "FALSIFIED"
        verdict_rationale = (
            "Only one or zero clusters found; no polysemy separation occurred."
        )
    elif canon_ari >= 0.40:
        verdict = "WORKS"
        verdict_rationale = (
            f"Cross-model ARI at canonical mcs=10 is {canon_ari:.3f} >= 0.40 threshold. "
            f"{n_clust} clusters found. Cluster structure is replicable across embedding models."
        )
    elif canon_ari < 0.20:
        verdict = "FALSIFIED"
        verdict_rationale = (
            f"Cross-model ARI at canonical mcs=10 is {canon_ari:.3f} < 0.20. "
            "Cluster structure does not survive embedding-model change. "
            "Polysemy structure is not separable by this technique at this corpus scale."
        )
    else:
        verdict = "INCONCLUSIVE"
        verdict_rationale = (
            f"Cross-model ARI at canonical mcs=10 is {canon_ari:.3f} (between 0.20 and 0.40). "
            "Moderate stability; some cluster correspondence but not robust across models."
        )

    # Build cluster table string
    cluster_table_lines = ["| Cluster | n | Code-block % | Imperative % | Type-post % | r2_match % | Dominant subreddit |"]
    cluster_table_lines += ["|---|---|---|---|---|---|---|"]
    for _, row in feat_df.iterrows():
        cid = int(row['cluster_id'])
        n   = int(row['n'])
        cb  = f"{row['code_block_frac']*100:.1f}%"
        imp = f"{row['imperative_frac']*100:.1f}%"
        tp  = f"{row['frac_post']*100:.1f}%"
        r2  = f"{row['r2_match_frac']*100:.1f}%"
        ds  = row.get('dominant_subreddit', '')
        cluster_table_lines.append(f"| {'Noise' if cid==-1 else f'C{cid}'} | {n} | {cb} | {imp} | {tp} | {r2} | {ds} |")
    cluster_table = '\n'.join(cluster_table_lines)

    # Professional-specific table
    prof_table = ""
    if seed == 'professional' and 'prof_help_pm5_frac' in feat_df.columns:
        prof_lines = ["| Cluster | n | `professional help` ±5 % |"]
        prof_lines += ["|---|---|---|"]
        for _, row in feat_df.iterrows():
            cid = int(row['cluster_id'])
            n   = int(row['n'])
            ph  = f"{row['prof_help_pm5_frac']*100:.1f}%"
            prof_lines.append(f"| {'Noise' if cid==-1 else f'C{cid}'} | {n} | {ph} |")
        prof_table = "\n\n## `professional help` bigram co-occurrence (within ±5 tokens)\n\n" + '\n'.join(prof_lines)

    # Exemplar section per cluster
    exemplar_sections = []
    if len(ex_df) > 0 and 'cluster_id' in ex_df.columns:
        for c in sorted(set(ex_df['cluster_id'])):
            sub = ex_df[ex_df['cluster_id']==c].head(5)
            exemplar_sections.append(f"\n### C{int(c)} exemplars (5 nearest centroid)\n")
            for i, (_, row) in enumerate(sub.iterrows(), 1):
                ctx = row.get('full_context','')[:300]
                exemplar_sections.append(f"{i}. ...{ctx}...")
    else:
        exemplar_sections.append("No stable clusters under canonical configuration; no exemplars produced.")
    exemplar_text = '\n'.join(exemplar_sections)

    # Config overview table
    label_dict = result['label_dict']
    config_lines = ["| Configuration | Clusters | Noise fraction |", "|---|---|---|"]
    for cfg, labels in label_dict.items():
        nc = len(set(labels)-{-1})
        nf2 = (labels==-1).sum()/len(labels)
        config_lines.append(f"| {cfg} | {nc} | {nf2:.3f} |")
    config_table = '\n'.join(config_lines)

    notes = f"""# Sense-Discovery Notes: `{seed}`

**Date:** 2026-05-17
**Method:** §1.8 sense-discovery via embedding-cluster on KWIC contexts
**Corpus:** `lcr_pass1b_canonical.csv` — 26,158 rows (1,173 posts + 24,985 comments)
**Deliverables directory:** `deliverables/phase_2_5_pass1b_sense_discovery/{seed}/`

---

## 1. Hit count

{n_kwic} KWIC occurrences of `{seed}` (case-insensitive, word-boundary) across the Pass 1b corpus.

---

## 2. Cluster structure overview

Under the canonical configuration (MiniLM-L6-v2 + mcs=10), HDBSCAN found **{n_clust} clusters** plus a noise partition:

{cluster_table}
{prof_table}

Cluster counts across all 6 configurations (model × mcs):

{config_table}

---

## 3. Cross-model stability

| Comparison | Mean ARI |
|---|---|
| Cross-model, same mcs (minilm vs. mpnet) | {cross_ari:.3f} |
| Within-model, varying mcs | {within_ari:.3f} |
| **Canonical (mcs=10) cross-model ARI** | **{canon_ari:.3f}** |

---

## 4. Exemplar contexts (no sense labels — researcher decides)
{exemplar_text}

---

## 5. §1.8 Verdict

**{verdict}**

{verdict_rationale}

**Comparison to sleep §1.8 falsification:** The sleep-nudge seeds (*sleep, rest, break, tired*) produced cross-model ARI in the 0.2 range across all three corpus scales; no seed ever reached 0.40. The current seed `{seed}` with canon_ari={canon_ari:.3f} is classified as {verdict}.

---

## 6. Anomalies and notes

- No sense labels are assigned here. Researcher decides at Pattern A checkpoint.
- Do not apply the retired four-domain framework.
- Noise fraction: {nf:.3f}. {'High noise (>0.60) indicates genuine semantic heterogeneity; the seed term modifies many different noun phrases or appears in diverse contexts.' if nf > 0.60 else 'Moderate noise fraction; clusters capture a meaningful proportion of contexts.' if nf > 0.30 else 'Low noise fraction; most contexts cluster tightly.'}
"""

    notes_path = os.path.join(seed_dir, f"sense_discovery_notes_{seed}.md")
    with open(notes_path, 'w', encoding='utf-8') as f:
        f.write(notes)
    print(f"  Notes saved: {notes_path}")
    return verdict, verdict_rationale, canon_ari, n_clust, nf

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading corpus...")
    with open(CORPUS_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows)} rows.")

    print("\nLoading embedding models...")
    models_loaded = {}
    for key, model_name in MODELS.items():
        print(f"  Loading {model_name}...")
        models_loaded[key] = SentenceTransformer(model_name)
    print("Models loaded.")

    results = {}
    for seed in SEEDS:
        try:
            result = analyze_seed(seed, rows, models_loaded)
            if result is not None:
                results[seed] = result
        except Exception as e:
            print(f"  ERROR on seed '{seed}': {e}")
            import traceback; traceback.print_exc()

    # Write per-seed notes and collect verdicts
    verdicts = {}
    for seed, result in results.items():
        verdict, rationale, canon_ari, n_clust, nf = write_notes(result)
        verdicts[seed] = {
            'verdict': verdict,
            'rationale': rationale,
            'canon_ari': canon_ari,
            'n_clusters_canon': n_clust,
            'noise_frac_canon': nf,
            'n_kwic': result['n_kwic'],
        }
        print(f"  {seed}: §1.8 verdict = {verdict} (ARI={canon_ari:.3f}, k={n_clust}, noise={nf:.3f})")

    # Save verdicts JSON
    with open(os.path.join(OUT_DIR, 'verdicts.json'), 'w') as f:
        json.dump(verdicts, f, indent=2)

    print("\nAll seeds processed. Verdicts summary:")
    for seed, v in verdicts.items():
        print(f"  {seed}: {v['verdict']} (k={v['n_clusters_canon']}, ARI={v['canon_ari']:.3f}, noise={v['noise_frac_canon']:.3f}, n={v['n_kwic']})")

    print("\nDone. All outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
