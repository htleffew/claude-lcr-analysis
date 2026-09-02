"""
Re-runs only the notes-writer and verdict-collector from the Phase 2.5 sense-discovery
results that were already saved to disk. All heavy compute (embedding, clustering) is
already done; this reads the saved CSVs and writes the notes + verdicts JSON.
"""

import csv, json, os, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT_DIR = r"C:/Users/drhea/estate/projects/research/claude-lcr\deliverables\phase_2_5_pass1b_sense_discovery"
SEEDS   = ["professional", "episode", "concerned", "worried", "mental"]
CANONICAL_MODEL = "minilm"
CANONICAL_MCS   = 10
MCS_VALUES = [5, 10, 20]
MODELS = {"minilm": "all-MiniLM-L6-v2", "mpnet": "all-mpnet-base-v2"}

def verdict_from(canon_ari, n_clust):
    if n_clust <= 1:
        v = "FALSIFIED"
        r = "Only one or zero clusters found; no polysemy separation occurred."
    elif canon_ari >= 0.40:
        v = "WORKS"
        r = (f"Cross-model ARI at canonical mcs=10 is {canon_ari:.3f} >= 0.40 threshold. "
             f"{n_clust} clusters found. Cluster structure is replicable across embedding models.")
    elif canon_ari < 0.20:
        v = "FALSIFIED"
        r = (f"Cross-model ARI at canonical mcs=10 is {canon_ari:.3f} < 0.20. "
             "Cluster structure does not survive embedding-model change.")
    else:
        v = "INCONCLUSIVE"
        r = (f"Cross-model ARI at canonical mcs=10 is {canon_ari:.3f} (between 0.20 and 0.40). "
             "Moderate stability; some cluster correspondence but not robust across models.")
    return v, r

def load_seed_results(seed):
    seed_dir = os.path.join(OUT_DIR, seed)

    kwic_df  = pd.read_csv(os.path.join(seed_dir, f"kwic_full_{seed}.csv"), encoding='utf-8-sig')
    feat_df  = pd.read_csv(os.path.join(seed_dir, f"syntactic_features_per_cluster_{seed}.csv"), encoding='utf-8-sig')
    stab_df  = pd.read_csv(os.path.join(seed_dir, f"stability_crosstable_{seed}.csv"), encoding='utf-8-sig')
    ex_path  = os.path.join(seed_dir, f"exemplars_{seed}.csv")
    try:
        ex_df = pd.read_csv(ex_path, encoding='utf-8-sig') if os.path.exists(ex_path) else pd.DataFrame()
    except Exception:
        ex_df = pd.DataFrame()

    # Load cluster label files to reconstruct label_dict
    label_dict = {}
    for model_key in MODELS:
        for mcs in MCS_VALUES:
            cfg = f"{model_key}_mcs{mcs}"
            cpath = os.path.join(seed_dir, f"clusters_{seed}_{model_key}_mcs{mcs}.csv")
            if os.path.exists(cpath):
                c_df = pd.read_csv(cpath, encoding='utf-8-sig')
                label_dict[cfg] = c_df['cluster_id'].values
            else:
                label_dict[cfg] = np.array([])

    # Stability stats
    cross_model_pairs = stab_df[
        stab_df.apply(lambda r: r['config_a'].split('_mcs')[0] != r['config_b'].split('_mcs')[0], axis=1)
    ]
    within_model_pairs = stab_df[
        stab_df.apply(lambda r: r['config_a'].split('_mcs')[0] == r['config_b'].split('_mcs')[0], axis=1)
    ]
    mean_cross_ari  = cross_model_pairs['ari'].mean() if len(cross_model_pairs) else float('nan')
    mean_within_ari = within_model_pairs['ari'].mean() if len(within_model_pairs) else float('nan')
    canon_row = stab_df[
        stab_df['config_a'].str.contains('mcs10') & stab_df['config_b'].str.contains('mcs10')
    ]
    canon_ari = float(canon_row['ari'].values[0]) if len(canon_row) else float('nan')

    canon_labels = label_dict.get(f"{CANONICAL_MODEL}_mcs{CANONICAL_MCS}", np.array([]))
    n_clusters_canon = len(set(canon_labels) - {-1}) if len(canon_labels) else 0
    noise_frac_canon = float((canon_labels == -1).sum() / len(canon_labels)) if len(canon_labels) else 1.0

    return {
        'seed': seed,
        'n_kwic': len(kwic_df),
        'n_clusters_canon': n_clusters_canon,
        'noise_frac_canon': noise_frac_canon,
        'mean_cross_ari': mean_cross_ari,
        'mean_within_ari': mean_within_ari,
        'canon_mcs10_cross_ari': canon_ari,
        'feat_df': feat_df,
        'kwic_df': kwic_df,
        'exemplar_df': ex_df,
        'label_dict': label_dict,
        'stab_df': stab_df,
        'seed_dir': seed_dir,
    }

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
    label_dict = result['label_dict']

    verdict, verdict_rationale = verdict_from(canon_ari, n_clust)

    # Cluster table
    cluster_table_lines = ["| Cluster | n | Code-block % | Imperative % | Type-post % | r2_match % | Dominant subreddit |"]
    cluster_table_lines += ["|---|---|---|---|---|---|---|"]
    for _, row in feat_df.iterrows():
        cid = int(row['cluster_id'])
        n   = int(row['n'])
        cb  = f"{row.get('code_block_frac',0)*100:.1f}%"
        imp = f"{row.get('imperative_frac',0)*100:.1f}%"
        tp  = f"{row.get('frac_post',0)*100:.1f}%"
        r2  = f"{row.get('r2_match_frac',0)*100:.1f}%"
        ds  = row.get('dominant_subreddit', '')
        cluster_table_lines.append(f"| {'Noise' if cid==-1 else f'C{cid}'} | {n} | {cb} | {imp} | {tp} | {r2} | {ds} |")
    cluster_table = '\n'.join(cluster_table_lines)

    # professional help table
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

    # Exemplars
    exemplar_sections = []
    if len(ex_df) > 0 and 'cluster_id' in ex_df.columns:
        for c in sorted(set(ex_df['cluster_id'])):
            sub = ex_df[ex_df['cluster_id']==c].head(5)
            exemplar_sections.append(f"\n### C{int(c)} exemplars (5 nearest centroid)\n")
            for i, (_, row) in enumerate(sub.iterrows(), 1):
                ctx = str(row.get('full_context',''))[:300]
                exemplar_sections.append(f"{i}. ...{ctx}...")
    else:
        exemplar_sections.append("No stable clusters under canonical configuration; no exemplars produced.")
    exemplar_text = '\n'.join(exemplar_sections)

    # Config table
    config_lines = ["| Configuration | Clusters | Noise fraction |", "|---|---|---|"]
    for cfg, labels in label_dict.items():
        if len(labels) == 0:
            continue
        nc  = len(set(labels)-{-1})
        nf2 = float((labels==-1).sum()/len(labels))
        config_lines.append(f"| {cfg} | {nc} | {nf2:.3f} |")
    config_table = '\n'.join(config_lines)

    noise_note = (
        'High noise (>0.60) indicates genuine semantic heterogeneity.'
        if nf > 0.60 else
        'Moderate noise fraction; clusters capture a meaningful proportion of contexts.'
        if nf > 0.30 else
        'Low noise fraction; most contexts cluster tightly.'
    )

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

Cluster counts across all 6 configurations (model x mcs):

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

**Comparison to sleep §1.8 falsification:** The sleep-nudge seeds (*sleep, rest, break, tired*) produced cross-model ARI in the 0.2 range across all three corpus scales; no seed ever reached 0.40. The current seed `{seed}` with canon ARI={canon_ari:.3f} is classified as **{verdict}**.

---

## 6. Anomalies and notes

- No sense labels assigned here. Researcher decides at Pattern A checkpoint.
- Do not apply the retired four-domain framework.
- Noise fraction: {nf:.3f}. {noise_note}
"""

    notes_path = os.path.join(seed_dir, f"sense_discovery_notes_{seed}.md")
    with open(notes_path, 'w', encoding='utf-8') as f:
        f.write(notes)
    print(f"  Notes written: {notes_path}")
    return verdict, verdict_rationale, canon_ari, n_clust, nf

def main():
    verdicts = {}
    for seed in SEEDS:
        print(f"Processing notes for: {seed}")
        try:
            result = load_seed_results(seed)
            verdict, rationale, canon_ari, n_clust, nf = write_notes(result)
            verdicts[seed] = {
                'verdict': verdict,
                'rationale': rationale,
                'canon_ari': round(float(canon_ari), 4),
                'n_clusters_canon': int(n_clust),
                'noise_frac_canon': round(float(nf), 4),
                'n_kwic': int(result['n_kwic']),
            }
            print(f"  {seed}: {verdict} (k={n_clust}, ARI={canon_ari:.3f}, noise={nf:.3f}, n={result['n_kwic']})")
        except Exception as e:
            import traceback
            print(f"  ERROR on {seed}: {e}")
            traceback.print_exc()

    with open(os.path.join(OUT_DIR, 'verdicts.json'), 'w') as f:
        json.dump(verdicts, f, indent=2)
    print("\nVerdicts saved.")
    print(json.dumps(verdicts, indent=2))

if __name__ == "__main__":
    main()
