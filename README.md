# Claude LCR Analysis

A clinical-phenomenological case analysis of the **Long Conversation Reminder (LCR) pathologizing phenomenon** observed in Anthropic's Claude Sonnet 4.5 following its September 29, 2025 release.

**Heather Leffew, PhD** &middot; [HAIIQU](https://haiiqu.com/)

Sibling project: [claude-sleep-analysis](https://github.com/htleffew/claude-sleep-analysis), which applies the same procedural methodology to the unsolicited sleep-nudging behavior reported in users of Claude Opus 4.7. The cross-generational role-violation question raised by the two corpora is flagged as future work in the present paper rather than litigated here.

---

## The paper

The preprint *Observing the Guardrail Paradox via Anthropic's Long Conversation Reminder* is the primary scholarly output of this repository.

| Artifact | Location |
|---|---|
| Compiled PDF | [`paper/leffew_2026_lcr_preprint_rewrite_v1.pdf`](paper/leffew_2026_lcr_preprint_rewrite_v1.pdf) (45 pp) |
| LaTeX source | [`paper/leffew_2026_lcr_preprint_rewrite_v1.tex`](paper/leffew_2026_lcr_preprint_rewrite_v1.tex) (`article` class; self-contained, no `.cls`/`.bbl`) |
| Bibliography source | [`paper/references.bib`](paper/references.bib) (the References section is emitted into the `.tex`) |
| Figures (600 DPI PNG) | [`paper/figures/`](paper/figures/) |
| Tables (LaTeX + Markdown) | [`paper/tables/`](paper/tables/) |
| arXiv submission bundle | `submissions/leffew_2026_lcr_preprint_rewrite_v1-arxiv.tar.gz` (build + verify per `academic_publishing/ARXIV_SUBMISSION.md`) |

The paper characterizes 35 hand-coded confirmed cases of LCR pathologizing against a five-property structural signature. The empirical centerpiece: 100% unsolicited issuance, 100% asymmetric restriction direction, and 0% yielded to user pushback (all 5 documented-pushback cases escalated).

---

## What this project investigates

In late September 2025, Claude Sonnet 4.5 began issuing unsolicited psychiatric-style attributions during extended conversations: characterizing users as manic, dissociative, or in crisis on the basis of creative writing, technical enthusiasm, or late-hour activity, and directing them toward professional mental health support. The community traced the behavior to a hidden system-prompt scaffolding called the *Long Conversation Reminder* (LCR) and gave the phenomenon a name: *The Flip*.

The October 2025 [Medium article](prior_artifacts/leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf) by the author framed the phenomenon as a clinical role-violation: an algorithmic system performing diagnostic functions without the training, role-warrant, or assessment instruments that license such functions in human professional practice. This repository scales that phenomenological characterization onto a Reddit corpus spanning August 1 to December 31, 2025.

The project is conducted within the tradition of **reflexive single-analyst qualitative inquiry**. The analyst is the instrument. Constructs emerge from the corpus through descriptive engagement. Quantification supports clinical observation rather than replacing it. The rigor criteria appropriate to this design (reflexive audit trail, sensitivity ablations, external lexical cross-walks, hand-coded precision-at-N) are specified in the procedural method document linked below.

---

## Methodological lineage

The procedural methodology that the analysis follows is documented in the universal methodology repository ([`llm-behavior-reddit-analysis-universal`](https://github.com/htleffew/llm-behavior-reddit-analysis-universal)):

1. **`community_reported_llm_behavior_method.md`**: the procedural spine. Phase-by-phase procedure with checkpoints and decision rules. Topic-agnostic; applies to any community-reported LLM behavior.
2. **`methods_library.md`**: the techniques toolkit. Organized by structural feature of the phenomenon (temporal, lexical, semantic, network, affective, behavioral-sequence).

---

## Repository structure

```
claude-lcr-analysis/
├── README.md                              this file
├── CITATION.cff
├── LICENSE-code                           MIT (source code)
├── LICENSE-data                           CC BY 4.0 (data, prior artifacts, paper)
├── requirements.txt
│
├── paper/                                 the preprint and its supporting artifacts
│   ├── leffew_2026_lcr_preprint_rewrite_v1.tex   active manuscript (article class, 45 pp)
│   ├── leffew_2026_lcr_preprint_rewrite_v1.pdf   compiled paper
│   ├── leffew_2026_lcr_preprint.{tex,pdf}        superseded prior version (retained for history)
│   ├── references.bib                     bibliography source (References section emitted into the .tex)
│   ├── figures/                           600 DPI PNGs + caption.md notes
│   │   ├── corpus_composition.png
│   │   ├── structural_signature.png
│   │   ├── case_signature_heatmap.png
│   │   ├── sense_discovery_umap.png
│   │   ├── subreddit_mechanism_vocabulary.png
│   │   └── temporal_acronym_emergence.png
│   └── tables/                            LaTeX + Markdown fragments
│       ├── table_1.{tex,md}               corpus composition
│       ├── table_2.{tex,md}               five-property structural signature (n=35)
│       ├── table_3.{tex,md}               Round 2 augmented seed-term list
│       └── table_4.{tex,md}               per-subreddit vocabulary rates
│
├── submissions/                           arXiv-ready bundle (rebuild per academic_publishing/ARXIV_SUBMISSION.md)
│   └── leffew_2026_lcr_preprint_rewrite_v1-arxiv.tar.gz
│
├── data/                                  corpus data
│   └── lcr_pass1b_canonical.csv           26,158 rows (1,173 posts + 24,985 comments)
│
├── deliverables/                          analytical outputs
│   ├── lcr_cases_coded_v2.csv             35 hand-coded cases (five-property schema)
│   ├── hand_coding/
│   │   └── round_1_positive_cases.csv     35 confirmed-positive case set
│   ├── term_validation/
│   │   ├── seed_terms_round_2.csv         Round 2 augmented seed list
│   │   └── round_2_term_validation.csv    Round 2 term retain/drop tally
│   ├── phase_2_5_pass1b_sense_discovery/  sense-discovery verdicts
│   └── phase_4_voice_segmentation/        segmentation confidence + LLM call log
│
├── src/                                   analysis pipeline (Python)
│   ├── pullpush_lcr_scraper.py            Arctic Shift wholesale scrape
│   ├── stitch_lcr_corpus.py               per-subreddit consolidation
│   ├── exclude_stub_posts.py              [removed]/[deleted] stub filter
│   ├── refetch_lcr_comments_seed_filtered.py  comment refetch on seed-matched posts
│   ├── lcr_round2_retrieval.py            iterative seed-term refinement (Round 2)
│   ├── phase2_pass1b_freq_collocation.py  Phase 2 descriptive engagement
│   ├── phase2_pass1b_lcr_kwic_network.py  Phase 2 KWIC + collocation network
│   ├── phase_2_5_pass1b_sense_discovery.py    sense-discovery for polysemous seeds
│   ├── phase_2_5_sense_discovery_professional.py  anchor-strategy for 'professional'
│   ├── voice_segmentation_lcr_v3_span.py  Phase 4 voice segmentation
│   ├── generate_paper_figures.py          paper figures from data + coded CSV
│   ├── generate_paper_tables.py           paper tables from data + coded CSV
│   └── emit_references_section.py         APA-formatted References block from .bib
│
└── prior_artifacts/                       foundational phenomenological priors
    ├── leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf
    ├── leffew_2025_nlp_evaluation_of_ai_safety_guardrails.html
    └── leffew_2025_claude_sonnet_harmful_helpfulness_companion.html
```

---

## Reproducing the analyses

The full pipeline is reproducible from the raw Arctic Shift scrape onward. Phases follow the procedural method's `§C` notation.

```bash
git clone https://github.com/htleffew/claude-lcr-analysis
cd claude-lcr-analysis
pip install -r requirements.txt

# Phase 1: corpus assembly
python src/pullpush_lcr_scraper.py              # Arctic Shift wholesale pull
python src/stitch_lcr_corpus.py                 # per-subreddit consolidation
python src/exclude_stub_posts.py                # filter [removed]/[deleted] stubs
python src/refetch_lcr_comments_seed_filtered.py  # comments on seed-matched post IDs

# Phase 2: descriptive engagement
python src/phase2_pass1b_freq_collocation.py    # frequency + collocation
python src/phase2_pass1b_lcr_kwic_network.py    # KWIC + co-occurrence network

# Phase 2.5: sense discovery
python src/phase_2_5_pass1b_sense_discovery.py
python src/phase_2_5_sense_discovery_professional.py

# Phase 4: voice segmentation (current canonical)
python src/voice_segmentation_lcr_v3_span.py

# Round 2 retrieval (iterative seed-term refinement)
python src/lcr_round2_retrieval.py

# Paper artifacts
python src/generate_paper_tables.py
python src/generate_paper_figures.py
```

Hand-coding of the 35 confirmed positive cases against the five-property structural signature was performed interactively; the resulting CSV is at [`deliverables/lcr_cases_coded_v2.csv`](deliverables/lcr_cases_coded_v2.csv) and is licensed CC BY 4.0 for independent re-coding.

To rebuild the PDF from the LaTeX source:

```bash
cd paper
pdflatex leffew_2026_lcr_preprint_rewrite_v1.tex
pdflatex leffew_2026_lcr_preprint_rewrite_v1.tex   # second pass for cross-references
```

To rebuild and verify the arXiv submission bundle, follow `academic_publishing/ARXIV_SUBMISSION.md`: stage the `article`-class source with its figures and table inputs (no `.cls`/`.bbl` needed), then clean-room compile before upload.

```bash
python ../Pm_html/academic_publishing/build_arxiv_bundle.py \
  paper/leffew_2026_lcr_preprint_rewrite_v1.tex --out submissions/
```

---

## Citation

```
Leffew, H. (2026). Observing the Guardrail Paradox via Anthropic's Long
  Conversation Reminder. Preprint, HAIIQU.

Leffew, H. (2025). Gaslighting in the name of AI safety: How Anthropic's
  Claude Sonnet 4.5 went from "you're absolutely right!" to "you're
  absolutely crazy." Medium, October 16, 2025.
```

A formal `CITATION.cff` is provided at the repository root.

---

## License

- **Code** (`src/`, paper-build scripts): MIT (see [`LICENSE-code`](LICENSE-code))
- **Data, prior artifacts, paper, deliverables**: CC BY 4.0 (see [`LICENSE-data`](LICENSE-data))

---

## Contact

[HAIIQU](https://haiiqu.com/) &middot; heatherleffew@forevueinsights.com
