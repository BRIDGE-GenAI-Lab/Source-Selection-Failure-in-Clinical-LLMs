# Source-Selection Failure in Clinical Language Models Under Adversarial Guideline Edits

Code and data release for the study *Source-Selection Failure in Clinical Language Models
Under Adversarial Guideline Edits*.

**Authors:** Alon Gorenshtein, Mahmud Omar, Yiftach Barash, Girish N. Nadkarni, Eyal Klang.
Code developed by Alon Gorenshtein. Associated with the Bridge Gen AI Lab, BIDMC,
Harvard Medical School, and Mount Sinai Medical Center, NY.

> **Synthetic clinical data only — not medical advice.**

---

## Abstract

We evaluated 21 large language models on a controlled two-document source-discrimination
task built from 500 synthetic, physician-reviewed clinical vignettes, yielding 10,500
model evaluations. In each item a model chose between an authentic clinical guideline and
an adversarially edited ("sham") counterpart, and we measured whether the model selected
the safe, authentic source. Overall accuracy was only 59.4%; on safety-critical
modifications models selected the harmful document 48.8% of the time, and selection was
dominated by document order rather than content — models chose the first-position document
72.7% of the time (position odds ratio 0.087 for correct identification when the sham
appeared first versus second).

## Repository layout

```
.
├── src/                 Core harness: case loading, prompts, runner, scoring, schemas
├── scripts/             Per-provider run scripts, scoring, and figure/appendix generation
│   └── analysis/        Canonical re-analysis + statistics + figure regeneration (this study)
├── data/                Synthetic clinical cases and paired guideline tools
│   └── outputs/         Processed result tables that back the manuscript numbers
├── pyproject.toml       Package metadata and dependencies
├── .env.example         Template for API keys (copy to .env; never commit .env)
└── LICENSE              MIT
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
# analysis scripts additionally need scientific dependencies:
pip install numpy pandas scipy statsmodels matplotlib
```

Copy `.env.example` to `.env` and add the API keys for the providers you intend to run.
Keys are read from the environment; none are stored in the repository.

## Reproduce

The published aggregates are regenerated from the canonical 10,500-record result set.
Run from the `scripts/analysis/` directory so the relative loader paths resolve:

```bash
cd scripts/analysis

# 1. Rebuild the canonical 21-model x 500-case set and verify the headline gate
#    (models=21, total=10,500, accuracy ~= 59.4%). Writes outputs/canonical_headline.json.
python load_canonical.py

# 2. Position-effect sensitivity (model-adjusted, cluster-robust, GEE), per-model
#    content-sensitivity with FDR (Table S8), and the model x modification-type
#    failure table (Table S9). Writes r2_sensitivity.json, r2_table_s8.csv, r2_table_s9.csv.
python r2_stats.py

# 3. Regenerate Figures 2, 3, and 4 (600 dpi PNG + PDF).
python regen_all_figures.py
```

`master_stats.py`, `a2_position_stratified.py`, `a3_ranking_vs_content.py`, and
`a4_compliance_rationales.py` recompute the remaining supplementary tables from the same
canonical loader. `load_canonical.py` reconstructs one complete run per model from the raw
`results/` runs and asserts the published aggregates as a gate.

> **Note on paths.** `load_canonical.py` expects the raw per-run `results/` directory of the
> evaluation harness (not distributed here). The processed result tables under `data/outputs/`
> are the exact artifacts those scripts produce, so the manuscript numbers can be inspected
> directly without re-running the full 10,500-call experiment. `r2_stats.py` and
> `regen_all_figures.py` contain absolute paths from the authors' working tree; adjust the
> `sys.path` / `BASE` constants near the top of each file to point at your local checkout.

## Data dictionary (`data/outputs/`)

| File | Contents |
|------|----------|
| `all_model_results.json` | Per-model summary (name, correct, total, accuracy, type, params, reasoning flag, architecture) for all 21 models. |
| `canonical_headline.json` | Headline aggregates: first-position selection rate (`primacy_pct` = 72.74), detection accuracy with sham in position A vs B (`acc_sham_A` = 36.7, `acc_sham_B` = 82.4), and the A–B swing in percentage points. |
| `master_stats.json` | Single source of truth for manuscript numbers: overall accuracy (59.4%) and Wilson CI, per-modification-type and per-category failure rates with 95% CIs, and sham-position counts. |
| `position_stratified.json` | Per-model position-stratified accuracy: overall, sham-in-A, sham-in-B, position-balanced accuracy, Tool-A selection rate, and Wilson CIs. |
| `ranking_vs_content.json` | Content-beyond-position decomposition: overall and per-model tests of whether `acc_sham_A + acc_sham_B > 1` (content used beyond mere ordering), with z and p values. |
| `compliance.json` | Classification of failure rationales into positional / equivalence / content / other, with counts and percentage shares. |
| `r2_sensitivity.json` | Position-effect odds ratios (sham-in-A vs B) under three specifications: model-adjusted logistic (OR 0.087), case-cluster-robust, and GEE exchangeable. |
| `r2_table_s8.csv` | Supplementary Table S8: per-model content sensitivity with FDR-adjusted significance. |
| `r2_table_s9.csv` | Supplementary Table S9: model x modification-type failure-rate matrix. |

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{gorenshtein2026sourceselection,
  title   = {Source-Selection Failure in Clinical Language Models Under Adversarial Guideline Edits},
  author  = {Gorenshtein, Alon and Omar, Mahmud and Barash, Yiftach and Nadkarni, Girish N. and Klang, Eyal},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

*Synthetic clinical data only — not medical advice.*
