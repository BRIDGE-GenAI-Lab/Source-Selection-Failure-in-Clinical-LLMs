# Large Language Models and Adversarially Edited Clinical Guideline Excerpts: A Paired Forced-Choice Simulation Study

**Authors:** Alon Gorenshtein, Mahmud Omar, Yiftach Barash, Girish N. Nadkarni, Eyal Klang.
Code developed by Alon Gorenshtein. Associated with the Bridge Gen AI Lab, BIDMC,
Harvard Medical School, and Mount Sinai Medical Center, NY.

> **Synthetic clinical data only. Not medical advice.**

## What the study did

Each of 500 synthetic clinical vignettes comes with two guideline excerpts: an authentic one and a modified
one, a rule-based edit of it of one of ten types (dosing error, missing warning, fabricated citation and so on).
A model saw the vignette, the clinical question and both excerpts, in positions A and B, and chose the excerpt it
would rely on. The outcome is whether it chose the authentic excerpt. Twenty-one models evaluated all 500 pairs,
10,500 evaluations in total.

## Correction

The audit of the original experiment found that in 167 of the 500 pairs the two excerpts were identical at the
model input: the edit was absent, or it was written to a field that no runner showed to the models
(`audit/pair_audit_summary.json`). The revised article reanalyzes the original evaluations of the other 333 pairs,
6,993 evaluations (`analysis_original/headline.json`). No new model responses were collected.

## Contents

| Directory | Contents |
|---|---|
| `src/` | Original experiment: case loading, prompts, the OpenAI function-calling runner, scoring |
| `scripts/` | Original experiment: the tool-pair generator (`scripts/generate_tools.py`) and the runners of the other providers |
| `data/` | Vignette batches, the case file as run (`data/original/`), the canonical result file of each of the 21 original runs (`data/original_runs/`) and the 10,500 per-evaluation records (`data/results_10500.csv`) |
| `audit/` | Reconstructed model inputs of the 10,500 evaluations and the audit of the 500 pairs (`audit/README.md`) |
| `analysis_original/` | Reanalysis of the original evaluations: tables, the pooled measures (`analysis_original/headline.json`), the Table 1 examples and Figures 2-4 (`analysis_original/figures/`) |
| `verification/` | Round-1 physician review: both physicians' answers (`verification/round1/`), their summary and the Q3 consensus |
| `r2/` | Code of the audit, the reanalysis, the physician-review summary and the supplementary tables; `r2/run/models_v2.json` holds the model list from which Figure 4 reads the reasoning classification |
| `tests/` | Tests of `r2/` |
| `docs/` | The data dictionary (`docs/DATA_FILES.md`) and the vignette-generation prompt (`docs/r2/vignette_generation_prompt.md`) |

The original runners read their API keys from the environment variables that `.env.example` lists.

## Reproduce

From the repository root, with the package installed (see Installation). Each command overwrites its outputs in
place, so `git status` afterwards shows whether any output changed.

```bash
python -m r2.audit.pair_audit               # audit/pair_audit.csv, pair_audit_by_type.csv, pair_audit_summary.json
python -m r2.audit.self_label_tokens        # audit/self_label_tokens.json
python -m r2.audit.format_cues              # audit/format_cues.json
python -m r2.analysis.original_corrected    # analysis_original/: tables, headline.json, table1_examples.csv
python -m r2.analysis.figures_original      # analysis_original/figures/: Figures 2-4 and their legends
python -m r2.verify.round1_summary          # verification/round1_summary.json, .csv, round1_analyzed_pairs.csv
python -m r2.analysis.clinically_evaluable  # analysis_original/clinically_evaluable.json, .csv
python -m r2.data_files_doc                 # docs/DATA_FILES.md
python -m pytest tests -q                   # the test suite
```

Three audit steps replay the raw responses of the original runs, which are not in this repository:
`python -m r2.audit.original_inputs` (`audit/original_model_inputs.jsonl.gz`, `audit/original_inputs_verification.csv`),
`python -m r2.audit.coerced_choices` (`audit/coerced_choices.csv`, `audit/coerced_choices_summary.json`) and
`python -m r2.audit.openai_tokens` (`audit/openai_input_tokens.csv`, `audit/openai_input_tokens_summary.json`).
Their outputs are committed, and the commands above read them. With the raw run directories, set `R2_LEGACY_ROOT` to
the directory that holds the original evaluation tree (safe-guideline-tooling-eval, with its results directories)
and run these three first. Without it they stop with an
error, and the tests that need the raw files are skipped.

The supplementary tables of the original experiment are written by `python -m r2.analysis.supp_tables <out_dir>`;
`S_run_inventory` also needs `R2_LEGACY_ROOT`.

## Data dictionary

Every data file, with its creator, role and SHA-256: [docs/DATA_FILES.md](docs/DATA_FILES.md).

## Archived release

The exact state cited by the article, including the revision's audit trail and a repeated experiment that was
prepared but not run, is the tag `v2.2-r2`
(https://github.com/BRIDGE-GenAI-Lab/Source-Selection-Failure-in-Clinical-LLMs/tree/v2.2-r2), archived at
https://doi.org/10.5281/zenodo.23501772. The state cited by the earlier version of the article is the tag
`r1-as-published`.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{gorenshtein2026sourceselection,
  title   = {Large Language Models and Adversarially Edited Clinical Guideline Excerpts: A Paired Forced-Choice Simulation Study},
  author  = {Gorenshtein, Alon and Omar, Mahmud and Barash, Yiftach and Nadkarni, Girish N. and Klang, Eyal},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

*Synthetic clinical data only. Not medical advice.*
