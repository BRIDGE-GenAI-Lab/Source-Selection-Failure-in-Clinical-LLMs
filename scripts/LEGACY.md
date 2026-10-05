# Legacy scripts: what each did and whether it fed a reported number

The original runner scripts (`scripts/run_*.py`, `scripts/generate_tools.py`, `scripts/map_cases.py` and `src/`) are kept unchanged, so the original experiment stays reproducible and so `python -m r2.audit.original_inputs` can rebuild the exact model inputs from them. Do not edit them. The R2 revision adds its code under `r2/`, never in these files.

Runner profiles (`r2/profiles.py`): `openai_fc` (4 models, `src/runner.py`), `gemini_inline7` (1, `scripts/run_gemini_experiment.py`), `deepseek_chat_inline7` (1, `scripts/run_deepseek_experiment.py`), `deepseek_reasoner_inline7` (1, `scripts/run_deepseek_reasoner.py`), `together_inline6` (14, `scripts/run_together_experiment.py`). Together these produced the 10,500 canonical records (21 models x 500 cases).

"Reported number" means a number in the R1 manuscript or supplement. Every R1 aggregate was computed over all 500 pairs, including the 167 pairs whose two documents were identical at model input (`audit/pair_audit_summary.json`); the corrected figures are in `analysis_original/`.

## Case and tool construction

- `scripts/map_cases.py`: keyword-maps cases to 8 guideline domains and writes `data/case_assignments.json` (2026-01-03). Fed the earlier case file only. No reported number traced to it.
- `scripts/extract_pdfs.py`: extracts text from the guideline PDFs for the tool excerpts. No reported number traced to it.
- `scripts/generate_tools.py`: builds the authentic (T) and modified (S) tools, the sham-type assignment and `data/tool_audit_log.json`; produced the as-run case file. Defined the stimuli behind every number. Its modification log claims more than the files show (`audit/pair_audit.csv`).

## Original runners (kept unchanged)

- `scripts/run_experiment.py`: command line entry to `src/runner.py` (OpenAI tool calling). Profile `openai_fc`. Produced the 4 OpenAI models' results.
- `scripts/run_gemini_experiment.py`: Gemini runner. Profile `gemini_inline7`. Produced gemini-2.5-flash-lite's results.
- `scripts/run_deepseek_experiment.py`: DeepSeek chat runner. Profile `deepseek_chat_inline7`. Produced deepseek-chat's results.
- `scripts/run_deepseek_reasoner.py`: DeepSeek reasoner runner. Profile `deepseek_reasoner_inline7`. Produced deepseek-reasoner's results.
- `scripts/run_together_experiment.py`: Together.ai runner. Profile `together_inline6`. Produced 14 models' results.
- `scripts/run_groq_experiment.py`: Groq runner. No canonical model used it. No reported number traced to it.
- `scripts/run_opensource_experiment.py`: local HuggingFace runner. No canonical model used it. No reported number traced to it.
- `scripts/run_opensource_mps.py`: local Apple-silicon runner (small Qwen2.5 models). No canonical model used it. No reported number traced to it.
- `scripts/score_and_report.py`: command line scoring and report from one run directory (`src/scoring.py`, `src/report.py`). Per-run scoring only. No reported number traced to it.

## R1 statistics (all-pairs; superseded by `analysis_original/`)

- `scripts/analysis/load_canonical.py`: selects one run per model and builds the 21 x 500 canonical set; asserts the 10,500 gate. Yes: defines the canonical set (`data/outputs/canonical_headline.json`).
- `scripts/analysis/master_stats.py`: master table of headline numbers (`data/outputs/master_stats.json`). Yes.
- `scripts/analysis/a2_position_stratified.py`: per-model position-stratified accuracy (`position_stratified.json`). Yes.
- `scripts/analysis/a3_ranking_vs_content.py`: ranking versus content decomposition (`ranking_vs_content.json`). Yes.
- `scripts/analysis/a4_compliance_rationales.py`: keyword classification of failure rationales (`compliance.json`). Yes.
- `scripts/analysis/r2_stats.py`: R1-response sensitivity (position odds ratio under three specifications; tables S8 and S9). Yes.
- `scripts/analysis/cluster_regression.py`: cluster-aware primary regression (table S11). Yes.
- `scripts/analysis/regen_all_figures.py`: regenerates R1 Figures 2-4. Figures only.
- `scripts/analysis/fig1_schematic.py`: R1 Figure 1 schematic. Figure only.
- `scripts/analysis/preflight_edit.py`: R1 manuscript, supplement and response text edits. No numbers computed (none traced).
- `scripts/calc_overall_pvalue.py`: overall p-value from the raw run directories (exploratory scratch). No reported number traced to it.
- `scripts/calc_position_stats_10500.py`: aggregate position-bias statistics for the 21 models (R1 appendix). Yes, R1 appendix.
- `scripts/compute_10500_stats.py`: statistics over the 10,500 records (R1 appendix). Yes, R1 appendix.
- `scripts/compute_appendix_stats.py`: writes `appendix_stats_10500.json`, the input of the R1 appendix tables. Yes, R1 appendix.
- `scripts/compute_complete_stats.py`: complete statistics including position-bias details (R1 appendix). Yes, R1 appendix.
- `scripts/compute_pvalues.py`: p-values for comparisons requested during R1 drafting, from `appendix_stats_10500.json`. Yes, R1 appendix.
- `scripts/scan_full_dataset.py`: scans every `results.jsonl` (all runs, including partial and duplicate) for aggregates. Exploratory, not the canonical set; no reported number traced.
- `scripts/verify_manuscript_stats.py`: checks manuscript statements against `appendix_stats_10500.json`. A check, not a source.
- `scripts/generate_vulnerability_analysis.py`: early vulnerability analysis and summary table on a hand-picked run list. Superseded; no reported number traced.

## R1 appendix and figure generators (formatting of numbers computed elsewhere)

- `scripts/generate_appendix_v2.py`, `scripts/generate_final_complete_appendix.py`, `scripts/generate_final_merged_appendix_v6.py`: successive versions of the pre-R1 appendix text built from `appendix_stats_10500.json`. They carried R1 appendix numbers; the final text was later superseded by the R1 supplement. The generation-process statements in these appendices (domain table, review description) are retired; see `docs/r2/AUTHOR_FACTS.md`.
- `scripts/generate_figures.py`, `scripts/generate_final_publication.py`, `scripts/generate_nature_figures.py`, `scripts/generate_nature_medicine_figures.py`, `scripts/generate_nejm_figures.py`, `scripts/generate_nejm_10500_figures.py`, `scripts/generate_publication_figures.py`: figure and table generators for earlier journal formats. Figures only; superseded by `scripts/analysis/regen_all_figures.py` and, for R2, by the figures built from the repeated experiment.
