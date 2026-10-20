"""Regenerate docs/DATA_FILES.md (one row per tracked data file, with SHA-256 and last commit).

    python -m r2.data_files_doc

Roles live in RULES below. Re-run after any data file changes; tests/r2/test_data_files.py fails
while the document is stale.
"""
import csv
import hashlib
import re
import subprocess

from r2.paths import REPO


def _evaluable_role(lead):
    from r2.analysis.clinically_evaluable import TIERS
    return (f"{lead} for {len(TIERS)} subsets of the 333 analyzed pairs defined by both physicians' answers: Q4, Q2 and "
            "Q7 each; all three (clinically evaluable); the clinically evaluable pairs with Q3 Y or P, and with Q3 Y "
            "(consensus where they disagreed); and the clinically evaluable pairs without the fabricated_citation "
            "pairs (cue-confounded DOI), plus the number of distinct authentic excerpts per subset. A subset whose "
            "records cover only one modified-excerpt position is marked not estimable. Post hoc")


DIRS = ["data", "audit", "verification", "analysis_original"]
FILES = ["docs/r2/vignette_generation_prompt.md", "r2/run/models_v2.json"]  # single tracked files outside DIRS
OUT = REPO / "docs" / "DATA_FILES.md"
ORIG = "original experiment"

# Roles that state counts or a state (answered or not) are read from the file itself, so the document never
# carries stale text.
REVIEWER_NAMES = {"AG": "A.G.", "MO": "M.O."}
QKEYS = ("q1_vignette_consistent", "q2_question_answerable", "q3_authentic_concordant", "q4_authentic_relevant",
         "q5_edit_visible", "q6_edit_matches_type", "q7_edit_reduces_trust", "q8_leakage")


def _csv(path):
    with open(REPO / path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _nrows(path):
    return len(_csv(path))


def _q3_consensus_role(path):
    rows = _csv(path)
    n = {v: sum(r["consensus_q3_authentic_concordant"] == v for r in rows) for v in ("Y", "P", "N")}
    dates = ", ".join(sorted({r["adjudicated_on"] for r in rows}))
    return (f"Joint Q3 consensus of A.G. and M.O. (adjudicated {dates}) for the {len(rows)} clinically evaluable pairs "
            "on which their Q3 answers differed: case_id, sham type, both original answers, the consensus "
            f"(Y {n['Y']}, P {n['P']}, N {n['N']}) and its note, imported from "
            "`verification/round1/q3_consensus_returned.xlsx`. Post hoc; used only for the clinically evaluable "
            "sensitivity analysis")


def _who(path):
    code = re.search(r"review_([A-Z]+)\.", path).group(1)
    return f"{REVIEWER_NAMES.get(code, code)} ({code})"


def _form_role(path):
    rows = _csv(path)
    answered = sum(all((r.get(q) or "").strip() for q in QKEYS) for r in rows)
    state = ("no answers recorded yet" if answered == 0 else
             f"answers recorded for {answered} of {len(rows)} pairs, imported from the returned workbook")
    return (f"Review form of {_who(path)}, round 1: one row per pair ({len(rows)} pairs; independent forms); {state}. "
            "Each pair is shown as it was prepared for a repeated experiment that was not run: every content field of "
            "both excerpts, including fields no original runner serialized, and the full vignette. For the 333 "
            "analyzed pairs the excerpt text and visible metadata equal the as-run excerpts except the DOI suffix of "
            "the 40 fabricated_citation modified excerpts (`verification/round1_summary.json`, "
            "`packets_vs_original_run`)")


# (regex on path, created by, used by, role); role is a string or a function of the path. First match wins.
RULES = [
    (r"data/500cases\.json$", "Gemini 3 Pro in Google AI Studio (copy-paste of the generator output; no script); saved 2026-01-03 12:20",
     "legacy, not used by any run directly",
     "Raw vignette batches. The file holds five batch objects (`Batch5` nested inside `Batch4`): four are top-level list items and `Batch5` (CASE_401-500) sits in `Batch4`'s `cases` array after its 35th case. `generated_utc` stamps: 2025-05-15T14:30:00Z, 14:35, 14:40, 14:45 (Batch4, CASE_301-335 only), 14:50 (Batch5). These stamps were written by the generator inside the JSON and are not real generation times. CASE_336-400 never existed. CASE_335 is truncated (9 of 15 fields, no clinical question). Input to the keyword mapping of `case_assignments.json`"),
    (r"data/500cases_merged\.json$", "unrecorded merge (no script on disk); `generated_utc` 2026-01-03T20:29:39Z",
     "legacy, not used by any run directly",
     "510 cases: the five batches above plus the 75-case batch (CASE_501-575, `Batch_Final_75`, which is not in this repository). All cases are field-identical to the batch files"),
    (r"data/500cases_final\.json$", "unrecorded reduction (no script on disk); `generated_utc` 2026-01-03T20:40:59Z",
     "input to `scripts/generate_tools.py`; vignette source for every case file",
     "500 cases, reduced from 510 by removing the last ten cases by identifier (original CASE_566-575); no content criterion was applied. Original CASE_501-565 were renumbered CASE_336-400. No vignette field changed in the merge, the reduction or the tool-pairing step"),
    (r"data/500cases_with_real_tools\.json$", "`scripts/generate_tools.py` (2026-01-07); published copy",
     f"{ORIG} (published copy); audit",
     "Published copy of the as-run case file. Differs from the as-run bytes (sha 58651ef5...) only in `source_pdf_path` / `source_pdf_used` (absolute path replaced by basename); `tests/r2/test_paths.py` pins this. Holds the 500 cases, their T/S tool pairs, `sham_type_counts`, `domain_counts` (69 guideline topics) and the embedded `tool_audit_log`. Regenerated on 2026-01-07 after the guideline library was expanded from 8 to 69 topics; the 2026-01-03 version (sha 6389e074...) is not preserved"),
    (r"data/original/500cases_with_real_tools\.as_run\.json$", "exact byte copy of the file the models saw; `generate_tools.py` output of 2026-01-07",
     f"{ORIG} (input to all 21 canonical runs); audit; `r2.paths.ORIGINAL_CASES`",
     "Input to every canonical original run, sha 58651ef5... (`generated_utc` 2026-01-07T06:20:44Z). Ground truth for the audit and the input reconstruction"),
    (r"data/case_assignments\.json$", "an R1 keyword-mapping script (2026-01-03) from `500cases.json`",
     "legacy, not used by any canonical run",
     "Keyword-to-guideline mapping of 336 cases (the 336 that existed in `500cases.json`) to 8 guideline domains (AIS, Sepsis, AF, CAP, ACS, UTI, ICH, Seizure): 171 mapped, 165 unmapped. It drove the 2026-01-03 case file, in which 296 of 500 cases received a placeholder tool. Superseded by the 69-topic mapping inside `scripts/generate_tools.py`"),
    (r"data/tool_audit_log\.json$", "`scripts/generate_tools.py` (2026-01-07)",
     "audit (modification-log claims in `pair_audit.csv`)",
     "The generator's modification log: one entry per case (mapped domain, source guideline, T page anchors, primary sham type, `where_in_tool`, `what_changed`). Also embedded in the case file under `tool_audit_log`. Its claims are reconciled against the actual T/S differences in `audit/pair_audit.csv`"),
    (r"data/results_10500\.csv$", "the original runners (`src/` and `scripts/run_*.py`), consolidated by the R1 canonical loader (added to the repo 2026-07-19)",
     f"{ORIG}; legacy R1 analyses",
     "Original per-evaluation results: 10,500 records (21 models x 500 cases; 6,235 correct, accuracy 0.5938). These are the per-evaluation records. They are not in the Zenodo record 10.5281/zenodo.21209850"),
    (r"data/original_runs/manifest\.json$", "export from the original run directories (`results/run_*`), with the selection rule of the R1 canonical loader",
     f"{ORIG}; audit",
     "Selection rule and, per canonical run, model, run directory, record count, run config and SHA-256 of the source `results.jsonl`"),
    (r"data/original_runs/.*\.results\.jsonl$", "copy of the run's `results.jsonl`, exported with `manifest.json`",
     f"{ORIG}; audit",
     "Canonical raw results of one model's original run (500 records; one run per model). Joined to the reconstructed inputs by `case_id`"),
    (r"audit/README\.md$", "authored", "audit documentation",
     "Describes the input reconstruction, what is verified against logged values, every `pair_audit.csv` column, the coerced-choice audit (`coerced_choices.csv`: runner conditions with file:line, columns, counts), the token-count corroboration of the OpenAI inputs (`openai_input_tokens.csv`: method, result, limits, columns) and the self-labelling-token audit (`self_label_tokens.json`: the word \"fake\" in the 40 original fabricated-citation edits)"),
    (r"audit/coerced_choices\.csv$", "`python -m r2.audit.coerced_choices`", "audit; analysis_original (`excluding_coerced`)",
     "One row per original record (10,500): whether the runner, not the model, set the choice to \"A\" (`class`: coerced_parse_failure / coerced_invalid_choice / not_coerced), from a replay of the runner's parse step on its raw response file, checked against the stored decision. 781 coerced, all in the 14 Together-hosted runs (655 unparseable + 126 invalid choices such as \"Neither\"), 368 of them on identical pairs; none in the other 7 runs. `hit_token_limit`: 381 of the parse failures used the runner's full max_tokens (2,048), from the stored usage"),
    (r"audit/coerced_choices_summary\.json$", "`python -m r2.audit.coerced_choices`", "audit",
     "Counts of `coerced_choices.csv` by class x profile x model, by `visible_status` and by profile x `visible_status`; replayed reasons; invalid choice values per model; the three records checked by rationale only; the one record whose stored decision was not written by the runner (gemini CASE_330); runner conditions with file:line; `hit_token_limit` counts by profile x class, its evidence counts, and each runner's max_tokens with file:line"),
    (r"audit/openai_input_tokens\.csv$", "`python -m r2.audit.openai_tokens`", "audit",
     "One row per openai_fc record (2,000): logged `usage.input_tokens` of the first and the decision call, tiktoken `o200k_base` count of the reconstructed input (original `src/runner.py` replayed on the logged replies), offset, and the offsets of two counterfactual inputs with `safety_guardrails` or `exclusions_or_cautions` serialized into both payloads. Decision-call offset 51 (gpt-5-nano 62) in all 1,990 `AB` records and 4 higher in the 10 records with one extra message; the first-call counts are halved when the reply holds two parallel function calls"),
    (r"audit/openai_input_tokens_summary\.json$", "`python -m r2.audit.openai_tokens`", "audit",
     "Per model: offset distribution (median, IQR, min, max, share within k of the median) of the decision call and of each counterfactual, median shift, tokens added, slope of offset on tokens added, per-structure strata, informative subset; first-call diagnostics; every call of the 10 multi-call records; tiktoken version and `encoding_for_model`"),
    (r"audit/self_label_tokens\.json$", "`python -m r2.audit.self_label_tokens`",
     "audit; analysis_original (`fabricated_citation_cue_confounded`)",
     "Rows of the 10,500 reconstructed original inputs whose model-visible text holds a self-labelling token (fake, fabricated, sham, bogus, dummy, placeholder, made-up, made up, not real; whole word, any case), by location (modified payload, authentic payload, prompt text), profile, sham type and model. \"fake\" is in the modified payload of 840 inputs (40 fabricated_citation cases x 21 models; DOI 10.9999/fake.NNNN), so the original fabricated_citation results are cue-confounded. It is also in the instructions of all 2,000 openai_fc inputs (\"fake citations\", the same text for both tools)"),
    (r"audit/format_cues\.json$", "`python -m r2.audit.format_cues`", "audit",
     "The two format cues of the original fabricated_citation inputs, counted over the 10,500 reconstructed inputs: the blank line before the Evidence line in the modified excerpt (840 of 840 fabricated_citation inputs, 0 authentic; 680 in the serialized message, the 160 openai_fc payloads being function outputs) and the cited page span (135 to 1,643 pages, median 1,017), by profile and model; the two notes phrases (\"Supported by recent mega-trial data\", \"Historical guideline version for reference only\"), in the notes of 40 and 50 as-run cases and in 0 model inputs each; and the 42 outdated_version inputs whose modified year is not older than the authentic one (CASE_387, CASE_488 x 21 models)"),
    (r"audit/original_model_inputs\.jsonl\.gz$", "`python -m r2.audit.original_inputs`",
     f"audit; {ORIG}",
     "Rebuilt system prompt, user message and tool payloads for each of the 10,500 original evaluations (original runner code run on the as-run case file), with `hash_status` and `verification_scope`. Joins `data/original_runs/` on model and `case_id`"),
    (r"audit/original_inputs_verification\.csv$", "`python -m r2.audit.original_inputs`", "audit",
     "Per-model counts of verified, no_logged_hash and MISMATCH rows: 10,496 verified, 4 without a logged hash, 0 mismatches. Includes the profile and the 167 identical pairs per model"),
    (r"audit/pair_audit\.csv$", "`python -m r2.audit.pair_audit`", "audit; analysis_original",
     "Classification of all 500 pairs by what a model could see (`visible_status`, `raw_status`), reconciled with the modification log"),
    (r"audit/pair_audit_by_type\.csv$", "`python -m r2.audit.pair_audit`", "audit",
     "Tabulation of `raw_status` and `visible_status` by sham type"),
    (r"audit/pair_audit_summary\.json$", "`python -m r2.audit.pair_audit`", "audit; README correction notice",
     "Overall counts: `visible_status` identical 167 / metadata_only 64 / excerpt_differs 269; `raw_status` visible 333 / hidden_only 110 / no_edit 57; 69 distinct authentic excerpts"),
    (r"analysis_original/headline\.json$", "`python -m r2.analysis.original_corrected`", "analysis of the original runs (exploratory)",
     "Pooled measures for all, discriminable and identical pairs (accuracy, primacy, accuracy by sham position, position-averaged accuracy) with case-cluster bootstrap CIs, and `excluding_coerced`: the same measures without the 781 records whose choice the original runner coerced to \"A\" (`audit/coerced_choices.csv`); and `fabricated_citation_cue_confounded`: the original fabricated_citation accuracy, failure rate, primacy and position rates with case-cluster bootstrap CIs, labelled cue-confounded because every modified excerpt carried the DOI 10.9999/fake.NNNN (`audit/self_label_tokens.json`); the non-identical-pair measures without the 40 cue-confounded fabricated_citation pairs, also without the coerced records; and `confidence`: stated confidence with the R1 definitions (overconfidence, ECE, MCE, Brier, AUROC, high-confidence errors, per-model Mann-Whitney with BH FDR) for all and non-identical pairs, with and without the coerced records, case-cluster bootstrap CIs. Inferential results are exploratory, not pre-specified"),
    (r"analysis_original/confidence_(all|discriminable)\.csv$", "`python -m r2.analysis.original_corrected` (`r2.analysis.confidence_original`)", "analysis of the original runs (confidence, R1 definitions)",
     lambda path: f"Per model ({_nrows(path)} models), {'all pairs' if path.endswith('_all.csv') else 'the non-identical pairs'}: rows with a stated confidence, accuracy, mean confidence (also in correct and incorrect selections), overconfidence, ECE and MCE (ten equal-width bins), Brier score, AUROC, share of failures at confidence >= 0.90, two-sided Mann-Whitney U with BH FDR, and higher confidence when correct (FDR q < 0.05 and AUROC > 0.5)"),
    (r"analysis_original/confidence_discriminable_excluding_coerced\.csv$", "`python -m r2.analysis.original_corrected` (`r2.analysis.confidence_original`)", "analysis of the original runs (confidence, R1 definitions)",
     lambda path: f"Per model ({_nrows(path)} models), the non-identical pairs without the records whose choice the original runner set to A (their confidence was written by the runner, 0.5, or belongs to an answer the runner replaced): the columns of `confidence_discriminable.csv`, with the per-model Mann-Whitney U, BH FDR and higher confidence when correct recomputed, plus n_runner_set_removed"),
    (r"analysis_original/per_model_discriminable_excluding_coerced\.csv$", "`python -m r2.analysis.original_corrected`", "analysis of the original runs (per model without runner-set choices)",
     lambda path: f"Per model ({_nrows(path)} models), the non-identical pairs: evaluations, those whose choice the original runner set to A (n_runner_set), valid evaluations and their share, and position-averaged accuracy and accuracy by modified-excerpt position as recorded and without the runner-set records (`_valid`), with case-cluster bootstrap CIs and the valid n per position. The records left are the replies with a valid A or B choice, a selected subset, not a random sample"),
    (r"analysis_original/clinically_evaluable\.json$", "`python -m r2.analysis.clinically_evaluable`",
     "analysis of the original runs (clinically evaluable pairs; post hoc sensitivity analysis)",
     lambda path: _evaluable_role("Pooled measures (accuracy, primacy, accuracy by modified-excerpt position, "
                                  "position-averaged accuracy, with case-cluster bootstrap CIs, also without the "
                                  "runner-set records)")),
    (r"analysis_original/clinically_evaluable\.csv$", "`python -m r2.analysis.clinically_evaluable`",
     "analysis of the original runs (clinically evaluable pairs; post hoc sensitivity analysis)",
     lambda path: _evaluable_role("The tier counts and measures only (no case ids, no runner-set-excluded measures: "
                                  "the JSON has them)")),
    (r"analysis_original/table1_examples\.csv$", "`python -m r2.analysis.original_corrected` (`table1_examples`)", "Table 1 (exclusion-only reanalysis)",
     lambda path: f"Example original responses, {_nrows(path)} rows: for each sham type with non-identical pairs, a failure and a success, each the first record in (case_id, model) order, among the records whose choice the runner did not coerce, whose model is not yet in the table (else the first in that order); the rationale cut at 300 characters; fabricated_citation rows marked cue-confounded. Rule: `r2/analysis/original_corrected.py` docstring"),
    (r"analysis_original/figures/legend_Figure\d\.md$", "`python -m r2.analysis.figures_original`", "Figures 2-4 (exclusion-only reanalysis)",
     "Figure legend with n and the exclusion of the identical pairs, built from `analysis_original/headline.json` and the `_discriminable` tables"),
    (r"analysis_original/figures/Figure\d\.(png|pdf)$", "`python -m r2.analysis.figures_original`", "Figures 2-4 (exclusion-only reanalysis)",
     "Figure from the original evaluations of the non-identical pairs (2: position effects pooled and by configuration; 3: failure rate by type; 4: position-averaged accuracy by model); PNG at 300 dpi, PDF without a creation date"),
    (r"docs/r2/vignette_generation_prompt\.md$", "copied byte for byte from the original `supplementary_appendix.md` lines 1442-1542, under a 3-line provenance header",
     "documentation of the original stimulus generation; supplement",
     "Zero-shot prompt used in Google AI Studio with Gemini 3 Pro to generate the vignettes. `tests/r2/test_vignette_prompt.py` pins the body to the source lines"),
    (r"analysis_original/", "`python -m r2.analysis.original_corrected`", "analysis of the original runs (exploratory)",
     "Reanalysis of the original 10,500 records, split by pair status (all pairs vs discriminable pairs). Inferential results are exploratory, not pre-specified"),
    (r"verification/round1_analyzed_pairs\.csv$", "`python -m r2.verify.round1_summary`", "physician review (descriptive; exclusion-only reanalysis)",
     lambda path: f"The {_nrows(path)} analyzed pairs (raw_status visible): case_id, sham type and both physicians' round-1 answers to Q1-Q8 (no notes), from the imported round-1 forms. Describes; excludes nothing"),
    (r"verification/round1_q3_consensus\.csv$", "import of `verification/round1/q3_consensus_returned.xlsx`, checked against both physicians' original answers",
     "physician review (Q3 consensus for the clinically evaluable pairs; post hoc)", lambda path: _q3_consensus_role(path)),
    (r"verification/round1_summary\.(json|csv)$", "`python -m r2.verify.round1_summary`", "physician review (descriptive; exclusion-only reanalysis)",
     "Round-1 physician review described per item for all reviewed pairs and for the 333 analyzed pairs: each physician's answer counts, the proportion of identical answers, unweighted Cohen kappa and the pairs both physicians rated N (Q8: Y). Excludes nothing"),
    (r"verification/round1/review_.*\.csv$", "the round-1 review form as sent, with the physician's answers imported from the returned workbook", "physician review (input to `python -m r2.verify.round1_summary`)",
     lambda path: _form_role(path)),
    (r"verification/round1/q3_consensus_form\.xlsx$", "written from the round-1 review forms and both physicians' answers", "physician review (post hoc adjudication)",
     "physician review (Q3 consensus form for the clinically evaluable pairs, as sent; post hoc)"),
    (r"verification/round1/q3_consensus_returned\.xlsx$", "returned by A.G. and M.O.", "physician review (post hoc adjudication)",
     "physician review (Q3 consensus form for the clinically evaluable pairs, as returned by A.G. and M.O., byte for "
     "byte; column H holds their joint answer; post hoc)"),
    (r"r2/run/models_v2\.json$", "authored (model list of the repeated experiment, prepared but not run)",
     "Figure 4 and Supplementary Methods (the `reasoning` field only)",
     "The 21 models with provider, model identifier, `reasoning` flag and decoding settings. The reasoning classification of Figure 4 (`python -m r2.analysis.figures_original`) is read from its `reasoning` field; nothing else uses it"),
]


def _tracked():
    out = subprocess.run(["git", "ls-files", "-z", "--", *DIRS, *FILES], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    return sorted(p for p in out.split("\0") if p and not p.rsplit("/", 1)[-1].startswith("._"))


def _git(path):
    return subprocess.run(["git", "log", "-1", "--format=%h %cs", "--", path], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.split()


def _cell(s):
    return s.replace("|", "/").replace("\n", " ")


def matching_rule(path):
    """The first rule (regex, created by, used by, role) matching `path`, or None. Reads no file."""
    return next((r for r in RULES if re.search(r[0], path)), None)


def rule(path):
    """(created by, used by, role text) of the first rule matching `path`, or None. A role that is a function
    reads the file."""
    got = matching_rule(path)
    if got is None:
        return None
    _, by, used, role = got
    return by, used, (role(path) if callable(role) else role)


def _row(path):
    got = rule(path)
    if got is None:
        raise SystemExit(f"no rule for {path}")
    by, used, role = got
    sha, date = _git(path)[0], _git(path)[1]
    digest = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
    return f"| `{path}` | {_cell(by)}; commit {sha} | {date} | {_cell(role)} | {_cell(used)} | `{digest}` |"


def build():
    head = """# Data files: what each file is, who made it, who uses it

Every tracked file under `data/`, `data/original/`, `data/original_runs/`, `audit/`, `verification/` (including `round1/`) and `analysis_original/`, plus `docs/r2/vignette_generation_prompt.md` and `r2/run/models_v2.json`, has one row. Files of the revision process and of a repeated experiment that was prepared but not run are not on this branch; the complete revision record is at the tag `v2.2-r2`. `tests/r2/test_data_files.py` fails if a file is missing here or a SHA-256 does not match. Regenerate with `python -m r2.data_files_doc`.

- **Created by:** the script or process, and the commit that last changed the file (`git log -1 --format=%h -- <path>`).
- **Date:** date of that commit. Dates inside the data (`generated_utc`) are noted in the role where they matter.
- **Used by:** original experiment / audit / analysis of the original runs / physician review / legacy, not used.
- **SHA-256:** of the file as committed.

## Lineage of the stimuli

1. Vignettes were generated with Gemini 3 Pro in Google AI Studio and saved as batch files. `data/500cases.json` holds five batch objects (four of 100 cases and one, `Batch4`, that stops after 35 cases at CASE_335) with nested `Batch5`; a sixth batch of 75 cases (CASE_501-575) is not in this repository. CASE_336-400 never existed. Generation parameters are not reported, and no record of rejected or regenerated vignettes was retained.
2. 2026-01-03: the batches were merged into `500cases_merged.json` (510 cases) and reduced to `500cases_final.json` (500 cases) by removing the last ten cases by identifier (original CASE_566-575); no content criterion was applied.
3. `scripts/generate_tools.py` paired each case with an authentic tool (T) and a modified tool (S), writing `500cases_with_real_tools.json` (2026-01-07, sha 58651ef5...) and the modification log. This file was regenerated after the guideline library grew from 8 to 69 topics; the 2026-01-03 version is not preserved.
4. All 21 canonical original runs (10,500 evaluations) used the 2026-01-07 file. The audit found that for 167 of 500 pairs the two documents were identical at model input.
5. Two physicians answered the round-1 review forms (`verification/round1/review_AG.csv`, `review_MO.csv`); the reanalysis uses their answers for the 333 analyzed pairs descriptively (`verification/round1_summary.json`) and for a post hoc sensitivity analysis (`analysis_original/clinically_evaluable.json`).

## Files

| Path | Created by | Date | Role | Used by | SHA-256 |
|---|---|---|---|---|---|
"""
    rows = [_row(p) for p in _tracked()]
    tail = """

## Not in this repository

| Item | Where it is | Note |
|---|---|---|
| `75cases.json` (CASE_501-575) | authors' working folder only | Sixth vignette batch; ten of its cases (CASE_566-575) were dropped. Not released, so the 510-case merge cannot be reproduced from released files |
| 2026-01-03 `500cases_with_real_tools.json` (sha 6389e074...) | not preserved | Three gpt-5-nano development runs used it; they were not analyzed |
| Original raw run directories (`results/run_*`) | authors' working folder | `data/original_runs/` holds the canonical `results.jsonl` of each run. The audit steps that replay raw responses read them from `R2_LEGACY_ROOT` |
| Returned round-1 workbooks (`AG_r1.xlsx`, `MO_r1.xlsx`) | authors' working folder | Their answers are in `verification/round1/review_AG.csv` and `review_MO.csv`; their SHA-256 is in `verification/round1_summary.json` (`returned_workbooks`) |
| Guideline PDFs and extracts | authors' working folder | Source of the authentic excerpts; see `scripts/generate_tools.py` |
"""
    return head + "\n".join(rows) + tail, len(rows)


def is_current():
    """True when docs/DATA_FILES.md equals a fresh build (the tracked files, their SHA-256 and their roles)."""
    return OUT.exists() and OUT.read_text() == build()[0]


if __name__ == "__main__":
    text, n = build()
    OUT.write_text(text)
    print(f"{n} rows -> {OUT}")
