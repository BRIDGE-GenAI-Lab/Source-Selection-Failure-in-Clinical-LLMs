"""Regenerate docs/DATA_FILES.md (one row per tracked data file, with SHA-256 and last commit).

    python -m r2.data_files_doc

Roles live in RULES below. Re-run after any data file changes; tests/r2/test_data_files.py fails
while the document is stale.
"""
import csv
import hashlib
import json
import re
import subprocess

from r2.paths import REPO

DIRS = ["data", "audit", "verification", "analysis_original", "analysis_repeated"]
FILES = ["docs/r2/vignette_generation_prompt.md", "docs/r2/FREEZE_RECORD.md"]  # single tracked files outside DIRS
OUT = REPO / "docs" / "DATA_FILES.md"
LEG = "Legacy, as run in Jan 2026"
ORIG = "original experiment"

# Roles that state counts or a state (answered or not, resolved or not) are read from the file itself, so the
# document (and Supplementary Table S15, built from it) never carries stale text.
REVIEWER_NAMES = {"AG": "A.G.", "MO": "M.O."}
QKEYS = ("q1_vignette_consistent", "q2_question_answerable", "q3_authentic_concordant", "q4_authentic_relevant",
         "q5_edit_visible", "q6_edit_matches_type", "q7_edit_reduces_trust", "q8_leakage")


def _json(path):
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def _csv(path):
    with open(REPO / path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _nrows(path):
    return len(_csv(path))


def _pairs_role(path):
    d = _json(path)
    cases, unresolved = d.get("cases", []), d.get("unresolved") or []
    phys = sum((c.get("edit_record") or {}).get("edit_source") == "physician_manual" for c in cases)
    if unresolved:
        state = (f"Currently {len(cases)} resolved pairs and {len(unresolved)} unresolved pairs that need a "
                 "physician-authored edit; regenerated when those edits arrive")
    else:
        state = (f"All {len(cases)} pairs resolved, {phys} of them by a physician-written edit "
                 "(`data/v2/manual_edits.json`)")
    return ("Corrected stimulus set: for each resolved case, the authentic and the modified tool with the edit placed "
            f"in a field the model sees; records `derived_from_sha256` (58651ef5...). {state}. The 40 "
            "fabricated_citation excerpts carry the neutral DOI suffix 10.9999/cg.<year>.NNNN instead of the original "
            "10.9999/fake.NNNN (`edit_record.self_label_fix`); the build refuses any modified excerpt holding a "
            "self-labelling token")


def _who(path):
    code = re.search(r"review_([A-Z]+)\.", path).group(1)
    return f"{REVIEWER_NAMES.get(code, code)} ({code})"


def _form_role(path):
    rows = _csv(path)
    answered = sum(all((r.get(q) or "").strip() for q in QKEYS) for r in rows)
    rnd = ("round 1, every pair resolved when the forms were built, each shown as prepared for the repeated experiment "
           "(prepared, not run; `verification/round1/README_STATUS.md`)") if "/round1/" in path else \
        "round 2, the pairs rebuilt after round 1"
    state = ("no answers recorded yet" if answered == 0 else
             f"answers recorded for {answered} of {len(rows)} pairs, imported from the returned workbook")
    return (f"Review form of {_who(path)}, {rnd}: one row per pair ({len(rows)} pairs; independent forms); {state}")


def _workbook_role(path):
    rnd = "round 1" if "/round1/" in path else "round 2"
    return (f"Workbook sent to {_who(path)} for {rnd}: the rows of the matching .csv form with answer dropdowns. The "
            "returned copy is kept outside the repository; its answers are imported into the .csv")


def _merged_role(path):
    return (f"Merged review form of {_who(path)}: the round-1 answers of the pairs unchanged since round 1 and the "
            f"round-2 answers of the pairs rebuilt after it ({_nrows(path)} pairs); input to `score_review`")


def _excluded_role(path):
    d = _json(path)
    return (f"Pairs excluded from the primary analysis by the rule fixed in the plan (consensus N to any of Q3-Q7): "
            f"{len(d.get('excluded', []))} pairs, with the reasons, and `pairs_v2_sha256` binding them to the "
            "verified stimulus file. Written by the physician verification before the freeze; a frozen input")


def _flags_role(path):
    d = {k: v for k, v in _json(path).items() if k != "synthetic"}
    return (f"Pairs flagged for sensitivity analysis S3 (consensus N to Q1 or Q2, P to Q3, or Y to Q8): {len(d)} pairs, "
            "with the reasons. A flag does not exclude a pair")


def _freeze_role(path):
    from r2 import freeze as F
    text = (REPO / path).read_text(encoding="utf-8")
    pub = F.public_timestamp(text)
    changes = F.change_entries(text)
    return (f"Freeze record of the analysis plan (tag {F.TAG}): the base commit, the date ({F.record_date(text)}), the "
            "SHA-256 of the five frozen inputs and the analysis environment. Public timestamp: "
            + (f"{pub['kind']} {pub['url']}, public since {pub['since']}" if pub else "not yet recorded")
            + f". Dated changes after the freeze: {len(changes) or 'none'}")


def _runs_manifest_role(path):
    d = _json(path)
    return (f"Path, size and SHA-256 of every run-log file of the repeated experiment under `r2_runs/` (pilot, main "
            f"with its incident files and `_superseded/`, job logs): {d['totals']['files']} files, "
            f"{d['totals']['bytes']:,} bytes; and the SHA-256 of the tarball `{d['tarball']['name']}` "
            f"({d['tarball']['sha256'][:12]}...) deposited with the Zenodo version")


# (regex on path, created by, used by, role); role is a string or a function of the path. First match wins.
RULES = [
    (r"data/500cases\.json$", "Gemini 3 Pro in Google AI Studio (copy-paste of the generator output; no script); saved 2026-01-03 12:20",
     "legacy, not used by any run directly",
     "Raw vignette batches. The file holds five batch objects (`Batch5` nested inside `Batch4`): four are top-level list items and `Batch5` (CASE_401-500) sits in `Batch4`'s `cases` array after its 35th case. `generated_utc` stamps: 2025-05-15T14:30:00Z, 14:35, 14:40, 14:45 (Batch4, CASE_301-335 only), 14:50 (Batch5). These stamps were written by the generator inside the JSON and are not real generation times. CASE_336-400 never existed. CASE_335 is truncated (9 of 15 fields, no clinical question). Input to `scripts/map_cases.py`"),
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
    (r"data/100cases_with_tools\.json$", "earlier version of the tool generator (Dec 2025); mtime 2025-12-27",
     "legacy, not used",
     "Dec 2025 pilot file: the first 100 vignettes (field-identical to batch 1) with an earlier tool-pair format. Used only by the 2025-12-27 pilot runs. Not used for any reported number"),
    (r"data/case_assignments\.json$", "`scripts/map_cases.py` (2026-01-03) from `500cases.json`",
     "legacy, not used by any canonical run",
     "Keyword-to-guideline mapping of 336 cases (the 336 that existed in `500cases.json`) to 8 guideline domains (AIS, Sepsis, AF, CAP, ACS, UTI, ICH, Seizure): 171 mapped, 165 unmapped. It drove the 2026-01-03 case file, in which 296 of 500 cases received a placeholder tool. Superseded by the 69-topic mapping inside `scripts/generate_tools.py`"),
    (r"data/tool_audit_log\.json$", "`scripts/generate_tools.py` (2026-01-07)",
     "audit (modification-log claims in `pair_audit.csv`)",
     "The generator's modification log: one entry per case (mapped domain, source guideline, T page anchors, primary sham type, `where_in_tool`, `what_changed`). Also embedded in the case file under `tool_audit_log`. Its claims are reconciled against the actual T/S differences in `audit/pair_audit.csv`"),
    (r"data/results_10500\.csv$", "`src/` and `scripts/run_*.py`, consolidated by `scripts/analysis/load_canonical.py` (added to the repo 2026-07-19)",
     f"{ORIG}; legacy R1 analyses",
     "Original per-evaluation results: 10,500 records (21 models x 500 cases; 6,235 correct, accuracy 0.5938). These are the per-evaluation records. They are not in the Zenodo record 10.5281/zenodo.21209850 (see README)"),
    (r"data/original_runs/manifest\.json$", "`python -m r2.export_original_runs`",
     f"{ORIG}; audit",
     "Selection rule and, per canonical run, model, run directory, record count, run config and SHA-256 of the source `results.jsonl`"),
    (r"data/original_runs/.*\.results\.jsonl$", "`python -m r2.export_original_runs` (copy of the run's `results.jsonl`)",
     f"{ORIG}; audit",
     "Canonical raw results of one model's original run (500 records; one run per model). Joined to the reconstructed inputs by `case_id`"),
    (r"data/outputs/(canonical_headline|master_stats|position_stratified|ranking_vs_content|compliance|r2_sensitivity|r2_table_s8|r2_table_s9|r2_table_s11)\.(json|csv)$", "R1 analysis scripts in `scripts/analysis/` (see README)",
     "legacy R1 numbers, superseded by `analysis_original/`",
     "R1 aggregate table computed over all 500 pairs, including pairs whose two documents were identical at model input. Retained as published in R1"),
    (r"data/outputs/(all_model_results\.json|TableS_position_stratified\.csv)$", "R1 analysis (generating script not identified in this repository)",
     "legacy R1 numbers, superseded",
     "R1 per-model aggregate table. Retained as published in R1"),
    (r"data/v2/pairs_v2\.json$", "`python -m r2.stimuli.build_v2` (strictly, with the physician edits, by `python -m r2.pipeline import-edits`)",
     "repeated experiment (prepared, not run; stimulus set, a frozen input); physician verification",
     lambda path: _pairs_role(path)),
    (r"data/v2/manual_edit_queue\.csv$", "`python -m r2.stimuli.build_v2`",
     "repeated experiment (prepared, not run; physician edit queue)",
     "The 23 pairs for which no automatic edit could be made; each needs an edit written by a physician"),
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
    (r"verification/leakage_flags\.json$", "`r2.verify` (located leakage allow-list)", "repeated experiment (prepared, not run); physician verification",
     "Cases whose vignette text contains condition-tag substrings (e.g. `gi_bleed`) that appear as plain words in the tool; shown to reviewers under Q8"),
    (r"verification/round1/README_REVIEWERS\.md$", "`python -m r2.verify.build_packets`", "physician verification", "Reviewer instructions and the eight questions (Q1-Q8). Its sentence that the packets are exactly what the evaluated models receive refers to the repeated experiment, which was prepared, not run (`README_STATUS.md`)"),
    (r"verification/round1/packets\.html$", "`python -m r2.verify.build_packets`", "physician verification", "Rendered round-1 review packets: each pair as prepared for the repeated experiment (prepared, not run), with every content field and the full vignette (`README_STATUS.md`)"),
    (r"verification/round1/packets_manifest\.json$", "`python -m r2.verify.build_packets`", "physician verification",
     "Binds the forms to `data/v2/pairs_v2.json` by SHA-256 and lists one packet hash per case; scoring refuses forms from a different stimulus set"),
    (r"verification/round1_analyzed_pairs\.csv$", "`python -m r2.verify.round1_summary`", "physician review (descriptive; exclusion-only reanalysis)",
     lambda path: f"The {_nrows(path)} analyzed pairs (raw_status visible): case_id, sham type and both physicians' round-1 answers to Q1-Q8 (no notes), from the imported round-1 forms. Describes; excludes nothing"),
    (r"verification/round1/README_STATUS\.md$", "`python -m r2.verify.round1_summary`", "physician review (descriptive; exclusion-only reanalysis)",
     "Status of the round-1 forms: they show each pair as prepared for the repeated experiment (prepared, not run), every content field and the full vignette, more than any original configuration presented; the R2 reanalysis uses the answers for the analyzed pairs descriptively"),
    (r"verification/round1_summary\.(json|csv)$", "`python -m r2.verify.round1_summary`", "physician review (descriptive; exclusion-only reanalysis)",
     "Round-1 physician review described per item for all reviewed pairs and for the 333 analyzed pairs: each physician's answer counts, the proportion of identical answers, unweighted Cohen kappa and the pairs both physicians rated N (Q8: Y). Excludes nothing"),
    (r"verification/round1/review_.*\.csv$", "`python -m r2.verify.build_packets`; answers imported by `python -m r2.verify.round1_summary` or `python -m r2.pipeline consensus`", "physician verification",
     lambda path: _form_role(path)),
    (r"verification/round1/review_.*\.xlsx$", "`python -m r2.verify.build_packets`", "physician verification",
     lambda path: _workbook_role(path)),
    (r"verification/round1/q3_consensus_form\.xlsx$", "`python -m r2.verify.q3_adjudication write`", "physician review (post hoc adjudication)",
     "physician review (Q3 consensus form for the clinically evaluable pairs, as sent; post hoc)"),
    # round 2: the pairs rebuilt after round 1 (python -m r2.pipeline import-edits writes the forms)
    (r"verification/round2/README_REVIEWERS\.md$", "`python -m r2.pipeline import-edits`", "physician verification", "Reviewer instructions and the eight questions (Q1-Q8), round 2"),
    (r"verification/round2/packets\.html$", "`python -m r2.pipeline import-edits`", "physician verification", "Rendered round-2 review packets of the pairs rebuilt after round 1, as prepared for the repeated experiment (prepared, not run)"),
    (r"verification/round2/packets_manifest\.json$", "`python -m r2.pipeline import-edits`", "physician verification",
     "Binds the round-2 forms to `data/v2/pairs_v2.json` by SHA-256 and lists one packet hash per rebuilt case"),
    (r"verification/round2/review_.*\.csv$", "`python -m r2.pipeline import-edits`; answers imported by `python -m r2.pipeline consensus`", "physician verification",
     lambda path: _form_role(path)),
    (r"verification/round2/review_.*\.xlsx$", "`python -m r2.pipeline import-edits`", "physician verification",
     lambda path: _workbook_role(path)),
    # after both rounds: python -m r2.pipeline consensus
    (r"verification/review_(AG|MO)\.csv$", "`python -m r2.pipeline consensus` (`r2.verify.build_packets.merge_rounds`)", "physician verification (input to scoring)",
     lambda path: _merged_role(path)),
    (r"verification/packets_manifest\.json$", "`python -m r2.pipeline consensus` (`r2.verify.build_packets.merge_rounds`)", "physician verification",
     "Binds the merged forms to `data/v2/pairs_v2.json`: its SHA-256, one packet hash per case and the merged rounds; `score_review` refuses forms from a different stimulus set"),
    (r"verification/kappa\.csv$", "`python -m r2.pipeline consensus` (`r2.verify.score_review.score`)", "physician verification; supplement (verification table); numbers.json (`verif_kappa_*`)",
     "Per item (Q1-Q8): the share of pairs on which the two reviewers agree and Cohen's kappa"),
    (r"verification/adjudication_needed\.csv$", "`python -m r2.pipeline consensus` (`r2.verify.score_review.score`)", "physician verification (adjudication)",
     lambda path: f"Every item on which the two reviewers disagree (case_id, item, the two answers): {_nrows(path)} items. The physicians agree a consensus answer for each in `consensus.csv`"),
    (r"verification/consensus\.csv$", "the two physicians (adjudication); copied in by `python -m r2.pipeline consensus --consensus`", "physician verification",
     lambda path: f"The consensus answer for every disagreement (case_id, item, consensus): {_nrows(path)} rows. `score_review.apply_consensus` refuses a row for an item the reviewers agreed on"),
    (r"verification/excluded_pairs\.json$", "`python -m r2.pipeline consensus` (`r2.verify.score_review.apply_consensus`)", "repeated experiment (prepared, not run; primary analysis, a frozen input checked by the harness)",
     lambda path: _excluded_role(path)),
    (r"verification/flags\.json$", "`python -m r2.pipeline consensus` (`r2.verify.score_review.apply_consensus`)", "repeated experiment (prepared, not run; sensitivity analysis S3)",
     lambda path: _flags_role(path)),
    (r"verification/final_review\.csv$", "`python -m r2.pipeline consensus` (`r2.verify.score_review.apply_consensus`)", "physician verification; supplement; numbers.json",
     lambda path: f"The final answer of every pair and item after consensus (the reviewers' shared answer, else the consensus answer): {_nrows(path)} pairs"),
    # the physician-written edits
    (r"data/v2/manual_edits\.json$", "`python -m r2.pipeline import-edits` (`r2.verify.build_packets.import_author_edits`)", "repeated experiment (prepared, not run; input to `r2.stimuli.build_v2`)",
     lambda path: f"The physician-written edits for the pairs no rule could edit: per case_id the field, the text found and its replacement, the author and the date ({len(_json(path))} edits). `build_v2` applies them"),
    # the freeze
    (r"docs/r2/FREEZE_RECORD\.md$", "`python -m r2.freeze` (the freeze commit); `python -m r2.pipeline record-public` adds the public timestamp", "repeated experiment (prepared, not run; the harness checks the frozen hashes at the tag, the dated changes and the public timestamp before every main-run call); order check; numbers.json (`freeze_date`)",
     lambda path: _freeze_role(path)),
    # the repeated experiment: python -m r2.pipeline probe, pilot-summary, post-results, package-runs
    (r"analysis_repeated/model_availability\.csv$", "`python -m r2.pipeline probe` (`r2.run.probe_models`)", "repeated experiment (prepared, not run); Supplementary Table S2",
     "One probe call per model: whether it was served, the served identifier and fingerprint, or the error"),
    (r"analysis_repeated/pilot_summary\.csv$", "`python -m r2.pipeline pilot-summary`", "repeated experiment (prepared, not run; pilot review; numbers.json pilot keys)",
     "Per model, the pilot's valid replies, non-responses, API errors, truncations, served identifiers, median tokens, first and last call and parse rate. The pilot is never analysed"),
    (r"analysis_repeated/PILOT_REVIEW\.md$", "authored (the operator, from the pilot summary)", "repeated experiment (prepared, not run; pilot review); supplement",
     "The review of the pilot: dates, parse rate per model, served identifiers, median tokens, projected cost, and every change it prompted with its reason"),
    (r"analysis_repeated/completeness\.csv$", "`python -m r2.run.check_complete` (post-results)", "repeated experiment (prepared, not run; completeness gate)",
     "Per model: cells, missing, duplicates, unexpected, ok, non-responses, errors, superseded and persistent errors, unresolved cells, invalid records, harness commits, served identifiers and fingerprints"),
    (r"analysis_repeated/results_repeated\.csv$", "`python -m r2.run.check_complete` (export after the completeness gate)", "repeated experiment (prepared, not run)",
     "One row per final record of the main run (model, served identifier, case_id, order, labels, status, selection, correctness, confidence, rationale, differences found, harness commit, stimulus hash, time, error); joins `data/v2/pairs_v2.json` on `case_id`"),
    (r"analysis_repeated/r2_runs_manifest\.json$", "`python -m r2.pipeline package-runs`", "repeated experiment (prepared, not run; archive of the raw logs)",
     lambda path: _runs_manifest_role(path)),
    (r"analysis_repeated/headline\.json$", "`python -m r2.analysis.repeated` (post-results)", "repeated experiment (prepared, not run; pre-specified primary outcome); numbers.json",
     "Primary outcome (pooled position-averaged accuracy with its case-cluster bootstrap CI), order-invariant, position-driven and wrong-both rates, primacy, accuracy by sham position, the McNemar summary, the exclusion rule, substitutions, API errors, the GEE convergence flag and the dates of the main run"),
    (r"analysis_repeated/per_model\.csv$", "`python -m r2.analysis.repeated` (post-results)", "repeated experiment (prepared, not run); Supplementary Table (per model); Figures",
     "Per model: position-averaged accuracy (bootstrap CI), accuracy in each order (Wilson CIs), primacy, McNemar n10/n01 with BH-FDR, order-invariant and position-driven rates"),
    (r"analysis_repeated/per_type\.csv$", "`python -m r2.analysis.repeated` (post-results)", "repeated experiment (prepared, not run); Supplementary Table (per type)",
     "Per modification type: rates with case-cluster bootstrap CIs"),
    (r"analysis_repeated/per_category\.csv$", "`python -m r2.analysis.repeated` (post-results)", "repeated experiment (prepared, not run)",
     "Per modification category: rates with case-cluster bootstrap CIs"),
    (r"analysis_repeated/pair_level\.csv$", "`python -m r2.analysis.repeated` (post-results)", "repeated experiment (prepared, not run; Figures; Table 1)",
     "Per model and pair: the outcomes of both orders (order-invariant, position-driven, wrong both)"),
    (r"analysis_repeated/gee\.csv$", "`python -m r2.analysis.repeated` (post-results)", "repeated experiment (prepared, not run); Supplementary Table (GEE)",
     "GEE logistic model of correctness (modified-first, category with metadata as reference, model), exchangeable working correlation clustered by vignette"),
    (r"analysis_repeated/calibration_(overall|per_model)\.csv$", "`python -m r2.analysis.repeated` (post-results)", "repeated experiment (prepared, not run); Supplementary Table (calibration)",
     "Calibration of the stated confidence (in [0, 1]): expected calibration error, Brier score and AUROC, overall or per model"),
    (r"analysis_repeated/sensitivity\.json$", "`python -m r2.analysis.repeated` (post-results)", "repeated experiment (prepared, not run; pre-specified sensitivity analyses); Supplementary Table (sensitivity)",
     "Sensitivity analyses S1-S5 of the analysis plan, each with the pooled position-averaged accuracy and its case-cluster bootstrap CI"),
    (r"analysis_repeated/exploratory_.*\.(csv|json)$", "`python -m r2.analysis.repeated` (post-results)", "repeated experiment (prepared, not run; exploratory, labelled as such)",
     "Exploratory analyses E1-E3 of the analysis plan (differences noticed, rationales, concordance with the original experiment); not pre-specified inference"),
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


# Rows for files the RUNBOOK will add, emitted only while the file (or, for a directory, any file under it) is not
# tracked: (path, created by, role, when it appears).
PENDING_ROWS = [
    ("verification/excluded_pairs.json", "`python -m r2.pipeline consensus` (`r2.verify.score_review`)",
     "pairs excluded by the physician verification", "would be produced by the consensus step before a freeze; the repeated experiment was prepared, not run"),
    ("verification/flags.json", "`python -m r2.pipeline consensus` (`r2.verify.score_review`)",
     "pairs flagged for sensitivity analysis S3", "would be produced by the consensus step before a freeze; the repeated experiment was prepared, not run"),
    ("docs/r2/FREEZE_RECORD.md", "`python -m r2.freeze`",
     "the freeze record: frozen hashes, environment, public timestamp, dated changes", "would be produced by the freeze; the repeated experiment was prepared, not run"),
    ("analysis_repeated/", "`python -m r2.pipeline probe`, `pilot-summary`, `post-results` and `package-runs`",
     "model availability, pilot summary and the pre-specified analysis outputs of the repeated experiment (per model, per type, GEE, calibration, sensitivity S1-S5, completeness)",
     "would be produced by the repeated experiment, which was prepared, not run"),
    ("analysis_repeated/results_repeated.csv", "export of `r2_runs/` after the completeness gate",
     "one row per repeated evaluation, joined to `pairs_v2.json` by `case_id`", "would be produced by the repeated experiment, which was prepared, not run"),
    ("analysis_repeated/r2_runs_manifest.json", "`python -m r2.pipeline package-runs`",
     "SHA-256 and size of every raw run-log file under `r2_runs/` and of their tarball", "would be produced by the repeated experiment, which was prepared, not run"),
]


def _present(path, tracked):
    return any(t == path or (path.endswith("/") and t.startswith(path)) for t in tracked)


def build():
    head = """# Data files: what each file is, who made it, who uses it

Every tracked file under `data/`, `data/original/`, `data/original_runs/`, `data/v2/`, `audit/`, `verification/` (including `round1/` and `round2/`), `analysis_original/` and `analysis_repeated/`, plus `docs/r2/vignette_generation_prompt.md` and `docs/r2/FREEZE_RECORD.md`, has one row. The repeated experiment of `docs/r2/RUNBOOK.md` was prepared but not run; the R2 revision reports the reanalysis of the original evaluations (`analysis_original/`). A file that its steps would add has a "Not yet present" row. `tests/r2/test_data_files.py` fails if a file is missing here or a SHA-256 does not match. Regenerate with `python -m r2.data_files_doc`.

- **Created by:** the script or process, and the commit that last changed the file (`git log -1 --format=%h -- <path>`).
- **Date:** date of that commit. Dates inside the data (`generated_utc`) are noted in the role where they matter.
- **Used by:** original experiment / repeated experiment (prepared, not run) / audit only / legacy, not used.
- **SHA-256:** of the file as committed.

## Lineage of the stimuli

1. Vignettes were generated with Gemini 3 Pro in Google AI Studio and saved as batch files. `data/500cases.json` holds five batch objects (four of 100 cases and one, `Batch4`, that stops after 35 cases at CASE_335) with nested `Batch5`; a sixth batch of 75 cases (CASE_501-575) is not in this repository. CASE_336-400 never existed. Generation parameters are not reported, and no record of rejected or regenerated vignettes was retained.
2. 2026-01-03: the batches were merged into `500cases_merged.json` (510 cases) and reduced to `500cases_final.json` (500 cases) by removing the last ten cases by identifier (original CASE_566-575); no content criterion was applied.
3. `scripts/generate_tools.py` paired each case with an authentic tool (T) and a modified tool (S), writing `500cases_with_real_tools.json` (2026-01-07, sha 58651ef5...) and the modification log. This file was regenerated after the guideline library grew from 8 to 69 topics; the 2026-01-03 version is not preserved.
4. All 21 canonical original runs (10,500 evaluations) used the 2026-01-07 file. The audit found that for 167 of 500 pairs the two documents were identical at model input.
5. The repeated experiment, prepared but not run, would use the corrected stimuli in `data/v2/pairs_v2.json`; the round-1 physician review packets were rendered from that file.

## Files

| Path | Created by | Date | Role | Used by | SHA-256 |
|---|---|---|---|---|---|
"""
    tracked = _tracked()
    rows = [_row(p) for p in tracked]
    prow = [f"| `{p}` | {by} | n/a | {role}. Not yet present: {when} | repeated experiment (prepared, not run) | n/a |"
            for p, by, role, when in PENDING_ROWS if not _present(p, tracked)]
    tail = """

## Not in this repository

| Item | Where it is | Note |
|---|---|---|
| `75cases.json` (CASE_501-575) | authors' working folder only | Sixth vignette batch; ten of its cases (CASE_566-575) were dropped. Not released, so the 510-case merge cannot be reproduced from released files |
| 2026-01-03 `500cases_with_real_tools.json` (sha 6389e074...) | not preserved | Three gpt-5-nano development runs used it; they were not analyzed |
| Original raw run directories (`results/run_*`) | authors' working folder; to be archived with the release | `data/original_runs/` holds the canonical `results.jsonl` of each run |
| Guideline PDFs and extracts | authors' working folder | Source of the authentic excerpts; see `scripts/generate_tools.py` |
| `r2_runs/`: the raw log of every repeated-experiment API call (pilot, main run with incident files, job logs) | `R2_RUNS_V2` (default `/Volumes/Lexar/LLM_Tool_use/r2_runs/`); the tarball `r2_runs.tar.gz` (`python -m r2.pipeline package-runs`) goes into the Zenodo version | """ + ("Path, size and SHA-256 of every file, and of the tarball: `analysis_repeated/r2_runs_manifest.json`" if _present("analysis_repeated/r2_runs_manifest.json", tracked) else "Not present: the repeated experiment was prepared, not run; `package-runs` would record every file's SHA-256 in `analysis_repeated/r2_runs_manifest.json`") + """ |
"""
    return head + "\n".join(rows + prow) + tail, len(rows) + len(prow)


def is_current():
    """True when docs/DATA_FILES.md equals a fresh build (the tracked files, their SHA-256 and their roles)."""
    return OUT.exists() and OUT.read_text() == build()[0]


if __name__ == "__main__":
    text, n = build()
    OUT.write_text(text)
    print(f"{n} rows -> {OUT}")
