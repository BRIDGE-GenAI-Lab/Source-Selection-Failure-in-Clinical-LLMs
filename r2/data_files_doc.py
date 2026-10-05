"""Regenerate docs/DATA_FILES.md (one row per tracked data file, with SHA-256 and last commit).

    python -m r2.data_files_doc

Roles live in RULES below. Re-run after any data file changes; tests/r2/test_data_files.py fails
while the document is stale.
"""
import hashlib
import re
import subprocess

from r2.paths import REPO

DIRS = ["data", "audit", "verification", "analysis_original"]
OUT = REPO / "docs" / "DATA_FILES.md"
PEND = "produced after the repeated experiment"
LEG = "Legacy, as run in Jan 2026"
ORIG = "original experiment"

# (regex on path, created by, used by, role). First match wins.
RULES = [
    (r"data/500cases\.json$", "Gemini 3 Pro in Google AI Studio (copy-paste of the generator output; no script); saved 2026-01-03 12:20",
     "legacy, not used by any run directly",
     "Raw vignette batches. The file holds a list of four objects, but there are five dataset headers: `Batch5` (CASE_401-500) is nested inside `Batch4`'s `cases` array after its 35th case. `generated_utc` stamps: 2025-05-15T14:30:00Z, 14:35, 14:40, 14:45 (Batch4, CASE_301-335 only), 14:50 (Batch5). These stamps were written by the generator inside the JSON and are not real generation times. CASE_336-400 never existed. CASE_335 is truncated (9 of 15 fields, no clinical question). Input to `scripts/map_cases.py`"),
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
     "legacy R1 numbers, superseded by `analysis_original/` and the repeated experiment",
     "R1 aggregate table computed over all 500 pairs, including pairs whose two documents were identical at model input. Retained as published in R1"),
    (r"data/outputs/(all_model_results\.json|TableS_position_stratified\.csv)$", "R1 analysis (generating script not identified in this repository)",
     "legacy R1 numbers, superseded",
     "R1 per-model aggregate table. Retained as published in R1"),
    (r"data/v2/pairs_v2\.json$", "`python -m r2.stimuli.build_v2`",
     "repeated experiment (stimulus set); physician verification",
     "Corrected stimulus set: for each resolved case, the authentic and the modified tool with the edit placed in a field the model sees; records `derived_from_sha256` (58651ef5...). Currently 477 resolved pairs and 23 unresolved pairs that need a physician-authored edit; regenerated when those edits arrive"),
    (r"data/v2/manual_edit_queue\.csv$", "`python -m r2.stimuli.build_v2`",
     "repeated experiment (physician edit queue)",
     "The 23 pairs for which no automatic edit could be made; each needs an edit written by a physician"),
    (r"audit/README\.md$", "authored", "audit documentation",
     "Describes the input reconstruction, what is verified against logged values, and every `pair_audit.csv` column"),
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
    (r"analysis_original/", "`python -m r2.analysis.original_corrected`", "analysis of the original runs (exploratory)",
     "Corrected reanalysis of the original 10,500 records, split by pair status (all pairs vs discriminable pairs). Inferential results are exploratory, not pre-specified"),
    (r"verification/leakage_flags\.json$", "`r2.verify` (located leakage allow-list)", "repeated experiment; physician verification",
     "Cases whose vignette text contains condition-tag substrings (e.g. `gi_bleed`) that appear as plain words in the tool; shown to reviewers under Q8"),
    (r"verification/round1/README_REVIEWERS\.md$", "`python -m r2.verify.build_packets`", "physician verification", "Reviewer instructions and the eight questions (Q1-Q8)"),
    (r"verification/round1/packets\.html$", "`python -m r2.verify.build_packets`", "physician verification", "Rendered review packets: the text exactly as shown to the models"),
    (r"verification/round1/packets_manifest\.json$", "`python -m r2.verify.build_packets`", "physician verification",
     "Binds the forms to `data/v2/pairs_v2.json` by SHA-256 and lists one packet hash per case; scoring refuses forms from a different stimulus set"),
    (r"verification/round1/review_.*\.(csv|xlsx)$", "`python -m r2.verify.build_packets`", "physician verification",
     "Review form for one reviewer (AG = A.G., MO = M.O.; independent forms), one row per pair in the stimulus set at the time of building (477 pairs); no answers recorded yet"),
]


def _tracked():
    out = subprocess.run(["git", "ls-files", "-z", "--", *DIRS], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    return sorted(p for p in out.split("\0") if p and not p.rsplit("/", 1)[-1].startswith("._"))


def _git(path):
    return subprocess.run(["git", "log", "-1", "--format=%h %cs", "--", path], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.split()


def _cell(s):
    return s.replace("|", "/").replace("\n", " ")


def _row(path):
    for rx, by, used, role in RULES:
        if re.search(rx, path):
            break
    else:
        raise SystemExit(f"no rule for {path}")
    sha, date = _git(path)[0], _git(path)[1]
    digest = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
    return f"| `{path}` | {_cell(by)}; commit {sha} | {date} | {_cell(role)} | {_cell(used)} | `{digest}` |"


def build():
    head = """# Data files: what each file is, who made it, who uses it

Every tracked file under `data/`, `data/original/`, `data/original_runs/`, `data/v2/`, `audit/`, `verification/` (including `round1/`) and `analysis_original/` has one row. `tests/r2/test_data_files.py` fails if a file is missing here or a SHA-256 does not match. Regenerate with `python -m r2.data_files_doc`.

- **Created by:** the script or process, and the commit that last changed the file (`git log -1 --format=%h -- <path>`).
- **Date:** date of that commit. Dates inside the data (`generated_utc`) are noted in the role where they matter.
- **Used by:** original experiment / repeated experiment / audit only / legacy, not used.
- **SHA-256:** of the file as committed.

## Lineage of the stimuli

1. Vignettes were generated with Gemini 3 Pro in Google AI Studio and saved as batch files. `data/500cases.json` holds five batch objects (four of 100 cases and one, `Batch4`, that stops after 35 cases at CASE_335) with nested `Batch5`; a sixth batch of 75 cases (CASE_501-575) is not in this repository. CASE_336-400 never existed. Generation parameters are not reported, and no record of rejected or regenerated vignettes was retained.
2. 2026-01-03: the batches were merged into `500cases_merged.json` (510 cases) and reduced to `500cases_final.json` (500 cases) by removing the last ten cases by identifier (original CASE_566-575); no content criterion was applied.
3. `scripts/generate_tools.py` paired each case with an authentic tool (T) and a modified tool (S), writing `500cases_with_real_tools.json` (2026-01-07, sha 58651ef5...) and the modification log. This file was regenerated after the guideline library grew from 8 to 69 topics; the 2026-01-03 version is not preserved.
4. All 21 canonical original runs (10,500 evaluations) used the 2026-01-07 file. The audit found that for 167 of 500 pairs the two documents were identical at model input.
5. The repeated experiment uses the corrected stimuli in `data/v2/pairs_v2.json`.

## Files

| Path | Created by | Date | Role | Used by | SHA-256 |
|---|---|---|---|---|---|
"""
    rows = [_row(p) for p in _tracked()]
    pend = [
        ("analysis_repeated/", "`python -m r2.analysis.repeated`", "pre-specified analysis outputs of the repeated experiment (per model, per type, GEE, calibration, sensitivity S1-S5, completeness)"),
        ("r2_runs/", "`r2/run/harness.py` (stored outside the repository under `R2_RUNS_V2`, default `/Volumes/Lexar/LLM_Tool_use/r2_runs/`)", "raw logged record of every repeated-experiment API call (pilot, main run, logs)"),
        ("data/v2/results_repeated.csv", "export of `r2_runs/` after the completeness gate", "one row per repeated evaluation, joined to `pairs_v2.json` by `case_id`"),
        ("verification/excluded_pairs.json", "`r2.verify.score_review` after consensus", "pairs excluded by the physician verification"),
    ]
    prow = [f"| `{p}` | {by} | n/a | {role}. Not yet present: {PEND} | repeated experiment | n/a |" for p, by, role in pend]
    tail = """

## Not in this repository

| Item | Where it is | Note |
|---|---|---|
| `75cases.json` (CASE_501-575) | authors' working folder only | Sixth vignette batch; ten of its cases (CASE_566-575) were dropped. Not released, so the 510-case merge cannot be reproduced from released files |
| 2026-01-03 `500cases_with_real_tools.json` (sha 6389e074...) | not preserved | Three gpt-5-nano development runs used it; they were not analyzed |
| Original raw run directories (`results/run_*`) | authors' working folder; to be archived with the release | `data/original_runs/` holds the canonical `results.jsonl` of each run |
| Guideline PDFs and extracts | authors' working folder | Source of the authentic excerpts; see `scripts/generate_tools.py` |
"""
    return head + "\n".join(rows + prow) + tail, len(rows) + len(prow)


if __name__ == "__main__":
    text, n = build()
    OUT.write_text(text)
    print(f"{n} rows -> {OUT}")
