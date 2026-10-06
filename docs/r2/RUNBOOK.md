# Runbook: from the physicians' forms to the rebuilt documents

This is the operator's sequence for the rest of the R2 revision. Every step is one command of
`python -m r2.pipeline` (in `r2/pipeline.py`). Each command prints what it runs. A step stops with
`REFUSED: <reason>` (exit status 2) when an input is missing, an input is rejected (an author edit, a review
form) or a check fails (completeness, order, QA), and with `WAITING: <reason>` (exit status 3) when people must
act first. A harness job reports through its own exit status (section 7). Rerunning a step that already
finished changes nothing.

Run every command from the repository root, `~/npj_r2/r2_repo`, with the project interpreter:

```bash
cd ~/npj_r2/r2_repo
PY=~/.venvs/npj-r2/bin/python
```

**Never set `R2_SANDBOX` for the real run.** With `R2_SANDBOX` set, every write goes to the sandbox, the API is
never called, and the documents are marked SIMULATED. Check that it is unset with `echo "${R2_SANDBOX:-unset}"`.
`R2_REPO` may stay unset; the document scripts then read `~/npj_r2/r2_repo`.

Steps that need the authors' approval are marked **[authors approve]**. The operator does not approve them.

## 0. The rehearsal (already run; rerun after any code change)

```bash
$PY -m r2.rehearse /path/to/new/empty/dir        # outside the repository and outside /Volumes/Lexar/LLM_Tool_use
```

The rehearsal runs every step below in a new sandbox. Simulated physicians fill the forms. A fake API answers
the probe, the pilot and the main run. The fake API has a position bias, invalid replies, `<think>` tags, 400
and 429 errors, and one cell whose error never clears. Before and after the run, the rehearsal hashes the real
`data/v2`, `verification`, `analysis_repeated`, `r2_runs` and `npj_R2_WORKING` folders, and the git state of the
repository. It fails if anything real changed. Its report is `rehearsal_report.md` in the sandbox. Success means:
- every step `ok`, except the first consensus step, which shows `waiting`, and the main run before the public
  timestamp, which shows `refused (expected)`;
- `Post-results checks`: every QA script with 0 failures, and the order check PASS;
- `NUMERIC markers ... NOT filled: 0`;
- `Real-path check: unchanged: True`.

Nothing in the sandbox is data. Never copy anything from it into the repository or into `npj_R2_WORKING`.
In a sandbox, every git command runs without inherited `GIT_*` variables and only on a repository inside the
sandbox (`r2/sandbox_git.py`); `python -m r2.freeze` refuses to run under `R2_SANDBOX` (use `r2.pipeline freeze`).

## 1. API keys [authors act]

1. Find every plaintext key in the legacy tree:

   ```bash
   $PY -m r2.pipeline scan-legacy-keys            # default /Volumes/Lexar/LLM_Tool_use; read-only
   ```

   It prints each file that holds a key and the provider of each key, and how many distinct keys each provider
   has. It never prints any part of a key. It skips `._*` files, virtual environments, binary files and git
   repositories (it lists those; check them with the secrets `git grep`).
2. Treat every listed key as compromised. For **each provider the scan lists**, revoke **every** key in that
   provider's dashboard, not only the keys the files hold (a provider can have more than one leaked key: the
   scan of 2026-10-05 found two distinct DeepSeek keys). Then create new keys. A provider the repeated
   experiment does not use (Groq) needs revoking only.
3. Copy `.env.example` to `~/npj_r2/r2_repo/.env`. The `.env` file is gitignored. Put the new keys there, and
   only there, under the four names it lists: `OPENAI_API_KEY`, `TOGETHER_API_KEY`, `DEEPSEEK_API_KEY` and
   `GEMINI_API_KEY`.
4. Check that no key is tracked. This must print nothing:

   ```bash
   git grep -nE "tgp[_]v1[_]|s[k]-[A-Za-z0-9]{20}|AI[z]a[0-9A-Za-z_-]{20}|gs[k]_[A-Za-z0-9]{20}"
   ```

   (The pattern of the project constraints, written with one-character classes so that this file does not match
   itself.)

   That pattern misses OpenAI's `sk-proj-` keys, so also run
   `$PY -m r2.pipeline scan-legacy-keys ~/npj_r2/r2_repo`: it must list only `.env`.
5. The harness and the probe read the keys from `.env`, which wins over an exported variable of the same name.
   Each run prints where every key comes from (`key OPENAI_API_KEY: .../.env`), never any part of it, and a
   `WARNING` when an exported value differs from `.env` (for example a revoked legacy key still exported).

## 2. Round 1 back: import the 23 physician-written edits

**The authors do this.** The two reviewers named in the forms, A.G. and M.O., fill in
`verification/round1/review_AG.xlsx` and `review_MO.xlsx` independently. A.G. also fills in the `Author_edits`
sheet of his workbook: one find and replace per queued pair. Keep the returned files outside the repository, for example in
`~/npj_r2/returned/`.

```bash
$PY -m r2.pipeline import-edits ~/npj_r2/returned/AG_r1.xlsx --author A.G. --date YYYY-MM-DD
```

This command:
- checks every edit and writes the edits to `data/v2/manual_edits.json`;
- rebuilds `data/v2/pairs_v2.json` strictly, so all 500 pairs are present and re-audited. The build stops if
  any pair renders identically;
- removes `manual_edit_queue.csv`;
- writes the round-2 forms for the 23 rebuilt pairs to `verification/round2/`.

**Success:** `"pairs": 500, "unresolved": 0, "identical_visible": 0`.
**If an edit is rejected:** the command stops with `REFUSED: author edit rejected: ...` (exit status 2), naming
the case and the reason, for example "'find' must occur exactly once" or "subtle_inversion edits belong in
excerpt_text". Nothing is written. Send that row back to A.G. and rerun the command.

Commit `data/v2/` and `verification/round2/`.

## 3. Round 2 back: consensus, kappa and exclusions

**The authors do this.** Both physicians return their round-2 workbooks, `verification/round2/review_*.xlsx`.

```bash
$PY -m r2.pipeline consensus AG_r1.xlsx MO_r1.xlsx AG_r2.xlsx MO_r2.xlsx      # full paths
```

The first run imports the four workbooks and merges round 1 with round 2. It refuses a form whose packet
content was edited, and a pair that changed after it was reviewed. It writes `verification/kappa.csv` and
`verification/adjudication_needed.csv`. If the physicians disagree on any item, it stops with
`WAITING: N disagreements ...`. The two physicians then agree a consensus answer for each listed row. They write
the answers to `verification/consensus.csv` with the columns `case_id,item,consensus`. Rerun the same command;
it then writes `excluded_pairs.json`, `flags.json` and `final_review.csv`.

**Success:** `"pairs_reviewed": 500`, with the excluded and flagged counts. The exclusion rule is fixed in
`r2/verify/score_review.py`. Commit `verification/`.

## 4. Budget and probe [authors approve the budget]

Show the authors the projected token use and the current prices; see F8 in
`.superpowers/sdd/.../author-facts-findings.md` for the projection. Once they approve:

```bash
$PY -m r2.pipeline probe --yes
```

The probe makes one call per model. It writes `analysis_repeated/model_availability.csv`, which
Supplementary Table S2 needs. Without `--yes`, the command only prints what it would run.
**Success:** every model shows `available`, with a served identifier.

## 5. Pilot and pilot review

```bash
$PY -m r2.pipeline pilot --yes                 # the first 10 pairs, every model, both orders: 420 calls
$PY -m r2.pipeline pilot-summary               # reads the pilot; writes analysis_repeated/pilot_summary.csv
```

`pilot_summary.csv` gives `numbers.json` the pilot dates and the range of parse rates (`pilot_call_dates`,
`pilot_parse_rate_min`, `pilot_parse_rate_max`, `pilot_n_models`) for the authors' pilot sentence.
Write `analysis_repeated/PILOT_REVIEW.md` from the summary. Record:
- the pilot dates;
- the parse rate per model (target: at least 95% `ok`);
- the served identifiers;
- the median tokens and the projected main-run cost.

Only these changes are allowed before the freeze, each logged in `PILOT_REVIEW.md` with its reason:
- a higher `max_tokens` for a model whose non-responses are truncations (the `truncated` column);
- a `served_prefix`;
- a substitution under the plan's substitution rule.

How to record a substitution in `r2/run/models_v2.json` (S4 and S5/E2 depend on it):
- keep `original_name` exactly as in the original experiment: S5 and E2 join the two experiments on it;
- set `model_id` (and `provider` if it changes) to the new identifier, and `served_prefix` if the provider serves
  it under another name;
- add `"substituted_for": "<the model_id of the original experiment>"`: S4 excludes every spec that has it;
- log it in `PILOT_REVIEW.md` with the reason.

After the freeze, models_v2.json is a frozen input: a substitution then also needs a dated entry in
`docs/r2/FREEZE_RECORD.md` (section 7), and the superseded results file moves to `r2_runs/main/_superseded/`.

After any change, rerun the pilot. The pilot is never analyzed. The supplement marker "pilot dates, parse rate
per model and any change it prompted" is written by the authors from `PILOT_REVIEW.md` and the pilot keys.

**Commit `analysis_repeated/` (`model_availability.csv`, `pilot_summary.csv`, `PILOT_REVIEW.md`) before the
freeze:** the freeze refuses a working tree that is not clean.

## 6. Freeze, then make the tag public, BEFORE the main run [authors approve]

```bash
$PY -m r2.pipeline freeze --dry-run            # prints the hashes it would freeze; changes nothing
$PY -m r2.pipeline freeze --yes                # after the authors approve: commit + local annotated tag
```

The freeze refuses if:
- `excluded_pairs.json` is missing;
- `pairs_v2.json` does not hold 500 resolved pairs;
- a model has no pilot records;
- the working tree is dirty;
- the tag already exists;
- the plan has lost its DRAFT banner.

The freeze writes `docs/r2/FREEZE_RECORD.md`; its date fills `freeze_date`. Nothing is pushed.

**Pushing the tag requires the authors' approval.** The operator never pushes on his own. Once the authors
approve, push the tag (or register the frozen plan on OSF, or deposit it on Zenodo), then record the public
timestamp:

```bash
git push origin r2-analysis-plan-frozen        # only after the authors approve the push
$PY -m r2.pipeline record-public https://github.com/<owner>/<repo>/releases/tag/r2-analysis-plan-frozen \
    [--date YYYY-MM-DD]                        # or https://osf.io/<id>, or 10.5281/zenodo.<n>; default today
git add docs/r2/FREEZE_RECORD.md && git commit -m "r2: public timestamp of the freeze"
```

`record-public` accepts only the pushed tag's URL (`.../releases/tag/r2-analysis-plan-frozen` or
`.../tree/r2-analysis-plan-frozen`), an OSF link or a Zenodo DOI, and a calendar date from the freeze day (the
record's date and the freeze commit's date, UTC) to today. A GitHub tag date is set by the client; prefer an OSF
registration or a Zenodo deposit of the frozen plan, whose date the server sets, and keep its JSON record. It writes the URL, its kind, the date and the tag commit into `FREEZE_RECORD.md`, in place of "Public
timestamp: pending a push". These fill the "public timestamp of the freeze" markers.

**`pipeline main` refuses until `FREEZE_RECORD.md` records the public timestamp** (`REFUSED: ... records no
public timestamp of the freeze`). The only override is `--accept-unpublished-freeze`: the run then starts, the
command prints `ACCEPTED: main run without a public timestamp ...`, and an incident of kind
`unpublished_freeze` is written to `r2_runs/main/freeze.incidents.jsonl`. Use it only on the authors'
explicit decision. Post-results checks the order (section 8) and fails if the freeze date or the public
timestamp is after the first main-run call, unless that incident exists (then it prints `ACCEPTED`).

## 7. Main run

The harness itself enforces the freeze before any main-run call, so a direct `python -m r2.run.harness` call
is guarded too; `pipeline main` runs the same check first and refuses with `REFUSED` (exit status 2). The main
run refuses unless:
- the tag `r2-analysis-plan-frozen` exists;
- `pairs_v2.json` has the hash recorded in `excluded_pairs.json`;
- `FREEZE_RECORD.md` at the tag holds the five frozen hashes, and the current `FREEZE_RECORD.md` keeps those lines
  unchanged;
- each of the five frozen files (`data/v2/pairs_v2.json`, `r2/stimuli/render.py`,
  `verification/excluded_pairs.json`, `r2/run/models_v2.json`, `r2/run/harness.py`) has its frozen SHA-256, or a
  dated entry under "Changes after the freeze" in `FREEZE_RECORD.md` gives its new SHA-256 (see below);
- `FREEZE_RECORD.md` records the public timestamp (section 6), unless `--accept-unpublished-freeze`;
- `r2/`, `data/v2/`, `tests/`, `verification/` and `docs/r2/FREEZE_RECORD.md` are committed.

**A change to a frozen file after the freeze** (for example a bug fix in `harness.py` during the run, or a
substitution in `models_v2.json`) needs the authors' decision. Then add one line under "Changes after the
freeze" in `docs/r2/FREEZE_RECORD.md`, dated from the freeze day to today, and commit it with the change:

```
- YYYY-MM-DD: <path> sha256 <new SHA-256>: <reason>
```

The harness then prints `FROZEN INPUT CHANGED <path>` and writes a `frozen_input_changed` incident to
`r2_runs/main/freeze.incidents.jsonl` at every main-run start. Report every such change in the paper. Never edit
the frozen hash lines themselves: the harness reads them from the tag.

Run one background job per provider. The command prints the exact harness call:

```bash
mkdir -p /Volumes/Lexar/LLM_Tool_use/r2_runs/logs
for p in openai together deepseek gemini; do
  nohup $PY -m r2.pipeline main --yes --models provider:$p > /Volumes/Lexar/LLM_Tool_use/r2_runs/logs/$p.log 2>&1 &
done
```

Watch the logs with the Monitor tool. A job that dies resumes when you rerun the same command: completed cells
are skipped. Each job ends with `FINISHED <n> model(s); stopped <k>: [...]; <m> cells need a resume`. Its exit
status is 0 only when no model stopped and no cell needs a resume.

What to do for each line the harness prints:

| Line or incident | Meaning | What to do |
|---|---|---|
| `RETRY <model> try i/10: ... sleeping s` | A transient error (429, timeout, 5xx). Backoff honours `Retry-After`, capped at 900 s. | Nothing. |
| `PENDING <model>: N cells end in an error ...` | API cell errors (400/422) were recorded; those cells are not complete yet. | Rerun the same command once. A re-attempt with the same error becomes final. |
| `FINAL <model> <case> <order>: ... persistent_error` | The same cell error happened twice (compared by `error_key`, so request ids are ignored). The cell is final as a non-response. Incident `persistent_error`. | Nothing, unless the error is provider-wide. Then fix the cause and rerun with `--retry-persistent` (incident `reopened_persistent`). |
| `STOPPED <model>: ModelMismatch` | The provider served a different model. Incident `model_mismatch` keeps the paid responses. | A human decision under the substitution rule. Do not resume until it is made. After a substitution, move the superseded file to `main/_superseded/`. |
| `STOPPED <model>: ModelUnavailable` | Authentication, permission or not-found errors; a 429 that persisted for 10 tries; or the circuit breaker (3 consecutive cell errors, or the same error text 3 times in a row). Incident `stopped`. | Read `<safe_name>.incidents.jsonl` and the last records. Fix provider-wide causes (key, quota, outage), then rerun. |
| `STOPPED <model>: RunLocked` | Another process is running that model. | Do not start a second one. |
| `STOPPED <model>: StaleRecords` | Stored records were made with another stimulus file, prompt or spec. | Stop and find out why. Rerun with `--accept-spec-change` only for a deliberate, documented change; it logs incidents of kind `accepted_spec_change`. |
| `STOPPED <model>: CorruptResults` | A results file has an unparseable line that is not the last one. | Stop. Do not edit the file; report it. |
| `WARNING: dropped torn final line` | A crash left half a line at the end of a file. The harness truncates it and logs incident `torn_tail` with the dropped bytes. | Nothing. |
| `STOPPED <model>: unexpected ...` | A bug or an unknown error. | Stop and report with the traceback. |
| `REFUSED: the analysis plan is not frozen: ...` | No freeze tag. | Go back to step 6. |
| `REFUSED: ... records no public timestamp of the freeze` | The freeze is not public yet. | Section 6: the authors approve the push; then `record-public`. |
| `ACCEPTED: main run without a public timestamp ...` | `--accept-unpublished-freeze` was given. Incident `unpublished_freeze` in `freeze.incidents.jsonl`. | Only on the authors' decision; the methods must then say so. |
| `refusing the main run: <path> has sha256 ..., but FREEZE_RECORD.md at the tag records ...` | A frozen file changed after the freeze, without a dated change entry. | Revert the change, or (authors' decision) add the dated entry above, commit, rerun. |
| `FROZEN INPUT CHANGED <path>: ...` | A documented change to a frozen file. Incident `frozen_input_changed`. | Nothing more; the paper reports it. |
| `refusing the main run: ...` (other) | The freeze tag is missing, the stimulus hash differs, the record at the tag is missing, or its hash lines were edited. | Go back to step 6, or rebuild consistently. |
| `commit r2/, data/v2/, tests/, verification/, docs/r2/FREEZE_RECORD.md before running` | Code, stimuli, verification files or the freeze record are uncommitted. | Commit, then rerun. |
| `STOPPED <model>: StaleRecords ... a response to another input is never kept` | A stored record's prompt or model mapping differs. `--accept-spec-change` covers decoding changes only. | Restore the spec, or move the file aside under the substitution rule. |

Every `*.incidents.jsonl` file (including `freeze.incidents.jsonl`) and the job logs in `r2_runs/logs/` stay in
`r2_runs/`; `package-runs` (section 9) archives all of them with a SHA-256 manifest.

## 8. Post-results: analyses, tables, figures, documents, checks

```bash
$PY -m r2.pipeline post-results                # add --no-render to skip the LibreOffice PDFs
```

The steps run in order. The first failure stops the command with `REFUSED: <step>: <reason>` (exit status 2):
1. the completeness gate (`check_complete`): every model has 1,000 cells, no duplicates, no invalid records and
   no unresolved errors. On failure, read the `UNRESOLVED` lines and go back to step 7;
2. the repeated analyses (`analysis_repeated/`); the flat export `analysis_repeated/results_repeated.csv` is
   written by the gate (not into `data/v2/`, so a later harness run is not blocked);
3. the order check. It prints two lines, and both must say PASS:
   `ORDER CHECK freeze: <freeze date> <= first main-run call <time>: PASS` and
   `ORDER CHECK public: <public date> <= first main-run call <time>: PASS` (compared by day). It fails when either
   date is after the first main-run call, or when it cannot be made (no freeze date, no public timestamp, no
   call dates). An `unpublished_freeze` incident turns a failed public line into `ACCEPTED`. Nothing is built
   after a failure;
4. Supplementary Tables S2, S7, S8 and S11 to S14 (`npj_R2_WORKING/tables/`), and S15 once the release is
   recorded;
5. Table 1;
6. Figures 2 to 4 (`npj_R2_WORKING/05_Figures/`);
7. `edits/numbers.py` and `edits/numbers_supplement.py`;
8. the manuscript (`edits/run_all.py`, which also runs the numbers test and the idempotency check), the
   supplement and the response;
9. the four QA scripts. Each prints one line, and each must say PASS:
   `QA qa_numbers.py: UNEXPLAINED: 0: PASS`, `QA qa_xref.py: Cross-reference failures: 0: PASS`,
   `QA qa_keyfacts.py: Key-fact failures: 0: PASS`, `QA qa_hygiene.py: Total hygiene hits: 0: PASS`. Each script
   also exits non-zero on any failure. One failure stops post-results with `REFUSED: qa: QA failed: <script>:
   <verdict> (see <report>)`. Read that report (`npj_R2_WORKING/qa_*_report.md`; a FAIL line names the place),
   fix the source (the builder or the content, never the report), and rerun post-results;
10. `PENDING_INDEX.md`;
11. the remaining markers, with the list in `npj_R2_WORKING/MARKERS_REMAINING.md`.

**Success:**
- every ORDER CHECK and QA line says PASS;
- before the release, `NUMERIC (should be 0)` counts only the release markers: `release_tag`,
  `zenodo_new_doi` and Supplementary Table S15 (9 markers in the current documents: `release_tag` 6,
  `zenodo_new_doi` 2, S15 1); after the release (section 9) it is 0;
- every other remaining marker is PROSE.

Then the authors:
- write the PROSE markers in `edits/content.py` and `build_response.py`, using the numbers.json keys, such as
  `{{rep_head_primary_position_averaged_accuracy}}` (keys are listed in `edits/numbers.json`);
- check the sentences in the confirm lists of `PENDING.md` and `PENDING_response.md`;
- trim the text to the word limits.

Rerun `post-results` after every change.

## 9. Release QA, then the release [authors approve the release]

**Every check runs before any tag, push or Zenodo step.** A stale or failing file found after the tag is public
and the DOI is minted cannot be fixed in that release.

1. post-results has passed (section 8): every ORDER CHECK and QA line says PASS, and `NUMERIC` counts only the
   release markers.
2. Package the run logs (every job has ended; no harness may be running):

   ```bash
   $PY -m r2.pipeline package-runs
   ```

   It writes `analysis_repeated/r2_runs_manifest.json` (path, size and SHA-256 of every file under `r2_runs/`:
   the pilot, the main run with its incident files and `_superseded/`, the job logs, the probe's availability
   file; totals by directory) and the tarball `r2_runs_archive/r2_runs.tar.gz` next to `r2_runs/`, holding exactly
   those files; the manifest records the tarball's SHA-256 and size. It excludes `*.lock`, `._*` and
   `.DS_Store`, and refuses while a harness holds a model lock or when a log holds a plaintext key. Before
   archiving, read the incident files for provider error text that echoes part of a key, and scrub it.
3. Regenerate the data-file list: `$PY -m r2.data_files_doc`.
4. Commit the outputs: `git add analysis_repeated docs && git commit -m "r2: repeated-experiment results"`.
5. The release QA:

   ```bash
   $PY -m r2.pipeline release-check
   ```

   It prints one line per check, and every line must say PASS: `RELEASE CHECK clean tree`, `DATA_FILES.md
   current`, `DATA_FILES.md complete` (no "Not yet present" row left; Supplementary Table S15 is built from it),
   `run-log manifest` (the manifest matches every file under `r2_runs/` and the tarball), `numbers.json`
   (`edits/test_numbers.py`), `QA scripts` (the four verdicts) and `test suite`. On a FAIL, fix the cause, rerun
   the step that writes the file, commit, and rerun `release-check`. Nothing is tagged until it passes.

Then, after the authors approve, in this order:

6. create the release tag (`git tag -a <tag> -m ...`);
7. push the branch and the tags;
8. mint the new Zenodo version from the pushed tag. The GitHub integration archives only the git tree, so
   upload `r2_runs_archive/r2_runs.tar.gz` and `analysis_repeated/r2_runs_manifest.json` to that version as
   separate files, and check the uploaded tarball's SHA-256 against the manifest. These are the logs of every
   repeated-experiment call that the Data Availability statement promises.

Then record the release and fill the release markers:

```bash
$PY -m r2.pipeline record --release-tag <tag> --zenodo-doi 10.5281/zenodo.<n>   # the tag must exist
$PY -m r2.pipeline post-results                # fills release_tag, zenodo_new_doi and Supplementary Table S15
git add docs/r2/PUBLIC_RECORD.json && git commit -m "r2: release record"
```

S15 is built only from a `DATA_FILES.md` that equals a fresh build and has no "Not yet present" row; otherwise
post-results refuses. **Success:** `NUMERIC (should be 0): 0`, and `git status` shows no change under
`analysis_repeated/`. Before submission, PROSE must also be 0: no `⟦PENDING` anywhere.

## Files each step writes

| Step | Writes |
|---|---|
| import-edits | `data/v2/manual_edits.json`, `data/v2/pairs_v2.json`, `verification/round2/*` |
| consensus | `verification/round{1,2}/review_*.csv`, `verification/review_*.csv`, `packets_manifest.json`, `kappa.csv`, `adjudication_needed.csv`, `excluded_pairs.json`, `flags.json`, `final_review.csv` |
| probe | `r2_runs/model_availability.csv`, `analysis_repeated/model_availability.csv` |
| pilot / main | `r2_runs/pilot/*`, `r2_runs/main/*` (results, incidents, locks) |
| pilot-summary | `analysis_repeated/pilot_summary.csv` |
| freeze | `docs/r2/ANALYSIS_PLAN.md` (filled), `docs/r2/FREEZE_RECORD.md`, one commit, one local tag |
| record-public | `docs/r2/FREEZE_RECORD.md` (the public timestamp) |
| main | `r2_runs/main/freeze.incidents.jsonl` (`unpublished_freeze` with `--accept-unpublished-freeze`; `frozen_input_changed` for a documented change) |
| record | `docs/r2/PUBLIC_RECORD.json` |
| post-results | `analysis_repeated/*` (with `results_repeated.csv`), `npj_R2_WORKING/{tables,05_Figures}/*`, the three documents, their PDFs and side files |
| package-runs | `analysis_repeated/r2_runs_manifest.json`, `r2_runs_archive/r2_runs.tar.gz` (next to `r2_runs/`) |
| data_files_doc | `docs/DATA_FILES.md` |
| release-check | nothing (the QA scripts rewrite their reports in `npj_R2_WORKING`) |
