"""docs/DATA_FILES.md must list every tracked data file and every listed SHA-256 must match."""
import hashlib
import re
import subprocess

import pytest

from r2 import data_files_doc as D
from r2.paths import REPO

DOC = REPO / "docs" / "DATA_FILES.md"
DIRS, FILES = D.DIRS, D.FILES
ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|(.*)\|\s*$")
SHA = re.compile(r"`([0-9a-f]{64})`")
SHA_CELL = re.compile(r"`([0-9a-f]{64})`\s*$")


def _sha_cell(rest):
    """The dedicated SHA-256 cell: the last cell of the row, not a hash quoted in the prose."""
    m = SHA_CELL.search(rest.rstrip())
    return m.group(1) if m else None


def _rows():
    rows = {}
    for line in DOC.read_text().splitlines():
        m = ROW.match(line)
        if m:
            rows[m.group(1)] = m.group(2)
    return rows


def _tracked():
    out = subprocess.run(["git", "ls-files", "-z", "--", *DIRS, *FILES], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    return [p for p in out.split("\0") if p and not p.rsplit("/", 1)[-1].startswith("._")]


def test_every_tracked_file_is_listed():
    rows = _rows()
    missing = [p for p in _tracked() if p not in rows]
    assert not missing, f"not in DATA_FILES.md: {missing}"


def test_listed_sha256_matches_file():
    bad = []
    n = 0
    for path, rest in _rows().items():
        sha = _sha_cell(rest)
        if sha:
            n += 1
            if hashlib.sha256((REPO / path).read_bytes()).hexdigest() != sha:
                bad.append(path)
    assert n == len(_tracked())
    assert not bad, f"SHA mismatch: {bad}"


def test_every_json_and_csv_input_has_a_sha():
    rows = _rows()
    nosha = [p for p in _tracked() if p.endswith((".json", ".csv", ".jsonl"))
             and not _sha_cell(rows.get(p, ""))]
    assert not nosha, nosha


def test_pending_rows_appear_only_while_the_file_is_absent():
    """Final review I2: a "Not yet present" row only for a file the RUNBOOK will add and that is not tracked yet;
    once tracked, the file has its own row with a SHA-256."""
    rows, tracked = _rows(), _tracked()
    for path, _by, _role, _when in D.PENDING_ROWS:
        if D._present(path, tracked):
            assert path not in rows or "Not yet present" not in rows[path], path
        else:
            assert "Not yet present" in rows[path], path
    assert "verification/excluded_pairs.json" not in rows or "after the repeated experiment" not in \
        rows["verification/excluded_pairs.json"]  # it is produced before the freeze, and frozen
    assert "`r2_runs/`" in DOC.read_text()  # the raw logs: in "Not in this repository", with the manifest named
    assert "r2_runs_manifest.json" in DOC.read_text()


# Every file the RUNBOOK's steps write into the tracked directories (RUNBOOK "Files each step writes", and the
# outputs of r2/analysis/expected_outputs.json under analysis_repeated/) has a rule.
RUNBOOK_FILES = [
    "data/v2/manual_edits.json", "data/v2/pairs_v2.json",
    *[f"verification/round2/{n}" for n in ("README_REVIEWERS.md", "packets.html", "packets_manifest.json",
                                           "review_AG.csv", "review_AG.xlsx", "review_MO.csv", "review_MO.xlsx")],
    "verification/round1/review_AG.csv", "verification/round1/review_MO.csv",
    "verification/review_AG.csv", "verification/review_MO.csv", "verification/packets_manifest.json",
    "verification/kappa.csv", "verification/adjudication_needed.csv", "verification/consensus.csv",
    "verification/excluded_pairs.json", "verification/flags.json", "verification/final_review.csv",
    "analysis_repeated/model_availability.csv", "analysis_repeated/pilot_summary.csv",
    "analysis_repeated/PILOT_REVIEW.md", "analysis_repeated/results_repeated.csv",
    "analysis_repeated/r2_runs_manifest.json", "docs/r2/FREEZE_RECORD.md",
]


def _expected_analysis_outputs():
    import json
    out = json.loads((REPO / "r2" / "analysis" / "expected_outputs.json").read_text())["outputs"]
    return sorted(p for p in out if p.startswith("analysis_repeated/"))


@pytest.mark.parametrize("path", RUNBOOK_FILES + _expected_analysis_outputs())
def test_every_file_the_runbook_adds_has_a_rule(path):
    assert D.matching_rule(path) is not None, path
    assert any(path.startswith(d + "/") for d in DIRS) or path in FILES, path


def test_data_files_doc_is_current():
    """DATA_FILES.md equals a fresh build: rows, SHA-256 and the roles read from the files."""
    assert D.is_current(), "docs/DATA_FILES.md is stale: run python -m r2.data_files_doc"


# ---- state-derived roles: nothing stale in DATA_FILES.md or in Supplementary Table S15 (final review I2) ----
@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "REPO", tmp_path)
    return tmp_path


def _write(root, rel, text):
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text)


def test_form_roles_follow_the_answers_in_the_file(fake_repo):
    head = "case_id,packet_sha256," + ",".join(D.QKEYS) + ",notes\n"
    _write(fake_repo, "verification/round1/review_AG.csv", head + "C1,x" + "," * 8 + ",\n" + "C2,y" + "," * 8 + ",\n")
    assert "no answers recorded yet" in D.rule("verification/round1/review_AG.csv")[2]
    assert "(2 pairs" in D.rule("verification/round1/review_AG.csv")[2]
    _write(fake_repo, "verification/round1/review_AG.csv", head + "C1,x," + ",".join("Y" * 8) + ",\n" +
           "C2,y," + ",".join("Y" * 8) + ",note\n")
    role = D.rule("verification/round1/review_AG.csv")[2]
    assert "answers recorded for 2 of 2 pairs" in role and "no answers" not in role
    assert "round 2" in (_write(fake_repo, "verification/round2/review_MO.csv", head) or
                         D.rule("verification/round2/review_MO.csv")[2])


def test_pairs_role_follows_the_stimulus_file(fake_repo):
    import json
    _write(fake_repo, "data/v2/pairs_v2.json", json.dumps({"cases": [{"edit_record": {}}] * 3, "unresolved": ["U1"]}))
    assert "Currently 3 resolved pairs and 1 unresolved" in D.rule("data/v2/pairs_v2.json")[2]
    _write(fake_repo, "data/v2/pairs_v2.json", json.dumps(
        {"cases": [{"edit_record": {"edit_source": "physician_manual"}}] * 2 + [{"edit_record": {}}], "unresolved": []}))
    role = D.rule("data/v2/pairs_v2.json")[2]
    assert "All 3 pairs resolved, 2 of them by a physician-written edit" in role and "Currently" not in role


def test_verification_and_freeze_roles_carry_the_counts(fake_repo):
    import json

    from r2 import freeze as F
    _write(fake_repo, "verification/excluded_pairs.json", json.dumps({"excluded": ["C1", "C2"], "pairs_v2_sha256": "x"}))
    _write(fake_repo, "verification/flags.json", json.dumps({"C3": ["q8_leakage"], "synthetic": True}))
    assert "2 pairs" in D.rule("verification/excluded_pairs.json")[2]
    assert "1 pairs" in D.rule("verification/flags.json")[2]
    text = F.freeze_record("0" * 40, {r: "a" * 64 for r in F.HASHED}, "2026-10-06")
    _write(fake_repo, "docs/r2/FREEZE_RECORD.md", text)
    assert "Public timestamp: not yet recorded" in D.rule("docs/r2/FREEZE_RECORD.md")[2]
    _write(fake_repo, "docs/r2/FREEZE_RECORD.md", text.replace(
        "Public timestamp: pending a push.", "- Public URL: https://osf.io/abc12\n- Kind: OSF registration\n"
        "- Public since: 2026-10-07\n- Tag commit: x\n"))
    assert "OSF registration https://osf.io/abc12, public since 2026-10-07" in D.rule("docs/r2/FREEZE_RECORD.md")[2]


def test_required_legacy_rows_are_described():
    rows = _rows()
    assert "58651ef5" in rows["data/original/500cases_with_real_tools.as_run.json"]
    assert "58651ef5" in rows["data/500cases_with_real_tools.json"]
    assert "510" in rows["data/500cases_merged.json"]
    assert "pilot" in rows["data/100cases_with_tools.json"].lower()
    assert "five" in rows["data/500cases.json"].lower()
    assert "10,500" in rows["data/results_10500.csv"]


def test_task3b_outputs_are_described():
    rows = _rows()
    assert "781" in rows["audit/coerced_choices.csv"]
    assert "excluding_coerced" in rows["analysis_original/headline.json"]
    assert "Gemini 3 Pro" in rows["docs/r2/vignette_generation_prompt.md"]


def test_legacy_index_covers_every_script():
    text = (REPO / "scripts" / "LEGACY.md").read_text()
    out = subprocess.run(["git", "ls-files", "-z", "--", "scripts"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    missing = [p for p in out.split("\0") if p.endswith(".py") and f"`{p}`" not in text]
    assert not missing, missing


def test_sha_is_taken_from_the_last_cell_not_a_hash_quoted_in_the_role():
    quoted = "a" * 64
    real = "b" * 64
    line = f"| `x/y.json` | script; commit abc1234 | 2026-10-05 | quotes `{quoted}` in prose | audit | `{real}` |"
    m = ROW.match(line)
    assert m and m.group(1) == "x/y.json"
    assert _sha_cell(m.group(2)) == real
    assert SHA.search(m.group(2)).group(1) == quoted  # the old first-match rule would pick this one


# ---- Supplementary Table S15 is built only from a current DATA_FILES.md with no "Not yet present" row ----
def _s15_world(tmp_path, monkeypatch, doc_text, current=True):
    from r2 import paths
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "DATA_FILES.md").write_text(doc_text)
    monkeypatch.setattr(paths, "REPO", repo)
    monkeypatch.setattr(paths, "SANDBOX", None)
    monkeypatch.setattr(paths, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(D, "is_current", lambda: current)


ROW_OK = "| `data/x.json` | script; commit abc1234 | 2026-10-05 | A file. More text | audit | `" + "a" * 64 + "` |\n"
ROW_PENDING = ("| `verification/flags.json` | consensus | n/a | flags. Not yet present: produced by the physician "
               "verification, before the freeze | repeated experiment | n/a |\n")


def test_s15_refuses_a_stale_data_files_doc(tmp_path, monkeypatch):
    from r2 import pipeline
    _s15_world(tmp_path, monkeypatch, ROW_OK, current=False)
    with pytest.raises(pipeline.Refused, match="stale"):
        pipeline.data_files_table()


def test_s15_refuses_rows_of_files_not_yet_present(tmp_path, monkeypatch):
    from r2 import pipeline
    _s15_world(tmp_path, monkeypatch, ROW_OK + ROW_PENDING)
    with pytest.raises(pipeline.Refused, match="verification/flags.json"):
        pipeline.data_files_table()


def test_s15_from_a_current_doc(tmp_path, monkeypatch):
    import csv

    from r2 import paths, pipeline
    _s15_world(tmp_path, monkeypatch, ROW_OK)
    assert pipeline.data_files_table() == 1
    (row,) = list(csv.DictReader(open(paths.TABLES_DIR / "S_data_files.csv")))
    assert row["path"] == "data/x.json" and row["role"] == "A file." and row["sha256"] == "a" * 64


# ---- the release allowlist: Supplementary Table S14 renders these roles, and its stale-term guard forbids the
# vocabulary of the repeated experiment, which was prepared, not run (R2 decision 2026-10-18) ----
RELEASE_TOP = ("500cases.json", "500cases_merged.json", "500cases_final.json", "case_assignments.json",
               "500cases_with_real_tools.json", "tool_audit_log.json")
RELEASE_FILES = ("data/original/500cases_with_real_tools.as_run.json", "data/results_10500.csv",
                 "verification/round1_analyzed_pairs.csv", "verification/round1_summary.csv",
                 "verification/round1_summary.json", "docs/r2/vignette_generation_prompt.md")
STALE_TERMS = ("repeated experiment", "21,000", "corrected stimulus set", "corrected set", "round 2", "freeze",
               "frozen", "pilot", "harness", "physician-authored", "re-rendered")


def _in_release(path):
    return (path in RELEASE_FILES or path in {f"data/{n}" for n in RELEASE_TOP}
            or path.startswith(("data/original_runs/", "analysis_original/"))
            or (path.startswith("audit/") and not path.endswith(".md")))  # the audit data files


def test_release_allowlist_roles_carry_no_stale_terms():
    paths = [p for p in _tracked() if _in_release(p)]
    assert {f"data/{n}" for n in RELEASE_TOP} | set(RELEASE_FILES) <= set(paths)
    assert any(p.startswith("data/original_runs/") for p in paths) and "audit/format_cues.json" in paths
    hits = {}
    for p in paths:
        by, used, role = D.rule(p)
        text = " ".join((by, used, role)).lower()
        found = [t for t in STALE_TERMS if t in text]
        if role.lstrip().lower().startswith("corrected"):  # read as the unrun corrected stimulus set
            found.append("leading 'Corrected'")
        if found:
            hits[p] = found
    assert not hits, hits
