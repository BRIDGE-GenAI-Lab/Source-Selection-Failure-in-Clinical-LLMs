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


def test_data_files_doc_is_current():
    """DATA_FILES.md equals a fresh build: rows, SHA-256 and the roles read from the files."""
    assert D.is_current(), "docs/DATA_FILES.md is stale: run python -m r2.data_files_doc"


# ---- state-derived roles: nothing stale in DATA_FILES.md (final review I2) ----
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


def test_required_legacy_rows_are_described():
    rows = _rows()
    assert "58651ef5" in rows["data/original/500cases_with_real_tools.as_run.json"]
    assert "58651ef5" in rows["data/500cases_with_real_tools.json"]
    assert "510" in rows["data/500cases_merged.json"]
    assert "five" in rows["data/500cases.json"].lower()
    assert "10,500" in rows["data/results_10500.csv"]


def test_task3b_outputs_are_described():
    rows = _rows()
    assert "781" in rows["audit/coerced_choices.csv"]
    assert "excluding_coerced" in rows["analysis_original/headline.json"]
    assert "Gemini 3 Pro" in rows["docs/r2/vignette_generation_prompt.md"]


def test_sha_is_taken_from_the_last_cell_not_a_hash_quoted_in_the_role():
    quoted = "a" * 64
    real = "b" * 64
    line = f"| `x/y.json` | script; commit abc1234 | 2026-10-05 | quotes `{quoted}` in prose | audit | `{real}` |"
    m = ROW.match(line)
    assert m and m.group(1) == "x/y.json"
    assert _sha_cell(m.group(2)) == real
    assert SHA.search(m.group(2)).group(1) == quoted  # the old first-match rule would pick this one


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


def test_clinically_evaluable_roles_are_split_and_count_the_tiers():
    from r2.analysis.clinically_evaluable import TIERS
    rules = [r for r in D.RULES if r[0].startswith("analysis_original/clinically_evaluable")]
    assert len(rules) == 2
    js, cs = (r[3]("x") for r in rules)
    assert "also without the runner-set records" in js and "also without the runner-set records" not in cs
    assert "tier counts and measures only" in cs
    for text in (js, cs):
        assert f"for {len(TIERS)} subsets" in text and "six" not in text
        assert "fabricated_citation" in text and "distinct authentic excerpts" in text


def test_per_model_role_says_valid_a_or_b_choice():
    rule = next(r for r in D.RULES if "per_model_discriminable_excluding_coerced" in r[0])
    text = rule[3]("analysis_original/per_model_discriminable_excluding_coerced.csv")
    assert "The records left are the replies with a valid A or B choice" in text
    assert "the runner could parse" not in text
