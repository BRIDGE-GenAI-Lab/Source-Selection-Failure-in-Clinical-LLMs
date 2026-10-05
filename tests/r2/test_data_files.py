"""docs/DATA_FILES.md must list every tracked data file and every listed SHA-256 must match."""
import hashlib
import re
import subprocess

from r2.paths import REPO

DOC = REPO / "docs" / "DATA_FILES.md"
DIRS = ["data", "audit", "verification", "analysis_original"]
PENDING = ["analysis_repeated/", "r2_runs/", "data/v2/results_repeated.csv",
           "verification/excluded_pairs.json"]
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
    out = subprocess.run(["git", "ls-files", "-z", "--", *DIRS], cwd=REPO, capture_output=True,
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


def test_future_outputs_are_marked_pending():
    rows = _rows()
    for p in PENDING:
        assert p in rows, p
        assert "produced after the repeated experiment" in rows[p]


def test_required_legacy_rows_are_described():
    rows = _rows()
    assert "58651ef5" in rows["data/original/500cases_with_real_tools.as_run.json"]
    assert "58651ef5" in rows["data/500cases_with_real_tools.json"]
    assert "510" in rows["data/500cases_merged.json"]
    assert "pilot" in rows["data/100cases_with_tools.json"].lower()
    assert "five" in rows["data/500cases.json"].lower()
    assert "10,500" in rows["data/results_10500.csv"]


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
