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
        m = SHA.search(rest)
        if m:
            n += 1
            if hashlib.sha256((REPO / path).read_bytes()).hexdigest() != m.group(1):
                bad.append(path)
    assert n == len(_tracked())
    assert not bad, f"SHA mismatch: {bad}"


def test_every_json_and_csv_input_has_a_sha():
    rows = _rows()
    nosha = [p for p in _tracked() if p.endswith((".json", ".csv", ".jsonl"))
             and not SHA.search(rows.get(p, ""))]
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
