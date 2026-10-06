"""r2.rehearse: a smoke rehearsal (3 models, no PDF render) in tmp_path fills every NUMERIC marker, marks the
simulated documents, and leaves every real location unchanged."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from r2 import paths, rehearse

REAL_DOCS = Path(paths.REAL_R2_WORKING)
R1_FILES = Path(paths.LEGACY_ROOT) / "npj_R1_SUBMIT_2026-07-15"
pytestmark = pytest.mark.skipif(not (REAL_DOCS / "edits" / "fill.py").exists() or not R1_FILES.exists(),
                                reason="npj_R2_WORKING or the R1 files are not available on this machine")


def _dirty_code():
    return subprocess.run(["git", "status", "--porcelain", "r2"], cwd=paths.REPO, capture_output=True,
                          text=True).stdout.strip()


def _independent_hashes():
    """Hashes taken here, not by r2.rehearse: the real .docx files and the real state directories."""
    out = {}
    roots = [REAL_DOCS, Path(paths.REAL_DATA_V2), Path(paths.REAL_VERIFY_DIR), Path(paths.REAL_ANALYSIS_REPEATED),
             Path(paths.REAL_RUNS_V2), Path(paths.REAL_RUNS_ARCHIVE)]
    for root in roots:
        if root.exists():
            for p in sorted(root.rglob("*")):
                if p.is_file() and not p.name.startswith("._") and "__pycache__" not in p.parts \
                        and p.name != ".DS_Store":
                    out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    if _dirty_code():
        pytest.skip("uncommitted changes under r2/: the supplement quotes committed code only (commit, then rerun)")
    root = tmp_path_factory.mktemp("rehearse") / "box"
    before = _independent_hashes()
    rc, guard = rehearse.rehearse(root, n_models=3, render=False, timeout=1200)
    after = _independent_hashes()
    result = json.loads((root / rehearse.CHILD_RESULT).read_text())
    return {"root": root, "rc": rc, "guard": guard, "before": before, "after": after, "result": result}


def test_smoke_rehearsal_fills_every_numeric_marker(rehearsal):
    res = rehearsal["result"]
    assert rehearsal["rc"] == 0, [s for s in res["steps"] if s["status"] not in rehearse.OK_STATUS]
    assert res["markers_before"]["counts"]["NUMERIC"] > 0
    assert res["markers_after"]["counts"] == {**res["markers_after"]["counts"], "NUMERIC": 0, "UNCLASSIFIED": 0}
    assert res["markers_after"]["counts"]["PROSE"] > 0           # author-written results text stays pending
    report = (rehearsal["root"] / rehearse.REPORT).read_text()
    assert "NOT filled: 0 (goal 0)" in report and "## Real-path check" in report


def test_smoke_rehearsal_exercises_the_harness_failure_paths(rehearsal):
    res = rehearsal["result"]
    first = res["api_events"]["first_pass"]
    for event in ("rate_limited_429", "transient_400", "persistent_400", "invalid_reply", "think_tags"):
        assert first.get(event, 0) > 0, event
    assert res["api_events"]["sleeps_requested_s"]                 # Retry-After honoured (recorded, not slept)
    assert len(res["persistent_records"]) == 1 and "<req>" in res["persistent_records"][0]["error_key"]
    steps = {s["step"]: s for s in res["steps"]}
    assert steps["pipeline consensus (stops for adjudication)"]["status"] == "waiting"
    refused = steps["pipeline main before the freeze is public (refused, as designed)"]
    assert refused["status"] == "refused (expected)" and "no public timestamp" in refused["result"]
    assert steps["pipeline main (fake API, first pass)"]["result"] == 1   # cells need a resume
    assert steps["pipeline main (resume)"]["result"] == 0


def _docx_parts(path):
    with zipfile.ZipFile(path) as z:
        body = z.read("word/document.xml").decode("utf-8")
        headers = [z.read(n).decode("utf-8") for n in z.namelist() if n.startswith("word/header")]
    return body, headers


def test_simulated_documents_carry_sim_values_and_the_watermark(rehearsal):
    work = rehearsal["root"] / "npj_R2_WORKING"
    for name in ("02_Manuscript.docx", "03_Manuscript_HIGHLIGHTED.docx", "04_Supplement.docx",
                 "01_Response_to_Reviewers.docx"):
        body, headers = _docx_parts(work / name)
        assert "[SIM]" in body, name
        assert headers and all("SIMULATED DATA" in h for h in headers), name


def test_rehearsal_leaves_every_real_location_unchanged(rehearsal):
    assert rehearsal["guard"]["unchanged"], rehearsal["guard"]
    assert rehearsal["before"] == rehearsal["after"]
    assert rehearsal["guard"]["n_docx"] >= 4


@pytest.mark.parametrize("where", ["repo", "legacy"])
def test_rehearse_refuses_a_real_location(where):
    target = {"repo": Path(paths.REPO) / "box", "legacy": Path(paths.LEGACY_ROOT) / "box"}[where]
    with pytest.raises(ValueError, match="not a sandbox"):
        rehearse.rehearse(target)
    assert not target.exists()


def test_post_results_checks_pass_in_the_rehearsal(rehearsal):
    post = {s["step"]: s for s in rehearsal["result"]["steps"]}["pipeline post-results"]["result"]
    assert all(v["failures"] == 0 and v["rc"] == 0 for v in post["qa"]["result"].values())
    oc = post["order_check"]["result"]
    assert oc["ok"] is True and oc["public_since"] and oc["freeze_date"] <= oc["first_main_call"][:10]
    record = (rehearsal["root"] / "freeze_repo" / "docs" / "r2" / "FREEZE_RECORD.md").read_text()
    assert f"- Public URL: {rehearse.SIM_TAG_URL}" in record


def test_rehearsal_packages_the_run_logs(rehearsal):
    """Final fix I3: the run-log manifest covers the sandbox r2_runs/ and its tarball sits in the sandbox."""
    from r2 import pipeline
    root = rehearsal["root"]
    man = json.loads((root / "analysis_repeated" / pipeline.RUNS_MANIFEST).read_text())
    assert man["synthetic"] is True and man["totals"]["by_directory"]["main"]["files"] > 0
    assert {"main", "pilot"} <= set(man["totals"]["by_directory"])
    tar = root / "r2_runs_archive" / pipeline.RUNS_TARBALL
    assert hashlib.sha256(tar.read_bytes()).hexdigest() == man["tarball"]["sha256"]
    assert not any(f["path"].endswith(".lock") for f in man["files"])


def test_rehearsal_reports_a_raised_max_tokens_as_a_range(rehearsal):
    """Final review I4: one model's higher max_tokens (a pilot change the RUNBOOK allows) does not stop the numbers
    step; the per-class limit becomes a range and the exception is named."""
    vals = json.loads((rehearsal["root"] / "npj_R2_WORKING" / "edits" / "numbers.json").read_text())["values"]
    assert " to " in vals["rep_maxtok_nonreasoning"]["text"]
    assert vals["rep_maxtok_exceptions"]["text"] != "none" and "tokens" in vals["rep_maxtok_exceptions"]["text"]


GIT_ID = ["-c", "user.name=T", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", "-c",
          "tag.gpgsign=false", "-c", "core.hooksPath=/dev/null"]


def test_the_suite_stays_green_with_the_post_results_files_committed(rehearsal, tmp_path):
    """Final review I2: a scratch clone of this checkout's HEAD (no remote) with the rehearsal's post-results files at
    their real locations (data/v2, verification, analysis_repeated, the filled plan and the freeze record),
    committed, and DATA_FILES.md regenerated: data_files_doc has a rule for every file, no "Not yet present" row
    is left, and the test suite passes in the clone (the rehearsal tests excepted: they made these files)."""
    from r2.sandbox_git import clean_env
    root, clone = rehearsal["root"], tmp_path / "clone"
    env = clean_env({k: v for k, v in os.environ.items() if k not in ("R2_SANDBOX", "R2_REPO", "R2_RUNS_V2")})
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO, env=env, capture_output=True, text=True,
                          check=True).stdout.strip()

    def g(*a):
        return subprocess.run(["git", *GIT_ID, *a], cwd=clone, env=env, capture_output=True, text=True, check=True)
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(paths.REPO), str(clone)], env=env, check=True,
                   capture_output=True)
    g("remote", "remove", "origin")
    g("checkout", "-q", "--detach", head)
    ignore = shutil.ignore_patterns("._*", ".DS_Store", "__pycache__")
    for rel in ("data/v2", "verification", "analysis_repeated"):
        shutil.rmtree(clone / rel, ignore_errors=True)
        shutil.copytree(root / rel, clone / rel, ignore=ignore)
    for rel in ("docs/r2/FREEZE_RECORD.md", "docs/r2/ANALYSIS_PLAN.md"):
        shutil.copyfile(root / "freeze_repo" / rel, clone / rel)
    g("add", "-A")
    g("commit", "-qm", "SIMULATED post-results files")
    doc = subprocess.run([sys.executable, "-m", "r2.data_files_doc"], cwd=clone, env=env, capture_output=True,
                         text=True)
    assert doc.returncode == 0, doc.stdout + doc.stderr      # before the fix: "no rule for data/v2/manual_edits.json"
    g("add", "-A")
    g("commit", "-qm", "DATA_FILES.md")
    files = (clone / "docs" / "DATA_FILES.md").read_text().split("## Not in this repository")[0]
    assert "Not yet present" not in files and "`analysis_repeated/r2_runs_manifest.json`" in files
    suite = subprocess.run([sys.executable, "-m", "pytest", "tests/r2", "-q", "-p", "no:cacheprovider",
                            "--ignore=tests/r2/test_rehearse.py"], cwd=clone, capture_output=True, text=True,
                           env={**env, "R2_RUNS_V2": str(root / "r2_runs"), "PYTHONDONTWRITEBYTECODE": "1"},
                           timeout=1500)
    assert suite.returncode == 0, suite.stdout[-5000:]
    assert g("status", "--porcelain").stdout.strip() == ""   # the suite wrote nothing into the clone
