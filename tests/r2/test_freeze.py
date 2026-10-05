import json
import subprocess
from pathlib import Path

import pytest

from r2 import freeze

PLAN = Path(__file__).resolve().parents[2] / "docs" / "r2" / "ANALYSIS_PLAN.md"
HASHED = ["data/v2/pairs_v2.json", "r2/stimuli/render.py", "verification/excluded_pairs.json",
          "r2/run/models_v2.json", "r2/run/harness.py"]


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def env(tmp_path):
    repo = tmp_path / "repo"
    runs = tmp_path / "runs"
    for rel in HASHED + ["docs/r2/ANALYSIS_PLAN.md"]:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / "data/v2/pairs_v2.json").write_text(json.dumps(
        {"unresolved": [], "cases": [{"case_id": f"c{i}"} for i in range(500)]}))
    (repo / "r2/stimuli/render.py").write_text("# render\n")
    (repo / "verification/excluded_pairs.json").write_text(json.dumps({"excluded": []}))
    (repo / "r2/run/models_v2.json").write_text(json.dumps(
        {"models": [{"model_id": "org/m-1", "original_name": "m1"}, {"model_id": "m2", "original_name": "m2"}]}))
    (repo / "r2/run/harness.py").write_text("# harness\n")
    (repo / "docs/r2/ANALYSIS_PLAN.md").write_text(PLAN.read_text())
    (repo / ".gitignore").write_text("._*\n")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "T")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "init")
    pilot = runs / "pilot"
    pilot.mkdir(parents=True)
    for name in ("org_m-1", "m2"):
        (pilot / f"{name}.jsonl").write_text('{"case_id": "c0", "order": "TS"}\n')
    return repo, runs


def run(env, **kw):
    repo, runs = env
    return freeze.freeze(repo=repo, runs_v2=runs, today="2026-10-20", **kw)


def commit_all(repo, msg="change"):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)


def test_missing_excluded_pairs(env):
    repo, _ = env
    git(repo, "rm", "-q", "verification/excluded_pairs.json")
    git(repo, "commit", "-qm", "rm")
    with pytest.raises(freeze.FreezeRefused, match="excluded_pairs.json"):
        run(env)


def test_unresolved_pairs(env):
    repo, _ = env
    (repo / "data/v2/pairs_v2.json").write_text(json.dumps(
        {"unresolved": ["c1"], "cases": [{"case_id": f"c{i}"} for i in range(500)]}))
    commit_all(repo)
    with pytest.raises(freeze.FreezeRefused, match="unresolved"):
        run(env)


def test_wrong_case_count(env):
    repo, _ = env
    (repo / "data/v2/pairs_v2.json").write_text(json.dumps({"unresolved": [], "cases": [{"case_id": "c0"}]}))
    commit_all(repo)
    with pytest.raises(freeze.FreezeRefused, match="500"):
        run(env)


def test_no_pilot_dir(env):
    _, runs = env
    for p in (runs / "pilot").iterdir():
        p.unlink()
    (runs / "pilot").rmdir()
    with pytest.raises(freeze.FreezeRefused, match="pilot"):
        run(env)


def test_pilot_missing_a_model(env):
    _, runs = env
    (runs / "pilot" / "m2.jsonl").write_text("")
    with pytest.raises(freeze.FreezeRefused, match="m2"):
        run(env)


def test_dirty_tree(env):
    repo, _ = env
    (repo / "r2/run/harness.py").write_text("# edited\n")
    with pytest.raises(freeze.FreezeRefused, match="clean"):
        run(env)


def test_untracked_file_is_dirty(env):
    repo, _ = env
    (repo / "stray.txt").write_text("x")
    with pytest.raises(freeze.FreezeRefused, match="clean"):
        run(env)


def test_tag_exists(env):
    repo, _ = env
    git(repo, "tag", "-a", freeze.TAG, "-m", "x")
    with pytest.raises(freeze.FreezeRefused, match="already exists"):
        run(env)


def test_plan_already_frozen(env):
    repo, _ = env
    p = repo / "docs/r2/ANALYSIS_PLAN.md"
    p.write_text(p.read_text().split("\n", 1)[1])
    commit_all(repo)
    with pytest.raises(freeze.FreezeRefused, match="banner"):
        run(env)


def test_dry_run_changes_nothing(env, capsys):
    repo, _ = env
    head = git(repo, "rev-parse", "HEAD")
    run(env, dry_run=True)
    out = capsys.readouterr().out
    assert freeze.TAG in out and "dry run" in out.lower()
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "tag", "-l").strip() == ""
    assert git(repo, "status", "--porcelain").strip() == ""
    assert (repo / "docs/r2/ANALYSIS_PLAN.md").read_text().startswith("DRAFT")


def test_success(env):
    import hashlib
    repo, _ = env
    head = git(repo, "rev-parse", "HEAD").strip()
    run(env)
    plan = (repo / "docs/r2/ANALYSIS_PLAN.md").read_text()
    assert not plan.startswith("DRAFT") and "DRAFT" not in plan.splitlines()[0]
    assert "<sha>" not in plan and "<YYYY-MM-DD>" not in plan
    assert "Date frozen: 2026-10-20." in plan
    for rel in HASHED:
        digest = hashlib.sha256((repo / rel).read_bytes()).hexdigest()
        assert f"- {rel}: {digest}" in plan
    record = (repo / "docs/r2/FREEZE_RECORD.md").read_text()
    assert freeze.TAG in record and head in record and "2026-10-20" in record
    assert "pending" in record.lower() and "push" in record.lower()
    for rel in HASHED:
        assert rel in record
    assert git(repo, "status", "--porcelain").strip() == ""
    assert git(repo, "tag", "-l").strip() == freeze.TAG
    assert git(repo, "cat-file", "-t", freeze.TAG).strip() == "tag"
    assert git(repo, "rev-parse", f"{freeze.TAG}^{{commit}}").strip() == git(repo, "rev-parse", "HEAD").strip()
    assert git(repo, "rev-parse", "HEAD^").strip() == head
    assert set(git(repo, "show", "--name-only", "--format=", "HEAD").split()) == {
        "docs/r2/ANALYSIS_PLAN.md", "docs/r2/FREEZE_RECORD.md"}
    assert "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" in git(repo, "log", "-1", "--format=%B")


def test_freeze_never_pushes():
    assert '"push"' not in Path(freeze.__file__).read_text()


def test_draft_plan_in_repo_has_slots_and_banner():
    text = PLAN.read_text()
    assert text.startswith("DRAFT — not frozen")
    assert text.count("<sha>") == 5 and "<YYYY-MM-DD>" in text
    assert "S5." in text and "Gemini-2.5-Flash-Lite" in text
    assert "Wilson 95%" in text and "not assessable" in text


def test_tag_failure_rolls_back(env, monkeypatch):
    repo, _ = env
    head = git(repo, "rev-parse", "HEAD").strip()
    plan_before = (repo / "docs/r2/ANALYSIS_PLAN.md").read_bytes()
    real_git = freeze._git

    def failing(r, *args, **kw):
        if args and args[0] == "tag":
            raise freeze.FreezeRefused("git tag failed: simulated")
        return real_git(r, *args, **kw)

    monkeypatch.setattr(freeze, "_git", failing)
    with pytest.raises(freeze.FreezeRefused, match="rolled back"):
        run(env)
    assert git(repo, "rev-parse", "HEAD").strip() == head
    assert git(repo, "tag", "-l").strip() == ""
    assert (repo / "docs/r2/ANALYSIS_PLAN.md").read_bytes() == plan_before
    assert not (repo / "docs/r2/FREEZE_RECORD.md").exists()
    assert git(repo, "status", "--porcelain").strip() == ""
