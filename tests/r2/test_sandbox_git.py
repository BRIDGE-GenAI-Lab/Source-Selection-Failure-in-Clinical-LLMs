"""Git under R2_SANDBOX never reaches a real repository: inherited GIT_* variables (git exports them inside hooks)
are removed, and every command first checks that the repository it acts on lies inside the sandbox."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from r2 import paths, pipeline, sandbox_git
from r2.run import harness

LOCATION_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES")


def _clean():
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _g(repo, *a):
    return subprocess.run(["git", "-c", "user.name=v", "-c", "user.email=v@x.invalid", "-c", "commit.gpgsign=false",
                           "-c", "tag.gpgsign=false", *a], cwd=repo, env=_clean(), check=True, capture_output=True,
                          text=True).stdout


def _victim(where):
    """A throwaway stand-in for a real repository: one commit, no tags."""
    where.mkdir(parents=True)
    _g(where, "init", "-q")
    (where / "real.txt").write_text("real\n")
    _g(where, "add", "real.txt")
    _g(where, "commit", "-q", "-m", "real")
    return where


def _fingerprint(repo):
    files = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(repo.rglob("*")) if p.is_file()}
    return {"log": _g(repo, "log", "--all", "--format=%H %s"), "tags": _g(repo, "tag", "--list"),
            "status": _g(repo, "status", "--porcelain"), "files": files}


def _point_git_at(monkeypatch, victim):
    gd = victim / ".git"
    values = {"GIT_DIR": gd, "GIT_WORK_TREE": victim, "GIT_INDEX_FILE": gd / "index", "GIT_OBJECT_DIRECTORY":
              gd / "objects", "GIT_ALTERNATE_OBJECT_DIRECTORIES": gd / "objects", "GIT_COMMON_DIR": gd,
              "GIT_CEILING_DIRECTORIES": victim.parent}
    for k, v in values.items():
        monkeypatch.setenv(k, str(v))


@pytest.fixture
def sandboxed(tmp_path, monkeypatch):
    box = tmp_path / "box"
    box.mkdir()
    layout = {"DATA_V2": box / "data" / "v2", "VERIFY_DIR": box / "verification", "RUNS_V2": box / "r2_runs",
              "ANALYSIS_REPEATED": box / "analysis_repeated", "R2_WORKING": box / "npj_R2_WORKING",
              "MODELS_FILE": box / "models_v2.json", "FREEZE_REPO": box / "freeze_repo",
              "PUBLIC_RECORD": box / "PUBLIC_RECORD.json"}
    layout.update({"TABLES_DIR": layout["R2_WORKING"] / "tables", "FIGURES_DIR": layout["R2_WORKING"] / "05_Figures",
                   "FREEZE_RECORD": layout["FREEZE_REPO"] / "docs" / "r2" / "FREEZE_RECORD.md"})
    monkeypatch.setattr(paths, "SANDBOX", box)
    for k, v in layout.items():
        monkeypatch.setattr(paths, k, v)
    monkeypatch.setattr(paths, "WRITE_PATHS", dict(layout))
    monkeypatch.setattr(harness, "SANDBOX", box)
    monkeypatch.setattr(harness, "FREEZE_REPO", layout["FREEZE_REPO"])
    return box


def freeze_inputs(make_pairs):
    """The inputs r2.freeze needs, written to the (sandboxed) r2.paths locations."""
    specs = harness.load_specs(paths.REAL_MODELS_FILE)[:2]
    paths.MODELS_FILE.write_text(json.dumps({"models": specs}))
    paths.DATA_V2.mkdir(parents=True)
    pairs = make_pairs([f"CASE_{i:03d}" for i in range(1, 501)])
    (paths.DATA_V2 / "pairs_v2.json").write_text(json.dumps(pairs))
    sha = hashlib.sha256((paths.DATA_V2 / "pairs_v2.json").read_bytes()).hexdigest()
    paths.VERIFY_DIR.mkdir()
    (paths.VERIFY_DIR / "excluded_pairs.json").write_text(json.dumps({"excluded": [], "pairs_v2_sha256": sha}))
    pilot = paths.RUNS_V2 / "pilot"
    pilot.mkdir(parents=True)
    for s in specs:
        harness.results_path(s, pilot).write_text('{"case_id": "CASE_001"}\n')
    return specs, pairs


# ---------------------------------------------------------------- the environment
def test_the_git_environment_loses_every_location_variable(monkeypatch, tmp_path):
    for k in LOCATION_VARS + ("GIT_CONFIG_PARAMETERS", "GIT_NAMESPACE"):
        monkeypatch.setenv(k, str(tmp_path))
    env = sandbox_git.clean_env()
    assert not [k for k in env if k.startswith("GIT_") and k != "GIT_OPTIONAL_LOCKS"]
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert set(LOCATION_VARS) <= set(sandbox_git.LOCATION_VARS)


def test_document_and_rehearsal_children_get_no_git_variables(sandboxed, monkeypatch, tmp_path):
    from r2 import rehearse
    _point_git_at(monkeypatch, tmp_path / "victim")
    for env in (pipeline._doc_env(), rehearse.child_env(sandboxed)):
        assert not [k for k in env if k in LOCATION_VARS]
        assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert rehearse.child_env(sandboxed)["R2_SANDBOX"] == str(sandboxed)


# ---------------------------------------------------------------- the sandbox freeze
def test_sandbox_freeze_ignores_git_variables_that_point_at_a_real_repository(sandboxed, tmp_path, monkeypatch,
                                                                               make_pairs):
    from r2 import freeze as F
    freeze_inputs(make_pairs)
    victim = _victim(tmp_path / "victim")
    before = _fingerprint(victim)
    _point_git_at(monkeypatch, victim)
    assert pipeline.freeze(dry_run=True)["dry_run"]
    res = pipeline.freeze(yes=True)
    assert res["frozen"]
    monkeypatch.undo()  # the victim is read back with a clean environment
    assert _fingerprint(victim) == before                      # no commit, tag, config or index change
    repo = sandboxed / "freeze_repo"
    assert (repo / ".git").is_dir()                             # its own repository, not a re-initialised victim
    assert F.TAG in _g(repo, "tag", "--list").split()
    assert _g(repo, "rev-parse", "--show-toplevel").strip() == str(repo.resolve())


def test_the_tag_check_under_a_sandbox_reads_only_the_freeze_repository(sandboxed, tmp_path, monkeypatch):
    victim = _victim(tmp_path / "victim")
    _g(victim, "tag", "-a", "t-frozen", "-m", "victim tag")
    repo = _victim(sandboxed / "freeze_repo")
    _point_git_at(monkeypatch, victim)
    assert harness._tag_exists("t-frozen") is False             # the victim's tag is not seen
    monkeypatch.undo()
    monkeypatch.setattr(harness, "SANDBOX", sandboxed)
    monkeypatch.setattr(harness, "FREEZE_REPO", repo)
    _g(repo, "tag", "-a", "t-frozen", "-m", "sandbox tag")
    _point_git_at(monkeypatch, victim)
    assert harness._tag_exists("t-frozen") is True


def test_a_git_dir_outside_the_sandbox_is_refused(sandboxed, tmp_path, make_pairs):
    victim = _victim(tmp_path / "victim")
    before = _fingerprint(victim)
    repo = sandboxed / "freeze_repo"
    repo.mkdir()
    (repo / ".git").write_text(f"gitdir: {victim / '.git'}\n")   # a gitfile pointing out of the sandbox
    with pytest.raises(sandbox_git.SandboxGitError, match="outside the sandbox"):
        sandbox_git.run(repo, "status", sandbox=sandboxed)
    freeze_inputs(make_pairs)
    with pytest.raises(pipeline.Refused, match="outside the sandbox|gitfile"):
        pipeline.freeze(yes=True)
    assert _fingerprint(victim) == before


def test_repository_discovery_that_escapes_the_sandbox_is_refused(tmp_path):
    victim = _victim(tmp_path / "victim")
    box = victim / "box"                                        # a sandbox inside some other work tree
    (box / "freeze_repo").mkdir(parents=True)
    with pytest.raises(sandbox_git.SandboxGitError, match="outside the sandbox"):
        sandbox_git.run(box / "freeze_repo", "status", sandbox=box)
    with pytest.raises(sandbox_git.SandboxGitError, match="outside the sandbox"):
        sandbox_git.run(tmp_path / "elsewhere", "init", "-q", sandbox=box)


def test_init_and_config_failures_are_reported(sandboxed, monkeypatch):
    real_run = subprocess.run

    def failing(cmd, *a, **k):
        if cmd[:1] == ["git"] and "config" in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "config is locked")
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(sandbox_git.subprocess, "run", failing)
    with pytest.raises(sandbox_git.SandboxGitError, match="config is locked"):
        sandbox_git.init_repo(sandboxed / "freeze_repo", sandbox=sandboxed)


# ---------------------------------------------------------------- no git against the code checkout
def test_the_sandbox_harness_commit_runs_no_git(sandboxed, monkeypatch):
    def no_git(cmd, *a, **k):
        raise AssertionError(f"git called under a sandbox: {cmd}")
    monkeypatch.setattr(harness.subprocess, "run", no_git)
    commit = harness._clean_commit()
    assert commit.startswith("sandbox:code-") and len(commit) == len("sandbox:code-") + 16


def test_python_m_r2_freeze_is_refused_under_a_sandbox(tmp_path):
    box = tmp_path / "box"
    env = {**_clean(), "R2_SANDBOX": str(box), "PYTHONDONTWRITEBYTECODE": "1"}
    res = subprocess.run([sys.executable, "-m", "r2.freeze", "--dry-run"], cwd=paths.REPO, env=env,
                         capture_output=True, text=True)
    assert res.returncode != 0
    assert "r2.pipeline freeze" in res.stderr + res.stdout
    assert not box.exists()
