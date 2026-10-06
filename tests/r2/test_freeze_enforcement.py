"""Final fix I1: the harness itself enforces the freeze in main mode (tag, the SHA-256 of every frozen file against
FREEZE_RECORD.md at the tag, the public timestamp), so a direct `python -m r2.run.harness` call is guarded too;
record-public rejects a date before the freeze commit's date or after today."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from r2 import freeze as F
from r2 import paths, pipeline
from r2.run import harness
from tests.r2.test_pipeline import _frozen, sandboxed  # noqa: F401  (sandboxed is a fixture)

REAL = Path(paths.REPO)
TAG_URL = "https://github.com/example/r2/releases/tag/r2-analysis-plan-frozen"
FREEZE_DAY = "2026-10-06"
GIT_ID = ["-c", "user.name=T", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
          "-c", "tag.gpgsign=false"]


def git(repo, *args):
    return subprocess.run(["git", *GIT_ID, *args], cwd=repo, capture_output=True, text=True, check=True).stdout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def commit_all(repo, msg="change"):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)


def add_public(repo, since=FREEZE_DAY):
    rec = repo / "docs/r2/FREEZE_RECORD.md"
    text = rec.read_text().replace(
        "Public timestamp: pending a push.",
        f"Public timestamp: recorded.\n\n- Public URL: {TAG_URL}\n- Kind: pushed tag\n- Public since: {since}\n"
        "- Tag commit: abc\n\nWas pending a push.", 1)
    rec.write_text(text)


def add_change(repo, rel, digest, day=FREEZE_DAY, reason="bug fix in the retry loop"):
    rec = repo / "docs/r2/FREEZE_RECORD.md"
    text = rec.read_text()
    assert text.rstrip().endswith("- none")
    rec.write_text(text.rstrip()[:-len("- none")] + f"- {day}: {rel} sha256 {digest}: {reason}\n")


@pytest.fixture
def frozen(tmp_path, monkeypatch, make_pairs):
    """A throwaway repository holding the five frozen inputs, FREEZE_RECORD.md written by r2.freeze, an annotated
    freeze tag, and the harness pointed at it (REPO, FREEZE_REPO and the state paths)."""
    repo = tmp_path / "repo"
    pairs = make_pairs([f"CASE_{i:03d}" for i in range(1, 501)])
    files = {
        "data/v2/pairs_v2.json": json.dumps(pairs),
        "r2/stimuli/render.py": (REAL / "r2/stimuli/render.py").read_text(),
        "r2/run/harness.py": (REAL / "r2/run/harness.py").read_text(),
        "r2/run/models_v2.json": json.dumps({"models": harness.load_specs(paths.REAL_MODELS_FILE)[:1]}),
        "tests/placeholder.txt": "x\n",
    }
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    (repo / "verification").mkdir()
    (repo / "verification/excluded_pairs.json").write_text(json.dumps(
        {"excluded": [], "pairs_v2_sha256": sha(repo / "data/v2/pairs_v2.json")}))
    git(repo, "init", "-q", "-b", "main")
    commit_all(repo, "inputs")
    base = git(repo, "rev-parse", "HEAD").strip()
    hashes = {rel: sha(repo / rel) for rel in F.HASHED}
    (repo / "docs/r2").mkdir(parents=True)
    (repo / F.RECORD).write_text(F.freeze_record(base, hashes, FREEZE_DAY))
    commit_all(repo, "freeze")
    git(repo, "tag", "-a", F.TAG, "-m", "frozen")
    for name, value in {"REPO": repo, "FREEZE_REPO": repo, "DATA_V2": repo / "data/v2",
                        "VERIFY_DIR": repo / "verification", "MODELS_FILE": repo / "r2/run/models_v2.json",
                        "SANDBOX": None}.items():
        monkeypatch.setattr(harness, name, value)
    monkeypatch.setattr(harness, "_today", lambda: "2026-10-08", raising=False)
    return repo


def run_main(repo, *extra):
    out = repo.parent / "runs" / "main"
    return harness.main(["--limit", "0", "--out", str(out), *extra], client_factory=lambda provider: None), out


def test_main_runs_when_every_frozen_input_matches_and_the_freeze_is_public(frozen):
    add_public(frozen)
    commit_all(frozen, "public")
    summary, out = run_main(frozen)
    assert summary == {"stopped": [], "need_resume": 0}
    assert not (out / harness.FREEZE_INCIDENTS).exists()


def test_a_committed_post_freeze_change_to_models_v2_is_refused(frozen):
    """The reviewer's scenario: temperature 0 -> 0.7 and max_tokens 2048 -> 4096 committed after the freeze."""
    add_public(frozen)
    models = frozen / "r2/run/models_v2.json"
    doc = json.loads(models.read_text())
    doc["models"][0].update(temperature=0.7, max_tokens=4096)
    models.write_text(json.dumps(doc))
    commit_all(frozen, "post-freeze change")
    with pytest.raises(SystemExit, match=r"refusing the main run: r2/run/models_v2.json has sha256 [0-9a-f]{64}, "
                                         r"but FREEZE_RECORD.md at the tag records [0-9a-f]{64}"):
        run_main(frozen)
    assert not (frozen.parent / "runs" / "main").exists()  # refused before anything was written


@pytest.mark.parametrize("rel", F.HASHED)
def test_a_change_to_any_frozen_input_is_refused(frozen, rel):
    add_public(frozen)
    path = frozen / rel
    path.write_bytes(path.read_bytes() + b"\n")
    if rel == "data/v2/pairs_v2.json":  # keep the verified-stimulus check consistent: the hash check must catch it
        ex = frozen / "verification/excluded_pairs.json"
        ex.write_text(json.dumps({"excluded": [], "pairs_v2_sha256": sha(path)}))
    commit_all(frozen)
    with pytest.raises(SystemExit, match=f"refusing the main run: {rel} has sha256"):
        run_main(frozen)


def test_a_dated_change_entry_accepts_the_new_hash_and_logs_an_incident(frozen, capsys):
    add_public(frozen)
    models = frozen / "r2/run/models_v2.json"
    doc = json.loads(models.read_text())
    doc["models"][0]["max_tokens"] = 4096
    models.write_text(json.dumps(doc))
    add_change(frozen, "r2/run/models_v2.json", sha(models), day="2026-10-07", reason="truncations in the pilot")
    commit_all(frozen, "documented change")
    summary, out = run_main(frozen)
    assert summary["stopped"] == []
    (line,) = (out / harness.FREEZE_INCIDENTS).read_text().splitlines()
    inc = json.loads(line)
    assert inc["kind"] == "frozen_input_changed" and inc["path"] == "r2/run/models_v2.json"
    assert inc["current_sha256"] == sha(models) and inc["entry_date"] == "2026-10-07"
    assert inc["reason"] == "truncations in the pilot" and inc["frozen_sha256"] != inc["current_sha256"]
    assert "FROZEN INPUT CHANGED r2/run/models_v2.json" in capsys.readouterr().out


@pytest.mark.parametrize("day,match", [("2026-10-01", "before the freeze"), ("2026-10-09", "after today")])
def test_a_change_entry_dated_outside_the_freeze_and_today_is_refused(frozen, day, match):
    add_public(frozen)
    models = frozen / "r2/run/models_v2.json"
    models.write_text(models.read_text() + "\n")
    add_change(frozen, "r2/run/models_v2.json", sha(models), day=day)
    commit_all(frozen)
    with pytest.raises(SystemExit, match=match):
        run_main(frozen)


def test_editing_the_frozen_hashes_in_the_record_is_refused(frozen):
    """The frozen hashes are read from FREEZE_RECORD.md at the tag; a working copy whose hash lines were rewritten
    to match a changed file is refused."""
    add_public(frozen)
    models = frozen / "r2/run/models_v2.json"
    old = sha(models)
    models.write_text(models.read_text() + "\n")
    rec = frozen / F.RECORD
    rec.write_text(rec.read_text().replace(old, sha(models)))
    commit_all(frozen)
    with pytest.raises(SystemExit, match="its hash lines differ from FREEZE_RECORD.md at the tag"):
        run_main(frozen)


def test_main_refuses_without_a_public_timestamp(frozen):
    with pytest.raises(SystemExit, match="records no public timestamp of the freeze"):
        run_main(frozen)


def test_accept_unpublished_freeze_runs_and_logs_an_incident(frozen, capsys):
    summary, out = run_main(frozen, "--accept-unpublished-freeze")
    assert summary["stopped"] == []
    (line,) = (out / harness.FREEZE_INCIDENTS).read_text().splitlines()
    assert json.loads(line)["kind"] == "unpublished_freeze"
    assert "ACCEPTED: main run without a public timestamp" in capsys.readouterr().out


def test_an_uncommitted_freeze_record_is_refused(frozen):
    add_public(frozen)  # recorded, not committed
    with pytest.raises(SystemExit, match="docs/r2/FREEZE_RECORD.md"):
        run_main(frozen)


def test_a_freeze_record_missing_at_the_tag_is_refused(frozen):
    git(frozen, "tag", "-d", F.TAG)
    git(frozen, "tag", "-a", F.TAG, "-m", "x", "HEAD~1")  # the tag on a commit without the record
    add_public(frozen)
    commit_all(frozen)
    with pytest.raises(SystemExit, match="no docs/r2/FREEZE_RECORD.md at the tag"):
        run_main(frozen)


def test_pilot_mode_skips_the_freeze_checks(frozen):
    models = frozen / "r2/run/models_v2.json"
    models.write_text(models.read_text() + "\n")
    commit_all(frozen)
    out = frozen.parent / "runs" / "pilot"
    assert harness.main(["--pilot", "--limit", "0", "--out", str(out)],
                        client_factory=lambda provider: None)["stopped"] == []


def test_the_harness_accepts_the_unpublished_freeze_flag():
    assert harness.parse_args(["--accept-unpublished-freeze"]).accept_unpublished_freeze is True
    assert harness.parse_args([]).accept_unpublished_freeze is False


def test_the_frozen_sources_are_the_hashed_files():
    assert set(harness.frozen_sources()) == set(F.HASHED)


# ---------------------------------------------------------------- the record parser (r2.freeze)
def test_record_parser_reads_hashes_changes_and_the_public_timestamp():
    hashes = {rel: c * 64 for rel, c in zip(F.HASHED, "abcde")}
    text = F.freeze_record("0" * 40, hashes, FREEZE_DAY)
    assert F.frozen_hashes(text) == hashes
    assert F.change_entries(text) == []
    assert F.public_timestamp(text) is None
    assert F.record_date(text) == FREEZE_DAY
    text = text.replace("Public timestamp: pending a push.",
                        f"- Public URL: {TAG_URL}\n- Kind: pushed tag\n- Public since: 2026-10-07\n- Tag commit: x\n")
    text = text.rstrip()[:-len("- none")] + f"- 2026-10-07: r2/run/harness.py sha256 {'f' * 64}: fix\n"
    assert F.public_timestamp(text) == {"url": TAG_URL, "kind": "pushed tag", "since": "2026-10-07",
                                        "tag_commit": "x"}
    assert F.change_entries(text) == [{"date": "2026-10-07", "path": "r2/run/harness.py", "sha256": "f" * 64,
                                       "reason": "fix"}]


def test_record_parser_refuses_a_missing_or_duplicated_hash():
    hashes = {rel: c * 64 for rel, c in zip(F.HASHED, "abcde")}
    text = F.freeze_record("0" * 40, hashes, FREEZE_DAY)
    with pytest.raises(F.FreezeRefused, match="r2/run/harness.py"):
        F.frozen_hashes(text.replace(f"- r2/run/harness.py: {'e' * 64}\n", ""))
    with pytest.raises(F.FreezeRefused, match="more than once"):
        F.frozen_hashes(text + f"\n- r2/run/harness.py: {'9' * 64}\n")


def test_the_record_explains_how_to_document_a_change():
    text = F.freeze_record("0" * 40, {rel: "a" * 64 for rel in F.HASHED}, FREEZE_DAY)
    assert "- YYYY-MM-DD: <path> sha256 <new SHA-256>: <reason>" in text
    assert "refuses" in text


# ---------------------------------------------------------------- the pipeline: refusals with exit status 2
def _sandbox_main_world(make_pairs):
    specs = _frozen(make_pairs)
    pairs = json.loads((paths.DATA_V2 / "pairs_v2.json").read_text())
    from r2 import rehearse
    return specs, rehearse.FakeAPI({c["case_id"]: c for c in pairs["cases"]}, specs)


def test_pipeline_main_refuses_a_changed_frozen_input(sandboxed, make_pairs, capsys):
    from r2 import rehearse
    specs, api = _sandbox_main_world(make_pairs)
    pipeline.record_public(TAG_URL)
    doc = json.loads(paths.MODELS_FILE.read_text())
    doc["models"][0]["temperature"] = 0.7
    paths.MODELS_FILE.write_text(json.dumps(doc))
    with pytest.raises(pipeline.Refused, match="r2/run/models_v2.json has sha256"):
        pipeline.run_harness("main", yes=True, client_factory=api.factory, sleep=rehearse.Sleeper(),
                             models=specs[0]["original_name"])
    assert not (paths.RUNS_V2 / "main").exists()


def test_pipeline_main_passes_the_override_to_the_harness_which_logs_one_incident(sandboxed, make_pairs, capsys):
    from r2 import rehearse
    specs, api = _sandbox_main_world(make_pairs)
    assert "--accept-unpublished-freeze" in pipeline.harness_argv("main", accept_unpublished_freeze=True)
    rc = pipeline.run_harness("main", yes=True, client_factory=api.factory, sleep=rehearse.Sleeper(),
                              models=specs[0]["original_name"], accept_unpublished_freeze=True)
    assert rc in (0, 1)
    lines = (paths.RUNS_V2 / "main" / harness.FREEZE_INCIDENTS).read_text().splitlines()
    assert [json.loads(x)["kind"] for x in lines] == ["unpublished_freeze"]  # once, by the harness
    assert "--accept-unpublished-freeze" in capsys.readouterr().out


# ---------------------------------------------------------------- record-public and the freeze commit's date
def test_record_public_rejects_a_date_before_the_freeze_commit(sandboxed, make_pairs):
    _frozen(make_pairs)
    rec = paths.FREEZE_RECORD
    rec.write_text(rec.read_text().replace(f"- Date: {pipeline._today()}", "- Date: 2000-01-01"))
    with pytest.raises(pipeline.Refused, match="before the freeze commit"):
        pipeline.record_public(TAG_URL, date="2000-06-01")  # after the record's Date, before the commit
    with pytest.raises(pipeline.Refused, match="after today"):
        pipeline.record_public(TAG_URL, date="2999-01-01")
    assert pipeline.record_public(TAG_URL)["since"] == pipeline._today()

