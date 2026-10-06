"""Final fix I3: python -m r2.pipeline package-runs writes analysis_repeated/r2_runs_manifest.json (SHA-256 and size
of every run-log file, with totals) and a deterministic tarball of r2_runs/ for the Zenodo deposit."""
import fcntl
import gzip
import hashlib
import json
import tarfile

import pytest

from r2 import paths, pipeline
from tests.r2.test_pipeline import sandboxed  # noqa: F401  (a fixture)

FILES = {
    "main/gpt-4.1.jsonl": '{"case_id": "C1", "order": "TS"}\n{"case_id": "C1", "order": "ST"}\n',
    "main/gpt-4.1.incidents.jsonl": '{"kind": "stopped"}\n',
    "main/freeze.incidents.jsonl": '{"kind": "unpublished_freeze"}\n',
    "main/_superseded/old-model.jsonl": '{"case_id": "C1", "order": "TS"}\n',
    "main/_superseded/old-model.incidents.jsonl": '{"kind": "model_mismatch"}\n',
    "pilot/gpt-4.1.jsonl": '{"case_id": "C1", "order": "TS"}\n',
    "logs/openai.log": "FINISHED 1 model(s); stopped 0: []; 0 cells need a resume\n",
    "model_availability.csv": "original_name,status\ngpt-4.1,available\n",
}
SKIPPED = {"main/gpt-4.1.lock": "", "main/._gpt-4.1.jsonl": "appledouble", ".DS_Store": "x",
           "pilot/gpt-4.1.lock": ""}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def runs(sandboxed):
    for rel, text in {**FILES, **SKIPPED}.items():
        p = paths.RUNS_V2 / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return paths.RUNS_V2


def test_manifest_lists_every_run_log_file_with_sha256_size_and_totals(runs):
    res = pipeline.package_runs()
    man = json.loads((paths.ANALYSIS_REPEATED / pipeline.RUNS_MANIFEST).read_text())
    assert res["files"] == len(FILES) and man["synthetic"] is True
    assert [f["path"] for f in man["files"]] == sorted(FILES)
    for f in man["files"]:
        data = FILES[f["path"]].encode()
        assert f["sha256"] == _sha(data) and f["bytes"] == len(data)
    assert man["totals"]["files"] == len(FILES)
    assert man["totals"]["bytes"] == sum(len(t.encode()) for t in FILES.values())
    assert man["totals"]["by_directory"]["main"]["files"] == 5
    assert man["totals"]["by_directory"]["pilot"]["files"] == 1
    assert man["totals"]["by_directory"]["."]["files"] == 1
    assert set(man["excluded_patterns"]) == {"*.lock", "._*", ".DS_Store"}


def test_tarball_holds_exactly_the_manifest_files_and_its_hash_is_recorded(runs):
    pipeline.package_runs()
    man = json.loads((paths.ANALYSIS_REPEATED / pipeline.RUNS_MANIFEST).read_text())
    tar_path = paths.RUNS_ARCHIVE / pipeline.RUNS_TARBALL
    assert man["tarball"]["name"] == pipeline.RUNS_TARBALL
    assert man["tarball"]["sha256"] == _sha(tar_path.read_bytes())
    assert man["tarball"]["bytes"] == tar_path.stat().st_size
    with tarfile.open(tar_path, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        assert sorted(members) == sorted(f"r2_runs/{p}" for p in FILES)
        for p, text in FILES.items():
            assert tar.extractfile(members[f"r2_runs/{p}"]).read() == text.encode()
            assert members[f"r2_runs/{p}"].uid == 0 and members[f"r2_runs/{p}"].uname == ""


def test_package_runs_is_deterministic_and_idempotent(runs):
    pipeline.package_runs()
    tar1 = (paths.RUNS_ARCHIVE / pipeline.RUNS_TARBALL).read_bytes()
    man1 = (paths.ANALYSIS_REPEATED / pipeline.RUNS_MANIFEST).read_bytes()
    pipeline.package_runs()
    assert (paths.RUNS_ARCHIVE / pipeline.RUNS_TARBALL).read_bytes() == tar1
    assert (paths.ANALYSIS_REPEATED / pipeline.RUNS_MANIFEST).read_bytes() == man1
    assert gzip.open(paths.RUNS_ARCHIVE / pipeline.RUNS_TARBALL).read()  # a valid gzip stream


def test_check_runs_manifest_reports_a_changed_added_or_removed_file(runs):
    pipeline.package_runs()
    assert pipeline.check_runs_manifest() == []
    (runs / "main" / "gpt-4.1.jsonl").write_text("changed\n")
    (runs / "main" / "new-model.jsonl").write_text("{}\n")
    (runs / "pilot" / "gpt-4.1.jsonl").unlink()
    problems = pipeline.check_runs_manifest()
    assert any("main/gpt-4.1.jsonl" in p and "changed" in p for p in problems)
    assert any("main/new-model.jsonl" in p and "not in the manifest" in p for p in problems)
    assert any("pilot/gpt-4.1.jsonl" in p and "missing" in p for p in problems)


def test_check_runs_manifest_reports_a_changed_tarball(runs):
    pipeline.package_runs()
    tar_path = paths.RUNS_ARCHIVE / pipeline.RUNS_TARBALL
    tar_path.write_bytes(tar_path.read_bytes() + b"x")
    assert any("tarball" in p for p in pipeline.check_runs_manifest())


def test_package_runs_refuses_while_a_harness_holds_a_model_lock(runs):
    with open(runs / "main" / "gpt-4.1.lock", "a") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(pipeline.Refused, match="gpt-4.1.lock is held"):
            pipeline.package_runs()
    assert not (paths.ANALYSIS_REPEATED / pipeline.RUNS_MANIFEST).exists()


def test_package_runs_refuses_a_plaintext_key_and_never_prints_it(runs, capsys):
    key = "tgp" + "_v1_" + "Q" * 30
    (runs / "logs" / "together.log").write_text(f"AuthenticationError: invalid key {key}\n")
    with pytest.raises(pipeline.Refused, match=r"logs/together.log.*Together") as exc:
        pipeline.package_runs()
    assert key not in str(exc.value) and key not in capsys.readouterr().out
    assert not (paths.RUNS_ARCHIVE / pipeline.RUNS_TARBALL).exists()


def test_package_runs_refuses_without_a_main_run(sandboxed):
    with pytest.raises(pipeline.Refused, match="missing input"):
        pipeline.package_runs()


def test_package_runs_is_a_cli_command(runs, capsys):
    assert pipeline.main(["package-runs"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("packaged 8 run-log files")
    out = json.loads(text[text.index("{"):])
    assert out["files"] == len(FILES) and out["manifest"].endswith(pipeline.RUNS_MANIFEST)


# ---------------------------------------------------------------- release-check: the release QA before any tag
@pytest.fixture
def release_world(monkeypatch):
    """Every release check passing, each replaceable by a test."""
    from r2 import data_files_doc
    calls = []
    monkeypatch.setattr(paths, "SANDBOX", None)
    monkeypatch.setattr(pipeline, "_git", lambda repo, *a, **k: type("R", (), {"stdout": "", "returncode": 0})())
    monkeypatch.setattr(data_files_doc, "is_current", lambda: True)
    monkeypatch.setattr(pipeline, "_pending_data_files", lambda: [])
    monkeypatch.setattr(pipeline, "check_runs_manifest", lambda: [])
    monkeypatch.setattr(pipeline, "_numbers_test", lambda: calls.append("numbers") or [])
    monkeypatch.setattr(pipeline, "run_qa", lambda: calls.append("qa") or {})
    monkeypatch.setattr(pipeline, "_suite", lambda: calls.append("suite") or [])
    return calls


def test_release_check_passes_and_prints_one_line_per_check(release_world, capsys):
    res = pipeline.release_check()
    assert all(res.values()) and list(res) == list(pipeline.RELEASE_CHECKS)
    out = capsys.readouterr().out
    for name in pipeline.RELEASE_CHECKS:
        assert f"RELEASE CHECK {name}: PASS" in out
    assert release_world == ["numbers", "qa", "suite"]


@pytest.mark.parametrize("broken,match", [
    ("tree", "clean tree"), ("data_files", "DATA_FILES.md current"), ("pending", "DATA_FILES.md complete"),
    ("manifest", "run-log manifest"), ("numbers", "numbers.json"), ("qa", "QA scripts"), ("suite", "test suite")])
def test_release_check_fails_on_any_check_and_runs_all_of_them(release_world, monkeypatch, capsys, broken, match):
    from r2 import data_files_doc
    if broken == "tree":
        monkeypatch.setattr(pipeline, "_git", lambda repo, *a, **k: type("R", (), {"stdout": " M x\n"})())
    elif broken == "data_files":
        monkeypatch.setattr(data_files_doc, "is_current", lambda: False)
    elif broken == "pending":
        monkeypatch.setattr(pipeline, "_pending_data_files", lambda: ["verification/flags.json"])
    elif broken == "manifest":
        monkeypatch.setattr(pipeline, "check_runs_manifest", lambda: ["main/x.jsonl: changed"])
    elif broken == "numbers":
        monkeypatch.setattr(pipeline, "_numbers_test", lambda: ["edits/test_numbers.py exited 1"])
    elif broken == "qa":
        def qa():
            raise pipeline.Refused("QA failed: qa_xref.py: Cross-reference failures: 2")
        monkeypatch.setattr(pipeline, "run_qa", qa)
    else:
        monkeypatch.setattr(pipeline, "_suite", lambda: ["pytest exited 1"])
    with pytest.raises(pipeline.Refused, match=f"release check failed: {match}"):
        pipeline.release_check()
    out = capsys.readouterr().out
    assert f"RELEASE CHECK {match}: FAIL" in out
    assert out.count("RELEASE CHECK ") == len(pipeline.RELEASE_CHECKS)  # every check runs and is reported


def test_release_check_refuses_under_a_sandbox(sandboxed):
    with pytest.raises(pipeline.Refused, match="without R2_SANDBOX"):
        pipeline.release_check()


def test_release_check_is_a_cli_command(release_world, capsys):
    assert pipeline.main(["release-check", "--no-suite"]) == 0
    assert "suite" not in release_world and "RELEASE CHECK test suite: SKIPPED" in capsys.readouterr().out
