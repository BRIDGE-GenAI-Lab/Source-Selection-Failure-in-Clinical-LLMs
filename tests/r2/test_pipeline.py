"""r2.pipeline: refusals, the --yes guard, sandbox write checks, the sandbox freeze, the public record, the
harness client factory under R2_SANDBOX, the fake API of r2.rehearse, and marker classification."""
import json
import subprocess
from pathlib import Path

import pytest

from r2 import paths, pipeline, rehearse
from r2.run import harness

REAL_DOCS = Path(paths.REAL_R2_WORKING)


@pytest.fixture
def sandboxed(tmp_path, monkeypatch):
    """r2.paths as R2_SANDBOX=<tmp>/box would set it, for the functions that read paths at call time."""
    box = tmp_path / "box"
    box.mkdir()
    layout = {"DATA_V2": box / "data" / "v2", "VERIFY_DIR": box / "verification", "RUNS_V2": box / "r2_runs",
              "ANALYSIS_REPEATED": box / "analysis_repeated", "R2_WORKING": box / "npj_R2_WORKING",
              "MODELS_FILE": box / "models_v2.json", "FREEZE_REPO": box / "freeze_repo",
              "PUBLIC_RECORD": box / "PUBLIC_RECORD.json", "RUNS_ARCHIVE": box / "r2_runs_archive"}
    layout.update({"TABLES_DIR": layout["R2_WORKING"] / "tables", "FIGURES_DIR": layout["R2_WORKING"] / "05_Figures",
                   "FREEZE_RECORD": layout["FREEZE_REPO"] / "docs" / "r2" / "FREEZE_RECORD.md"})
    monkeypatch.setattr(paths, "SANDBOX", box)
    for k, v in layout.items():
        monkeypatch.setattr(paths, k, v)
    monkeypatch.setattr(paths, "WRITE_PATHS", dict(layout))
    # the harness binds its paths at import (in a real sandbox run the whole process is sandboxed)
    for k in ("SANDBOX", "FREEZE_REPO", "DATA_V2", "VERIFY_DIR", "RUNS_V2", "MODELS_FILE"):
        monkeypatch.setattr(harness, k, box if k == "SANDBOX" else layout[k])
    load = harness.load_v2_cases
    monkeypatch.setattr(harness, "load_v2_cases", lambda path=None: load(path or layout["DATA_V2"] / "pairs_v2.json"))
    return box


# ---------------------------------------------------------------- the --yes guard and the printed command
def test_paid_steps_print_the_command_and_need_yes(capsys):
    with pytest.raises(pipeline.Refused, match="--yes"):
        pipeline.run_harness("pilot", yes=False)
    assert "command: python -m r2.run.harness --pilot --limit 10 --models all" in capsys.readouterr().out
    with pytest.raises(pipeline.Refused, match="--yes"):
        pipeline.run_harness("main", yes=False, models="gpt-4.1-2025-04-14", retry_persistent=True)
    assert "command: python -m r2.run.harness --models gpt-4.1-2025-04-14 --retry-persistent" \
        in capsys.readouterr().out
    with pytest.raises(pipeline.Refused, match="--yes"):
        pipeline.probe(yes=False)
    assert "command: python -m r2.run.probe_models" in capsys.readouterr().out


def test_cli_exit_status_2_and_a_clear_message_on_refusal(capsys):
    assert pipeline.main(["pilot"]) == 2
    assert "REFUSED:" in capsys.readouterr().err


def test_freeze_without_dry_run_needs_yes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "FREEZE_REPO", tmp_path)  # no repository: no tag lookup
    with pytest.raises(pipeline.Refused, match="--dry-run first"):
        pipeline.freeze()
    assert "command: python -m r2.freeze" in capsys.readouterr().out


# ---------------------------------------------------------------- sandbox guards
def test_a_sandbox_never_calls_a_real_api(sandboxed):
    with pytest.raises(pipeline.Refused, match="never calls a real API"):
        pipeline.run_harness("pilot", yes=True)
    with pytest.raises(pipeline.Refused, match="never calls a real API"):
        pipeline.probe(yes=True)


def test_every_write_location_must_be_inside_the_sandbox(sandboxed, tmp_path, monkeypatch):
    assert pipeline.check_write_paths() == sandboxed.resolve()
    monkeypatch.setitem(paths.WRITE_PATHS, "RUNS_V2", tmp_path / "elsewhere")
    with pytest.raises(pipeline.Refused, match="outside the sandbox"):
        pipeline.check_write_paths()


def test_a_sandbox_inside_the_repository_is_refused(monkeypatch):
    monkeypatch.setattr(paths, "SANDBOX", Path(paths.REPO) / "box")
    with pytest.raises(ValueError, match="not a sandbox"):
        pipeline.check_write_paths()


@pytest.mark.parametrize("call", [
    lambda: pipeline.import_edits("/nonexistent/AG.xlsx"),
    lambda: pipeline.consensus("/n/a1.xlsx", "/n/m1.xlsx", "/n/a2.xlsx", "/n/m2.xlsx"),
    lambda: pipeline.post_results(render=False),
    lambda: pipeline.freeze(dry_run=True),
], ids=["import-edits", "consensus", "post-results", "freeze"])
def test_steps_refuse_missing_inputs(sandboxed, call):
    with pytest.raises(pipeline.Refused, match="missing input"):
        call()


def test_sandbox_init_refuses_a_non_empty_directory(tmp_path):
    (tmp_path / "x").write_text("x")
    with pytest.raises(pipeline.Refused, match="not empty"):
        pipeline.sandbox_init(tmp_path)


# ---------------------------------------------------------------- record
def test_record_writes_the_release_facts_and_refuses_bad_or_conflicting_values(sandboxed, make_pairs):
    with pytest.raises(pipeline.Refused, match="no tag r2-final"):
        pipeline.record(release_tag="r2-final")                     # the release tag must exist
    _frozen(make_pairs)
    pipeline._git(paths.FREEZE_REPO, *pipeline.GIT_ID, "tag", "r2-final", check=True)
    data = pipeline.record(release_tag="r2-final", zenodo_doi="10.5281/zenodo.1234567")
    assert data == {"release_tag": "r2-final", "zenodo_new_doi": "10.5281/zenodo.1234567", "synthetic": True}
    assert json.loads(paths.PUBLIC_RECORD.read_text())["release_tag"] == "r2-final"
    with pytest.raises(pipeline.Refused, match="DOI"):
        pipeline.record(zenodo_doi="zenodo 123")
    with pytest.raises(pipeline.Refused, match="different values"):
        pipeline.record(zenodo_doi="10.5281/zenodo.7654321")
    with pytest.raises(pipeline.Refused, match="nothing to record"):
        pipeline.record()
    with pytest.raises(SystemExit):                                  # the push date moved to record-public
        pipeline.main(["record", "--freeze-pushed", "2026-10-20"])


# ---------------------------------------------------------------- sandbox freeze
def _git_tags(repo):
    return subprocess.run(["git", "tag", "--list"], cwd=repo, capture_output=True, text=True).stdout.split()


def test_sandbox_freeze_runs_r2_freeze_in_a_throwaway_repository(sandboxed, make_pairs):
    from r2 import freeze as F
    specs = harness.load_specs(paths.REAL_MODELS_FILE)[:2]
    paths.MODELS_FILE.write_text(json.dumps({"models": specs}))
    paths.DATA_V2.mkdir(parents=True)
    pairs = make_pairs([f"CASE_{i:03d}" for i in range(1, 501)])
    (paths.DATA_V2 / "pairs_v2.json").write_text(json.dumps(pairs))
    paths.VERIFY_DIR.mkdir()
    (paths.VERIFY_DIR / "excluded_pairs.json").write_text(json.dumps({"excluded": [], "pairs_v2_sha256": "x"}))
    pilot = paths.RUNS_V2 / "pilot"
    pilot.mkdir(parents=True)
    for s in specs:
        harness.results_path(s, pilot).write_text('{"case_id": "CASE_001"}\n')
    real_tags = _git_tags(paths.REPO)
    res = pipeline.freeze(dry_run=True)
    assert res["dry_run"] and not res["frozen"] and F.TAG not in _git_tags(paths.FREEZE_REPO)
    res = pipeline.freeze(yes=True)
    assert res["frozen"] and F.TAG in _git_tags(paths.FREEZE_REPO)
    assert paths.FREEZE_RECORD.exists()
    assert harness._tag_exists(F.TAG, repo=paths.FREEZE_REPO)
    assert pipeline.freeze(dry_run=True)["already"] is True  # idempotent: a freeze is final
    assert _git_tags(paths.REPO) == real_tags                 # the real repository is untouched


# ---------------------------------------------------------------- the harness client factory under R2_SANDBOX
def _small_world(make_pairs, n=3):
    pairs = make_pairs([f"CASE_{i:03d}" for i in range(1, n + 1)])
    return {c["case_id"]: c for c in pairs["cases"]}


def test_harness_under_sandbox_needs_an_injected_client(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "SANDBOX", tmp_path)
    with pytest.raises(SystemExit, match="no client was injected"):
        harness.main(["--pilot", "--out", str(tmp_path / "pilot")])


def test_harness_under_sandbox_marks_every_record_synthetic(tmp_path, monkeypatch, make_pairs):
    cases = _small_world(make_pairs)
    spec = harness.load_specs(paths.REAL_MODELS_FILE)[0]
    monkeypatch.setattr(harness, "SANDBOX", tmp_path)
    monkeypatch.setattr(harness, "load_v2_cases", lambda *a, **k: (cases, "sha-x"))
    monkeypatch.setattr(harness, "load_specs", lambda *a, **k: [spec])
    api = rehearse.FakeAPI(cases, [spec])
    out = tmp_path / "pilot"
    summary = harness.main(["--pilot", "--out", str(out)], client_factory=api.factory, sleep=rehearse.Sleeper())
    assert summary["stopped"] == []
    recs = harness.read_records(harness.results_path(spec, out))
    assert len(recs) == 6 and all(r["synthetic"] is True for r in recs)
    assert all(r["harness_commit"].startswith("sandbox:") for r in recs)


def test_tag_check_reads_the_freeze_repository(tmp_path, monkeypatch):
    def g(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@x.invalid", "-c", "tag.gpgsign=false",
                        "-c", "commit.gpgsign=false", *a], cwd=tmp_path, check=True, capture_output=True)
    g("init", "-q")
    (tmp_path / "f").write_text("x")
    g("add", "f")
    g("commit", "-q", "-m", "x")
    g("tag", "-a", "t-frozen", "-m", "x")
    monkeypatch.setattr(harness, "FREEZE_REPO", tmp_path)
    assert harness._tag_exists("t-frozen") and not harness._tag_exists("t-other")


# ---------------------------------------------------------------- the fake API
class _Forced(rehearse.FakeAPI):
    P_400, P_429, P_INVALID, P_INVALID_AGAIN, P_THINK_OPEN = 0.0, 1.0, 0.0, 0.0, 0.0


class _Invalid(rehearse.FakeAPI):
    P_400, P_429, P_INVALID, P_INVALID_AGAIN, P_THINK_OPEN = 0.0, 0.0, 1.0, 1.0, 0.0


def test_fake_api_429_with_retry_after_is_retried_and_the_delay_recorded(tmp_path, make_pairs):
    cases = _small_world(make_pairs)
    spec = harness.load_specs(paths.REAL_MODELS_FILE)[0]
    api, sleeper = _Forced(cases, [spec]), rehearse.Sleeper()
    api.persistent_cell = None
    path = harness.run_model(spec, cases, tmp_path, api.factory("openai"), {"stimulus_sha256": "s"}, sleep=sleeper)
    recs = harness.read_records(path)
    assert all(r["status"] == "ok" for r in recs) and api.events["rate_limited_429"] == 6
    assert len(sleeper.delays) == 6 and all(d >= 7 or 25 <= d <= 31 for d in sleeper.delays)


def test_fake_api_invalid_replies_end_as_non_responses_after_one_re_request(tmp_path, make_pairs):
    cases = _small_world(make_pairs, 1)
    spec = harness.load_specs(paths.REAL_MODELS_FILE)[0]
    api = _Invalid(cases, [spec])
    api.persistent_cell = None
    path = harness.run_model(spec, cases, tmp_path, api.factory("openai"), {"stimulus_sha256": "s"},
                             sleep=rehearse.Sleeper())
    recs = harness.read_records(path)
    assert [r["status"] for r in recs] == ["non_response", "non_response"]
    assert all(len(r["attempts"]) == 2 for r in recs)


def test_fake_api_reasoning_replies_carry_think_tags_and_still_parse(tmp_path, make_pairs):
    cases = _small_world(make_pairs, 1)
    spec = next(s for s in harness.load_specs(paths.REAL_MODELS_FILE) if s["reasoning"])
    api = rehearse.FakeAPI(cases, [spec])
    api.P_INVALID = api.P_THINK_OPEN = api.P_429 = api.P_400 = 0.0
    api.persistent_cell = None
    path = harness.run_model(spec, cases, tmp_path, api.factory(spec["provider"]), {"stimulus_sha256": "s"},
                             sleep=rehearse.Sleeper())
    recs = harness.read_records(path)
    assert all(r["status"] == "ok" for r in recs)
    assert all(r["attempts"][0]["response"]["choices"][0]["message"]["content"].startswith("<think>") for r in recs)


def test_fake_api_persistent_cell_ends_as_persistent_error_after_a_resume(tmp_path, make_pairs):
    cases = _small_world(make_pairs, 2)
    specs = harness.load_specs(paths.REAL_MODELS_FILE)[:2]
    api1 = rehearse.FakeAPI(cases, specs, round_no=1)
    model = api1.persistent_cell[0]
    spec = next(s for s in specs if s["model_id"] == model)
    for round_no in (1, 2):
        api = rehearse.FakeAPI(cases, specs, round_no=round_no)
        harness.run_model(spec, cases, tmp_path, api.factory("openai"), {"stimulus_sha256": "s"},
                          sleep=rehearse.Sleeper())
    final = {(r["case_id"], r["order"]): r for r in harness.final_records(harness.results_path(spec, tmp_path))}
    cell = final[api1.persistent_cell[1:]]
    assert cell["persistent_error"] is True and "<req>" in cell["error_key"]
    assert harness.unresolved_cells(harness.results_path(spec, tmp_path)) == []


# ---------------------------------------------------------------- marker classification
@pytest.mark.skipif(not (REAL_DOCS / "edits" / "fill.py").exists(), reason="npj_R2_WORKING not available")
def test_every_marker_of_the_real_documents_is_classified():
    res = pipeline.markers(work=REAL_DOCS, write=False)
    assert res["counts"]["UNCLASSIFIED"] == 0, res["UNCLASSIFIED"]
    assert res["counts"]["NUMERIC"] + res["counts"]["PROSE"] > 0


@pytest.mark.skipif(not (REAL_DOCS / "edits" / "fill.py").exists(), reason="npj_R2_WORKING not available")
@pytest.mark.parametrize("desc,cls", [
    ("freeze_date", "NUMERIC"), ("rep_call_dates", "NUMERIC"), ("dates of the main run", "NUMERIC"),
    ("Figure 3 image: failure rate by modification type", "NUMERIC"),
    ("Supplementary Table S13: per-model accuracy", "NUMERIC"),
    ("confirm the 23 physician-authored edits in pairs_v2.json", "NUMERIC"),
    ("primary outcome sentence: pooled position-averaged accuracy with 95% CI", "PROSE"),
    ("confirm the list below against the frozen analysis plan", "PROSE"),
    ("something new", "UNCLASSIFIED"),
])
def test_classification_rules(desc, cls):
    assert pipeline._load_fill(REAL_DOCS).classify(desc) == cls


def test_pilot_summary_counts_replies_per_model(sandboxed, make_pairs):
    cases = _small_world(make_pairs, 2)
    spec = harness.load_specs(paths.REAL_MODELS_FILE)[0]
    paths.MODELS_FILE.write_text(json.dumps({"models": [spec]}))
    api = rehearse.FakeAPI(cases, [spec])
    harness.run_model(spec, cases, paths.RUNS_V2 / "pilot", api.factory("openai"), {"stimulus_sha256": "s"},
                      sleep=rehearse.Sleeper())
    (row,) = pipeline.pilot_summary()
    import csv
    (written,) = list(csv.DictReader(open(paths.ANALYSIS_REPEATED / "pilot_summary.csv")))
    assert float(written["parse_rate"]) == row["ok"] / 4 and written["first_call"] <= written["last_call"]
    assert written["synthetic"] == "True"
    assert row["cells"] == 4 and row["ok"] + row["non_response"] == 4 and row["api_errors"] == 1
    assert row["served"] == spec["model_id"]


# ---------------------------------------------------------------- scan-legacy-keys
def _fake_keys():
    """Fake keys built at run time, so this file never matches the secrets grep."""
    return {"together": "tgp" + "_v1_" + "Qz7Wm2Xk9Lp4Rv8Ty3Nb6Hc1Jd5Fg0Sa",
            "groq": "g" + "sk_" + "Vb3Nm8Qw2Er7Ty4Ui9Op1As6Df5Gh0Jk",
            "gemini": "AI" + "za" + "SyQ1w2E3r4T5y6U7i8O9p0AaBbCcDd",
            "deepseek_a": "s" + "k-" + "9f3b1c7e4a6d2085f1c3b9e7a5d4c2b1",
            "deepseek_b": "s" + "k-" + "7d2e9a4c1b8f6035e2d4a8c6b1f9e3d7",
            "openai": "s" + "k-proj-" + "Xy9Zw8Vu7Ts6Rq5Po4Nm3Lk2Ji1Hg0Fe"}


def test_scan_legacy_keys_names_files_and_providers_and_never_a_key(tmp_path, capsys):
    k = _fake_keys()
    root = tmp_path / "legacy"
    files = {"scripts/run_together_experiment.py": f'API_KEY = "{k["together"]}"\n',
             "scripts/run_deepseek_experiment.py": f'client = OpenAI(api_key="{k["deepseek_a"]}", '
                                                   'base_url="https://api.deepseek.com")\n',
             "scripts/run_deepseek_reasoner.py": f'DEEPSEEK_KEY = "{k["deepseek_b"]}"\n',
             "copy/scripts/run_deepseek_reasoner.py": f'DEEPSEEK_KEY = "{k["deepseek_b"]}"\n',
             "eval/.env": f'OPENAI_API_KEY={k["openai"]}\nGEMINI_API_KEY={k["gemini"]}\n',
             "notes/groq.txt": f'groq: {k["groq"]}\n',
             "scripts/clean.py": "print('no key here')\n",
             "scripts/._run_together_experiment.py": f'{k["together"]}\n',     # AppleDouble: skipped
             ".venv/lib/site.py": f'{k["groq"]}\n',                              # virtual environment: skipped
             "some_repo/leak.py": f'{k["together"]}\n'}                          # a git repository: skipped
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    (root / "some_repo" / ".git").mkdir()
    (root / "blob.bin").write_bytes(b"\x00\x01" + k["together"].encode())          # binary: skipped
    assert pipeline.main(["scan-legacy-keys", str(root)]) == 0
    cap = capsys.readouterr()
    out = cap.out + cap.err
    res = pipeline.scan_legacy_keys(root, quiet=True)
    rel = {str(Path(p).relative_to(root)): v for p, v in res["files"].items()}
    assert rel == {"scripts/run_together_experiment.py": ["Together"],
                   "scripts/run_deepseek_experiment.py": ["DeepSeek"],
                   "scripts/run_deepseek_reasoner.py": ["DeepSeek"],
                   "copy/scripts/run_deepseek_reasoner.py": ["DeepSeek"],
                   "eval/.env": ["Google Gemini", "OpenAI"],
                   "notes/groq.txt": ["Groq"]}
    assert res["providers"]["DeepSeek"] == {"files": 3, "distinct_keys": 2}
    assert [Path(p).name for p in res["skipped_repositories"]] == ["some_repo"]
    for p in rel:
        assert p in out
    for key in k.values():                        # no part of any key value: no 6-character window, no prefix
        body = key
        assert not any(body[i:i + 6] in out for i in range(len(body) - 5)), key[:3]
    for prefix in ("tgp" + "_v1", "g" + "sk_", "AI" + "za", "s" + "k-"):
        assert prefix not in out


# ---------------------------------------------------------------- the public timestamp of the freeze (I4)
def _frozen(make_pairs):
    """A sandbox freeze (tag in the throwaway freeze repository) on today's date; returns the specs."""
    from tests.r2.test_sandbox_git import freeze_inputs
    specs, _ = freeze_inputs(make_pairs)
    pipeline.freeze(yes=True)
    return specs


TAG_URL = "https://github.com/example/r2-rehearsal/releases/tag/r2-analysis-plan-frozen"


@pytest.mark.parametrize("url,kind", [
    (TAG_URL, "pushed tag"),
    ("https://github.com/example/r2/tree/r2-analysis-plan-frozen", "pushed tag"),
    ("https://osf.io/abc12", "OSF registration"),
    ("https://doi.org/10.17605/OSF.IO/ABC12", "OSF registration"),
    ("10.5281/zenodo.1234567", "Zenodo DOI"),
    ("https://doi.org/10.5281/zenodo.1234567", "Zenodo DOI"),
    ("https://zenodo.org/records/1234567", "Zenodo DOI"),
    ("https://github.com/example/r2/releases/tag/other-tag", None),
    ("http://osf.io/abc12", None),
    ("https://example.org/freeze", None),
])
def test_public_timestamp_kinds(url, kind):
    assert pipeline.public_kind(url) == kind


def test_record_public_writes_the_freeze_record_and_refuses_bad_input(sandboxed, make_pairs):
    with pytest.raises(pipeline.Refused, match="missing input"):
        pipeline.record_public(TAG_URL)                               # no freeze yet
    _frozen(make_pairs)
    today = pipeline._today()
    with pytest.raises(pipeline.Refused, match="not a public timestamp"):
        pipeline.record_public("https://example.org/freeze")
    with pytest.raises(pipeline.Refused, match="not a calendar date"):
        pipeline.record_public(TAG_URL, date="2026-02-30")
    with pytest.raises(pipeline.Refused, match="before the freeze"):
        pipeline.record_public(TAG_URL, date="2000-01-01")
    with pytest.raises(pipeline.Refused, match="in the future"):
        pipeline.record_public(TAG_URL, date="2999-01-01")
    res = pipeline.record_public(TAG_URL)
    assert res == {**res, "url": TAG_URL, "kind": "pushed tag", "since": today}
    text = paths.FREEZE_RECORD.read_text()
    assert f"- Public URL: {TAG_URL}\n" in text and "pending a push" not in text
    assert f"- Tag commit: {res['tag_commit']}" in text
    assert pipeline.public_timestamp() == {k: res[k] for k in ("url", "kind", "since", "tag_commit")}
    assert pipeline.record_public(TAG_URL)["already"] is True        # idempotent
    with pytest.raises(pipeline.Refused, match="already records"):
        pipeline.record_public("https://osf.io/abc12")


def _main_world(make_pairs):
    specs = _frozen(make_pairs)
    pairs = json.loads((paths.DATA_V2 / "pairs_v2.json").read_text())
    cases = {c["case_id"]: c for c in pairs["cases"]}
    return specs, rehearse.FakeAPI(cases, specs)


def test_main_refuses_until_the_freeze_is_public(sandboxed, make_pairs):
    specs, api = _main_world(make_pairs)
    one = specs[0]["original_name"]
    with pytest.raises(pipeline.Refused, match="no public timestamp"):
        pipeline.run_harness("main", yes=True, client_factory=api.factory, sleep=rehearse.Sleeper(), models=one)
    assert not (paths.RUNS_V2 / "main").exists()                      # nothing ran, nothing written
    pipeline.record_public(TAG_URL)
    rc = pipeline.run_harness("main", yes=True, client_factory=api.factory, sleep=rehearse.Sleeper(), models=one)
    assert rc in (0, 1)
    assert not (paths.RUNS_V2 / "main" / pipeline.PIPELINE_INCIDENTS).exists()


def test_accept_unpublished_freeze_writes_an_incident(sandboxed, make_pairs, capsys):
    specs, api = _main_world(make_pairs)
    rc = pipeline.run_harness("main", yes=True, client_factory=api.factory, sleep=rehearse.Sleeper(),
                              models=specs[0]["original_name"], accept_unpublished_freeze=True)
    assert rc in (0, 1)
    (line,) = (paths.RUNS_V2 / "main" / pipeline.PIPELINE_INCIDENTS).read_text().splitlines()
    inc = json.loads(line)
    assert inc["kind"] == "unpublished_freeze" and inc["freeze_record"] == str(paths.FREEZE_RECORD)
    assert "ACCEPTED" in capsys.readouterr().out


def test_main_refuses_without_the_freeze_tag(sandboxed, make_pairs):
    from tests.r2.test_sandbox_git import freeze_inputs
    specs, pairs = freeze_inputs(make_pairs)
    api = rehearse.FakeAPI({c["case_id"]: c for c in pairs["cases"]}, specs)
    with pytest.raises(pipeline.Refused, match="not frozen"):
        pipeline.run_harness("main", yes=True, client_factory=api.factory, accept_unpublished_freeze=True)


def _order_world(freeze_date, first_call, public_since=None):
    rec = paths.FREEZE_RECORD
    rec.parent.mkdir(parents=True, exist_ok=True)
    text = f"# Freeze record\n\n- Date: {freeze_date}\n\nPublic timestamp: pending a push.\n"
    if public_since:
        text += (f"\n- Public URL: {TAG_URL}\n- Kind: pushed tag\n- Public since: {public_since}\n"
                 "- Tag commit: abc\n")
    rec.write_text(text)
    paths.ANALYSIS_REPEATED.mkdir(parents=True, exist_ok=True)
    (paths.ANALYSIS_REPEATED / "headline.json").write_text(json.dumps(
        {"rep_call_dates": {"first": first_call, "last": first_call}}))


def test_order_check_passes_and_prints_each_comparison(sandboxed, capsys):
    _order_world("2026-10-06", "2026-10-07T09:00:00+00:00", public_since="2026-10-06")
    res = pipeline.order_check()
    assert res["ok"] is True
    out = capsys.readouterr().out
    assert "ORDER CHECK freeze: 2026-10-06 <= first main-run call 2026-10-07T09:00:00+00:00: PASS" in out
    assert "ORDER CHECK public: 2026-10-06 <= first main-run call 2026-10-07T09:00:00+00:00: PASS" in out


@pytest.mark.parametrize("freeze_date,public,match", [
    ("2026-10-08", "2026-10-08", "freeze date 2026-10-08 is after the first main-run call"),
    ("2026-10-06", "2026-10-08", "public timestamp 2026-10-08 is after the first main-run call"),
    ("2026-10-06", None, "no public timestamp"),
])
def test_order_check_failures_stop_post_results(sandboxed, freeze_date, public, match):
    _order_world(freeze_date, "2026-10-07T09:00:00+00:00", public_since=public)
    with pytest.raises(pipeline.Refused, match=match):
        pipeline.order_check()


def test_order_check_accepts_a_logged_unpublished_freeze(sandboxed, capsys):
    _order_world("2026-10-06", "2026-10-07T09:00:00+00:00")
    harness._log_run_incident(paths.RUNS_V2 / "main", {"kind": "unpublished_freeze",
                                                      "freeze_record": str(paths.FREEZE_RECORD)})
    assert pipeline.order_check()["ok"] is True
    assert "ACCEPTED by --accept-unpublished-freeze" in capsys.readouterr().out


def test_order_check_cannot_be_skipped_once_the_main_run_exists(sandboxed):
    _order_world("2026-10-06", None, public_since="2026-10-06")
    with pytest.raises(pipeline.Refused, match="cannot be made"):
        pipeline.order_check()


# ---------------------------------------------------------------- QA verdicts stop post-results (I3)
def _stub_qa(work, failing=()):
    reports = {"qa_numbers.py": ("qa_numbers_report.md", "**UNEXPLAINED: {n}**"),
               "qa_xref.py": ("qa_xref_report.md", "Cross-reference failures: {n}"),
               "qa_keyfacts.py": ("qa_keyfacts_report.md", "Key-fact failures: {n}"),
               "qa_hygiene.py": ("qa_hygiene_report.md", "## Total hygiene hits: {n}")}
    work.mkdir(parents=True, exist_ok=True)
    for script, (report, line) in reports.items():
        n = 2 if script in failing else 0
        (work / script).write_text(
            "import sys\nfrom pathlib import Path\n"
            f"Path(__file__).with_name({report!r}).write_text({line.format(n=n)!r} + '\\n')\n"
            f"sys.exit({1 if n else 0})\n")


def test_qa_verdicts_are_printed_and_a_failure_stops(sandboxed, capsys):
    _stub_qa(paths.R2_WORKING)
    res = pipeline.run_qa()
    assert {k: v["failures"] for k, v in res.items()} == {s: 0 for s in pipeline.QA_SCRIPTS}
    out = capsys.readouterr().out
    assert "QA qa_xref.py: Cross-reference failures: 0: PASS" in out
    _stub_qa(paths.R2_WORKING, failing=("qa_xref.py",))
    with pytest.raises(pipeline.Refused, match=r"QA failed: qa_xref\.py: Cross-reference failures: 2"):
        pipeline.run_qa()


def test_a_qa_script_that_crashes_is_a_failure(sandboxed):
    _stub_qa(paths.R2_WORKING)
    (paths.R2_WORKING / "qa_keyfacts.py").write_text("raise RuntimeError('boom')\n")
    with pytest.raises(pipeline.Refused, match="qa_keyfacts.py"):
        pipeline.run_qa()


# ---------------------------------------------------------------- the completeness gate stops post-results (M9)
def test_post_results_stops_at_a_failing_completeness_gate(sandboxed, make_pairs, monkeypatch):
    from r2.run import check_complete
    from tests.r2.test_sandbox_git import freeze_inputs
    specs, _ = freeze_inputs(make_pairs)
    monkeypatch.setattr(check_complete, "load_v2_cases", lambda: harness.load_v2_cases(paths.DATA_V2 / "pairs_v2.json"))
    monkeypatch.setattr(check_complete, "load_specs", lambda: harness.load_specs(paths.MODELS_FILE))
    monkeypatch.setattr(check_complete, "ANALYSIS_REPEATED", paths.ANALYSIS_REPEATED)
    (paths.RUNS_V2 / "main").mkdir(parents=True)
    harness.results_path(specs[0], paths.RUNS_V2 / "main").write_text("")     # 0 of 1000 cells
    paths.ANALYSIS_REPEATED.mkdir(parents=True)
    (paths.ANALYSIS_REPEATED / "model_availability.csv").write_text("x\n")
    (paths.R2_WORKING / "edits").mkdir(parents=True)
    (paths.R2_WORKING / "edits" / "numbers.py").write_text("")
    with pytest.raises(pipeline.Refused, match="check_complete"):
        pipeline.post_results(render=False)
    assert not (paths.ANALYSIS_REPEATED / "headline.json").exists()          # nothing after the gate ran


# ---------------------------------------------------------------- import-edits refusals (M8e)
def test_a_rejected_author_edit_is_a_refusal(sandboxed, monkeypatch):
    from r2.verify import build_packets as BP
    (paths.VERIFY_DIR / "round1").mkdir(parents=True)
    (paths.VERIFY_DIR / "round1" / BP.MANIFEST).write_text(json.dumps({"unresolved": ["CASE_001"]}))
    paths.DATA_V2.mkdir(parents=True)
    (paths.DATA_V2 / "pairs_v2.json").write_text(json.dumps({"cases": [], "unresolved": ["CASE_001"]}))
    book = sandboxed / "AG.xlsx"
    book.write_text("x")

    def reject(*a, **k):
        raise ValueError("CASE_001: 'find' must occur exactly once in excerpt_text")
    monkeypatch.setattr(BP, "import_author_edits", reject)
    with pytest.raises(pipeline.Refused, match="exactly once"):
        pipeline.import_edits(book)
    assert pipeline.main(["import-edits", str(book)]) == 2


@pytest.mark.skipif(not (REAL_DOCS / "edits" / "fill.py").exists(), reason="npj_R2_WORKING not available")
def test_the_physician_edit_confirmation_needs_the_queue_count_and_the_named_author():
    fill = pipeline._load_fill(REAL_DOCS)

    def v(**kw):
        return {k: {"text": x} for k, x in kw.items()}
    good = v(v2_n_physician_edits="23", v2_n_queue_round1="23", v2_unresolved="23", v2_identical_all500="0",
             v2_manual_edit_authors="A.G.")
    assert fill.confirmed("physician_edits", good)
    assert not fill.confirmed("physician_edits", {**good, **v(v2_manual_edit_authors="SIMULATED")})
    assert not fill.confirmed("physician_edits", {**good, **v(v2_manual_edit_authors="A.G.; M.O.")})
    assert not fill.confirmed("physician_edits", {**good, **v(v2_n_physician_edits="22", v2_unresolved="22")})
    assert not fill.confirmed("physician_edits", {**good, **v(v2_identical_all500="1")})
    assert not fill.confirmed("physician_edits", {k: x for k, x in good.items() if k != "v2_n_queue_round1"})
