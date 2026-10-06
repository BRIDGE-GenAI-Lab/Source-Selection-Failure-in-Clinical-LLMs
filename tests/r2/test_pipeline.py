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
              "PUBLIC_RECORD": box / "PUBLIC_RECORD.json"}
    layout.update({"TABLES_DIR": layout["R2_WORKING"] / "tables", "FIGURES_DIR": layout["R2_WORKING"] / "05_Figures",
                   "FREEZE_RECORD": layout["FREEZE_REPO"] / "docs" / "r2" / "FREEZE_RECORD.md"})
    monkeypatch.setattr(paths, "SANDBOX", box)
    for k, v in layout.items():
        monkeypatch.setattr(paths, k, v)
    monkeypatch.setattr(paths, "WRITE_PATHS", dict(layout))
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
def test_record_writes_the_public_facts_and_refuses_bad_or_conflicting_values(sandboxed):
    data = pipeline.record(freeze_pushed="2026-10-20")
    assert data == {"freeze_pushed": "2026-10-20", "synthetic": True}
    data = pipeline.record(release_tag="r2-final", zenodo_doi="10.5281/zenodo.1234567")
    assert json.loads(paths.PUBLIC_RECORD.read_text())["release_tag"] == "r2-final"
    with pytest.raises(pipeline.Refused, match="YYYY-MM-DD"):
        pipeline.record(freeze_pushed="20 Oct 2026")
    with pytest.raises(pipeline.Refused, match="DOI"):
        pipeline.record(zenodo_doi="zenodo 123")
    with pytest.raises(pipeline.Refused, match="different values"):
        pipeline.record(release_tag="r2-other")
    with pytest.raises(pipeline.Refused, match="nothing to record"):
        pipeline.record()


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
    assert row["cells"] == 4 and row["ok"] + row["non_response"] == 4 and row["api_errors"] == 1
    assert row["served"] == spec["model_id"]
