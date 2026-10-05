"""A synthetic repeated experiment run through the whole post-results pipeline produces every key and file
that the documents depend on (r2/analysis/expected_outputs.json)."""
import csv
import json
from pathlib import Path

import pytest

from r2.analysis import repeated, synthetic
from r2.run import check_complete
from r2.run.harness import load_specs

EXPECTED = json.loads((Path(repeated.__file__).parent / "expected_outputs.json").read_text())


def _refuse(token):
    raise ValueError(f"invalid JSON constant {token}")


def _get(obj, dotted):
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            raise KeyError(dotted)
        obj = obj[part]
    return obj


def _keys(spec):
    for key in spec.get("keys", {}):
        if key.startswith("<S>."):
            for s in spec["each_of"]:
                yield s + key[3:]
        else:
            yield key


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    world = synthetic.build_world(root)
    dirs = synthetic.run_pipeline(world)
    return world, dirs


def _path(dirs, rel):
    head, name = rel.split("/", 1)
    return Path(dirs[head]) / name


def test_expected_outputs_reference_known_markers():
    markers = set(EXPECTED["markers"])
    used = set()
    for spec in EXPECTED["outputs"].values():
        for ids in list(spec.get("keys", {}).values()) + list(spec.get("columns", {}).values()) + \
                list(spec.get("rows", {}).values()) + [spec.get("markers", [])]:
            assert set(ids) <= markers, ids
            used |= set(ids)
    assert used == markers  # every marker is filled by at least one output


def test_the_documents_named_key_is_expected():
    keys = EXPECTED["outputs"]["analysis_repeated/headline.json"]["keys"]
    for k in ("primary.position_averaged_accuracy", "primary.paa_lo", "primary.paa_hi"):
        assert k in keys


@pytest.mark.parametrize("rel", sorted(EXPECTED["outputs"]))
def test_synthetic_run_produces_every_expected_output(run, rel):
    _, dirs = run
    spec = EXPECTED["outputs"][rel]
    path = _path(dirs, rel)
    assert path.exists() and path.stat().st_size > 0, path
    if spec["type"] == "json":
        data = json.loads(path.read_text(), parse_constant=_refuse)
        for key in _keys(spec):
            _get(data, key)
    elif spec["type"] == "csv":
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, path
        missing = set(spec.get("columns", {})) - set(rows[0])
        assert not missing, (rel, missing)
        if "rows" in spec:
            labels = {r[next(iter(rows[0]))] for r in rows}
            assert set(spec["rows"]) <= labels, (rel, set(spec["rows"]) - labels)


def test_aliases_hold_in_the_synthetic_run(run):
    _, dirs = run
    for rel, aliases in EXPECTED["aliases"].items():
        spec = EXPECTED["outputs"][rel]
        data = json.loads(_path(dirs, rel).read_text())
        for alias, key in aliases.items():
            names = spec.get("each_of", [None])
            for s in names:
                a, k = (alias, key) if s is None else (alias.replace("<S>", s), key.replace("<S>", s))
                assert _get(data, a) == _get(data, k)


def test_headline_numbers_are_finite_and_cis_bracket(run):
    _, dirs = run
    head = json.loads((Path(dirs["analysis_repeated"]) / "headline.json").read_text())
    prim = head["primary"]
    for est, lo, hi in (("accuracy", "acc_lo", "acc_hi"), ("primacy", "primacy_lo", "primacy_hi"),
                        ("acc_sham_first", "acc_sham_first_lo", "acc_sham_first_hi"),
                        ("acc_sham_second", "acc_sham_second_lo", "acc_sham_second_hi"),
                        ("position_averaged_accuracy", "paa_lo", "paa_hi")):
        assert prim[lo] <= prim[est] <= prim[hi], est
    assert head["n_models"] == len(load_specs()) and head["non_response"] > 0
    assert head["exclusion_rule"] and head["rep_call_dates"]["first"] < head["rep_call_dates"]["last"]


def test_synthetic_world_passes_the_completeness_gate(run):
    world, _ = run
    report = check_complete.check(world["runs_main"], load_specs(), sorted(world["case_ids"]))
    for r in report:
        for k in ("missing", "duplicates", "unexpected", "invalid_records"):
            assert r[k] == 0, (r["model"], k)
    assert sum(r["superseded_errors"] for r in report) >= 1  # the last-record rule is exercised
    assert sum(r["non_response"] for r in report) > 0


def test_synthetic_run_writes_nothing_outside_its_root(run):
    world, dirs = run
    root = Path(world["root"]).resolve()
    for d in dirs.values():
        assert root in Path(d).resolve().parents or Path(d).resolve() == root


def test_call_dates_include_the_superseded_early_error(run):
    from r2.run.harness import read_records, results_path
    world, dirs = run
    head = json.loads((Path(dirs["analysis_repeated"]) / "headline.json").read_text())
    every = [r["finished_at"] for spec in world["specs"] for r in read_records(results_path(spec, world["runs_main"]))]
    assert head["rep_call_dates"]["first"] == min(every) == "2026-11-01T08:00:00+00:00"
    assert head["rep_call_dates"]["last"] == max(every)
    assert set(head["rep_call_dates"]["per_model"]) == {s["original_name"] for s in world["specs"]}


# ---- fix round 1, I2: the synthetic world stays in a sandbox and marks everything it writes ----
# SAFETY: no test below may hand a real location to a writer. The guard is checked as a pure function on the
# real paths (it writes nothing); every test that would write uses fake protected roots inside tmp, so a
# regression can only ever write into tmp.
def _real_targets():
    from r2 import paths
    from r2.analysis import figures, supp_tables
    t = {"repo": paths.REPO, "parent_of_repo": paths.REPO.parent, "legacy": paths.LEGACY_ROOT / "synthetic_world",
         "runs_v2": paths.RUNS_V2, "runs_v2_main": paths.RUNS_V2 / "main", "analysis_out": repeated.OUT,
         "figures_out": figures.OUT_DIR, "tables_out": supp_tables.OUT_DIR}
    for sub in ("data", "verification", "analysis_repeated", "audit", "analysis_original"):
        t[sub] = paths.REPO / sub / "synthetic_world"
    main_repo = Path.home() / "npj_r2" / "r2_repo"
    if main_repo.exists():
        t["other_checkout"] = main_repo / "synthetic_world"
    return t


@pytest.mark.parametrize("where", sorted(_real_targets()))
def test_sandbox_guard_rejects_every_real_location(where):
    with pytest.raises(ValueError, match="sandbox"):
        synthetic.assert_sandbox(_real_targets()[where])  # pure check: writes nothing


def test_sandbox_guard_accepts_a_tmp_dir(tmp_path):
    assert synthetic.assert_sandbox(tmp_path / "world") == (tmp_path / "world").resolve()


@pytest.fixture
def fake_real(tmp_path, monkeypatch):
    """Protected roots that live in tmp: a fake checkout (with r2/paths.py), a fake legacy root and RUNS_V2."""
    from r2 import paths
    from r2.analysis import figures, supp_tables
    prot = tmp_path / "protected"
    repo, legacy = prot / "repo", prot / "legacy"
    (repo / "r2").mkdir(parents=True)
    (repo / "r2" / "paths.py").write_text("# fake checkout marker\n")
    (repo / "figures").mkdir()
    legacy.mkdir()
    monkeypatch.setattr(paths, "REPO", repo)
    monkeypatch.setattr(paths, "LEGACY_ROOT", legacy)
    monkeypatch.setattr(paths, "RUNS_V2", legacy / "r2_runs")
    monkeypatch.setattr(repeated, "OUT", repo / "analysis_repeated")
    monkeypatch.setattr(figures, "OUT_DIR", legacy / "npj_R2_WORKING" / "05_Figures")
    monkeypatch.setattr(supp_tables, "OUT_DIR", legacy / "npj_R2_WORKING" / "tables")
    return {"prot": prot, "repo": repo, "legacy": legacy}


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*"))


@pytest.mark.parametrize("rel", ["repo", "repo/data/w", "repo/verification/w", "repo/analysis_repeated",
                                 "repo/audit/w", "repo/analysis_original/w", "legacy/w", "legacy/r2_runs",
                                 "legacy/r2_runs/main", "."])
def test_build_world_refuses_protected_roots_and_creates_nothing(fake_real, rel):
    before = _tree(fake_real["prot"])
    with pytest.raises(ValueError, match="sandbox"):
        synthetic.build_world(fake_real["prot"] / rel)
    assert _tree(fake_real["prot"]) == before


def test_build_world_refuses_a_non_empty_root(tmp_path):
    (tmp_path / "keep.txt").write_text("not mine")
    with pytest.raises(ValueError, match="empty"):
        synthetic.build_world(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]


def test_run_pipeline_refuses_protected_output_dirs_before_any_step(fake_real, tmp_path):
    world = synthetic.build_world(tmp_path / "w")
    before = _tree(fake_real["prot"])
    with pytest.raises(ValueError, match="sandbox"):
        synthetic.run_pipeline(world, figures_dir=fake_real["repo"] / "figures")
    assert _tree(fake_real["prot"]) == before
    assert not (world["analysis_dir"] / "headline.json").exists()  # refused before any step ran


def test_synthetic_results_are_refused_in_the_default_output_dirs(fake_real, tmp_path):
    from r2.analysis import figures, supp_tables, table1
    world = synthetic.build_world(tmp_path / "w")
    dirs = synthetic.run_pipeline(world)
    before = _tree(fake_real["prot"])
    with pytest.raises(ValueError, match="sandbox"):
        repeated.run(world["runs_main"], analysis_dir=None, pairs_path=world["pairs_path"],
                     verify_dir=world["verify_dir"], specs=world["specs"])
    with pytest.raises(ValueError, match="sandbox"):
        figures.make_figures(dirs["analysis_repeated"], None, specs=world["specs"])
    with pytest.raises(ValueError, match="sandbox"):
        table1.write_table1(supp_tables.OUT_DIR, pairs_path=world["pairs_path"], verify_dir=world["verify_dir"],
                            runs_dir=world["runs_main"], specs=world["specs"])
    with pytest.raises(ValueError, match="sandbox"):
        supp_tables.write_repeated(None, root=world["root"])
    assert _tree(fake_real["prot"]) == before


def _csv_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_every_file_of_the_synthetic_world_is_marked(run):
    from r2.run.harness import read_records, results_path
    world, _ = run
    assert json.loads(world["pairs_path"].read_text())["synthetic"] is True
    for name in ("excluded_pairs.json", "flags.json"):
        assert json.loads((world["verify_dir"] / name).read_text())["synthetic"] is True, name
    for path in (world["verify_dir"] / "kappa.csv", world["verify_dir"] / "final_review.csv",
                 world["analysis_dir"] / "completeness.csv", world["analysis_dir"] / "model_availability.csv"):
        assert {r["synthetic"] for r in _csv_rows(path)} == {"True"}, path
    for spec in world["specs"]:
        assert all(r["synthetic"] is True for r in read_records(results_path(spec, world["runs_main"])))


def test_the_synthetic_flag_reaches_every_output(run):
    _, dirs = run
    ana, figs, tabs = (Path(dirs[k]) for k in ("analysis_repeated", "figures", "tables"))
    assert EXPECTED["provenance"]["key"] == "synthetic"
    for p in sorted(ana.glob("*.json")):
        assert json.loads(p.read_text())["synthetic"] is True, p.name
    for p in sorted(ana.glob("*.csv")) + sorted(tabs.glob("*.csv")):
        assert {r["synthetic"] for r in _csv_rows(p)} == {"True"}, p.name
    for p in sorted(figs.glob("*_legend.md")) + [tabs / "Table1_note.txt"]:
        assert p.read_text().startswith(synthetic.SYNTHETIC_BANNER), p.name





def test_expected_outputs_name_the_markers_sourced_elsewhere():
    out = {m["marker"] for m in EXPECTED["out_of_scope"]}
    for key in ("verif_n_reviewed", "verif_kappa_range", "verif_n_excluded", "verif_n_flagged", "freeze_date",
                "release_tag", "zenodo_new_doi", "v2_identical_all500", "v2_n_physician_edits"):
        assert any(key in m for m in out), key
    keys = EXPECTED["outputs"]["analysis_repeated/headline.json"]["keys"]
    for k in ("substitutions", "n_api_errors", "n_flagged_analyzed_pairs", "gee_converged", "rep_call_dates.per_model"):
        assert k in keys, k
    sens = EXPECTED["outputs"]["analysis_repeated/sensitivity.json"]["keys"]
    for k in ("primacy", "acc_sham_first", "acc_sham_second"):
        assert f"<S>.{k}_lo" in sens and f"<S>.{k}_hi" in sens, k


# ---- fix round 2: the guard is case-insensitive (the Lexar volume is) and guards the CLI ----
def _variants(path):
    s = str(path)
    return {"upper": s.upper(), "lower": s.lower(), "swapcase": s.swapcase(),
            "mixed": "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(s))}


def _protected_real():
    from r2 import paths
    t = {"runs_v2": paths.RUNS_V2, "runs_v2_main": paths.RUNS_V2 / "main", "legacy": paths.LEGACY_ROOT,
         "legacy_child": paths.LEGACY_ROOT / "npj_R2_WORKING" / "05_Figures", "repo": paths.REPO}
    for sub in ("data", "verification", "analysis_repeated", "audit", "analysis_original"):
        t[sub] = paths.REPO / sub
    return t


@pytest.mark.parametrize("where", sorted(_protected_real()))
@pytest.mark.parametrize("case", ["upper", "lower", "swapcase", "mixed"])
def test_sandbox_guard_rejects_case_variants_of_real_locations(where, case):
    target = _variants(_protected_real()[where])[case]
    with pytest.raises(ValueError, match="sandbox"):
        synthetic.assert_sandbox(target)  # pure check: writes nothing
    with pytest.raises(ValueError, match="sandbox"):
        synthetic.assert_sandbox(target + "/synthetic_world")


@pytest.mark.parametrize("case", ["upper", "lower", "swapcase", "mixed"])
def test_build_world_refuses_case_variants_of_a_protected_root(fake_real, case):
    # fake roots in tmp; the tmp volume is case-insensitive, so a variant names the same directory
    before = _tree(fake_real["prot"])
    for target in (fake_real["legacy"] / "r2_runs", fake_real["legacy"] / "w"):
        with pytest.raises(ValueError, match="sandbox"):
            synthetic.build_world(_variants(target)[case])
    assert _tree(fake_real["prot"]) == before


def test_sandbox_guard_uses_samefile_for_another_spelling(fake_real, tmp_path):
    link = tmp_path / "alias_of_legacy"
    link.symlink_to(fake_real["legacy"], target_is_directory=True)
    with pytest.raises(ValueError, match="sandbox"):
        synthetic.assert_sandbox(link / "w")


def test_cli_refuses_a_protected_root_and_creates_nothing(fake_real):
    before = _tree(fake_real["prot"])
    for target in (fake_real["legacy"] / "w", _variants(fake_real["legacy"] / "r2_runs")["upper"]):
        with pytest.raises(SystemExit, match="sandbox"):
            synthetic.main([str(target)])
    assert _tree(fake_real["prot"]) == before


def test_cli_runs_in_a_sandbox(tmp_path, capsys):
    synthetic.main([str(tmp_path / "world")])
    assert (tmp_path / "world" / "analysis_repeated" / "headline.json").exists()
    assert "figures" in capsys.readouterr().out
