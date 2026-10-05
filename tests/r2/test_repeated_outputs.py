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
