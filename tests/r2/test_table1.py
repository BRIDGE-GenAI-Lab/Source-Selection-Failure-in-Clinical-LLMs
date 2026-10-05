import csv
import json

import pytest

from r2 import paths
from r2.analysis import synthetic, table1
from r2.analysis.stats import CATEGORY
from r2.analysis.table1 import modification, table1_rows
from r2.paths import RUNS_V2, VERIFY_DIR
from r2.run.harness import final_records, results_path

KEYS = ["sham_type", "outcome", "case_id", "model", "order", "patient", "modification", "rationale_quote"]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return synthetic.build_world(tmp_path_factory.mktemp("t1"))


def _kw(world):
    return {"pairs_path": world["pairs_path"], "verify_dir": world["verify_dir"], "runs_dir": world["runs_main"],
            "specs": world["specs"]}


def _records(world):
    out = {}
    for spec in world["specs"]:
        for r in final_records(results_path(spec, world["runs_main"])):
            out[(r["original_name"], r["case_id"], r["order"])] = r
    return out


def test_one_failure_and_one_success_per_type_from_verified_pairs(world):
    rows = table1_rows(**_kw(world))
    assert all(list(r) == KEYS for r in rows)
    assert {(r["sham_type"], r["outcome"]) for r in rows} == {(t, o) for t in CATEGORY for o in ("Failure", "Success")}
    assert len(rows) == 2 * len(CATEGORY)
    excluded = set(json.loads((world["verify_dir"] / "excluded_pairs.json").read_text())["excluded"])
    flagged = set(json.loads((world["verify_dir"] / "flags.json").read_text()))
    recs = _records(world)
    for r in rows:
        assert r["case_id"] not in excluded | flagged
        rec = recs[(r["model"], r["case_id"], r["order"])]
        assert rec["status"] == "ok"
        assert rec["correct"] == (0 if r["outcome"] == "Failure" else 1)
        assert r["rationale_quote"].rstrip("…") in rec["decision"]["trust_rationale"]


def test_selection_is_deterministic_for_the_seed(world):
    assert table1_rows(**_kw(world)) == table1_rows(**_kw(world))
    assert table1_rows(seed=1, **_kw(world)) != table1_rows(**_kw(world))


def test_patient_and_modification_text(world):
    pairs = {c["case_id"]: c for c in json.loads(world["pairs_path"].read_text())["cases"]}
    for r in table1_rows(**_kw(world)):
        case = pairs[r["case_id"]]
        demo = case["demographics"]
        assert r["patient"].startswith(f"{demo['age']}-year-old ")
        assert case["chief_complaint"] in r["patient"]
        assert r["modification"] == modification(case["edit_record"])
        assert r["modification"].split(": ", 1)[1].startswith(("Removed '", "Added '", "Changed '"))


def test_modification_formats_and_truncates():
    long = "x" * 200
    rec = {"changes": [
        {"field": "safety_guardrails", "before": ["Keep this", "Assess bleeding risk"], "after": ["Keep this"]},
        {"field": "exclusions_or_cautions", "before": [], "after": ["Most contraindications are relative"]},
        {"field": "publication_year", "before": 2019, "after": 2014},
        {"field": "excerpt_text", "before": "a\nb", "after": "a\nb\n" + long},
    ]}
    text = modification(rec)
    parts = text.split("; ")
    assert parts[0] == "safety_guardrails: Removed 'Assess bleeding risk'"
    assert parts[1] == "exclusions_or_cautions: Added 'Most contraindications are relative'"
    assert parts[2] == "publication_year: Changed '2019' → '2014'"
    assert parts[3] == "excerpt_text: Added '" + "x" * 120 + "…'"


def test_changed_line_is_focused_on_the_edit():
    before = "• Anticoagulation in this setting for adults with confirmed disease: DOAC preferred for unprovoked VTE"
    after = before.replace("preferred", "avoided")
    text = modification({"changes": [{"field": "excerpt_text", "before": before, "after": after}]})
    assert "preferred" in text and "avoided" in text and "→" in text


def test_rationale_quote_is_cut_at_160_characters():
    assert table1.quote("a" * 160) == "a" * 160
    assert table1.quote("a" * 161) == "a" * 160 + "…"


def test_paths_come_from_r2_paths_at_call_time(world, monkeypatch):
    monkeypatch.setattr(paths, "DATA_V2", world["pairs_path"].parent)
    monkeypatch.setattr(paths, "VERIFY_DIR", world["verify_dir"])
    monkeypatch.setattr(paths, "RUNS_V2", world["runs_main"].parent)
    assert table1_rows() == table1_rows(**_kw(world))


def test_write_table1(world, tmp_path):
    rows = table1.write_table1(tmp_path, **_kw(world))
    with open(tmp_path / "Table1.csv", newline="") as fh:
        assert list(csv.DictReader(fh)) == rows
    note = (tmp_path / "Table1_note.txt").read_text()
    assert "20261005" in note and "excluded" in note and "TS" in note and "ST" in note


@pytest.mark.skipif(not (RUNS_V2 / "main").exists(), reason="the main run of the repeated experiment has not run")
def test_table1_quotes_real_responses_from_verified_pairs():
    rows = table1_rows()
    excluded = set(json.loads((VERIFY_DIR / "excluded_pairs.json").read_text())["excluded"])
    assert {r["outcome"] for r in rows} == {"Failure", "Success"}
    for r in rows:
        assert r["case_id"] not in excluded
        log = (RUNS_V2 / "main").glob("*.jsonl")
        found = False
        for path in log:
            if path.name.startswith("._"):
                continue
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                if (rec["case_id"], rec["order"], rec["original_name"]) == (r["case_id"], r["order"], r["model"]):
                    found = r["rationale_quote"].rstrip("…") in (rec["decision"] or {}).get("trust_rationale", "")
        assert found, r



def test_list_edits_are_grouped_per_field():
    rec = {"changes": [{"field": "applicability", "before": ["Adults", "Emergency settings"],
                        "after": ["All age groups", "Pediatric and adult populations"]}]}
    assert modification(rec) == ("applicability: Removed 'Adults' and 'Emergency settings', "
                                 "Added 'All age groups' and 'Pediatric and adult populations'")
