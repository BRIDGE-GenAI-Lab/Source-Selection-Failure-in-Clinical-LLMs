import csv
import json

import pytest

from r2 import profiles
from r2.analysis import supp_tables as st
from r2.paths import AUDIT_DIR, LEGACY_EVAL

RESULT_TREES = [LEGACY_EVAL / "results", LEGACY_EVAL / "results_opensource"]
PENDING_NAMES = ["S_models_repeated", "S_physician_verification", "S_per_model_repeated",
                 "S_per_type_repeated", "S_gee_repeated", "S_calibration_repeated", "S_sensitivity"]


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    out = tmp_path_factory.mktemp("tables")
    res = st.write_all(out)
    return out, res


def read(out, name):
    with open(out / f"{name}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_audit_by_type_has_ten_types_and_matches_summary(written):
    out, _ = written
    rows = read(out, "S_audit_by_type")
    summ = json.loads((AUDIT_DIR / "pair_audit_summary.json").read_text())
    assert len(rows) == 10
    assert all(r["root_cause"] for r in rows)
    for col in ("excerpt_differs", "identical", "metadata_only"):
        assert sum(int(r[col]) for r in rows) == summ["visible_status"][col]
    for col in ("visible", "hidden_only", "no_edit"):
        assert sum(int(r[col]) for r in rows) == summ["raw_status"][col]
    assert sum(int(r["pairs"]) for r in rows) == 500


def test_hash_table_sums_to_10500_with_no_mismatch(written):
    out, _ = written
    rows = read(out, "S_hash_verification")
    assert {r["profile"] for r in rows} == set(profiles.PROFILES)
    assert sum(int(r["records"]) for r in rows) == 10500
    assert sum(int(r["verified"]) for r in rows) == 10496
    assert sum(int(r["no_logged_hash"]) for r in rows) == 4
    assert sum(int(r["mismatch"]) for r in rows) == 0
    assert all(r["verification_scope"] for r in rows)


def test_original_pipelines_one_row_per_profile(written):
    out, _ = written
    rows = read(out, "S_original_pipelines")
    assert [r["profile"] for r in rows] == list(profiles.PROFILES)
    for r in rows:
        assert all(r[c] for c in ("models", "tool_presentation", "vignette_fields",
                                  "system_prompt_source", "decoding"))
    by = {r["profile"]: r for r in rows}
    assert by["together_inline6"]["decoding"].startswith("temperature 0.7")
    assert "run_gemini_experiment.py" in by["gemini_inline7"]["system_prompt_source"]


@pytest.mark.skipif(not all(t.exists() for t in RESULT_TREES), reason="legacy results tree not mounted")
def test_run_inventory_covers_every_results_dir(written):
    out, _ = written
    rows = read(out, "S_run_inventory")
    expected = {(t.name, d.name) for t in RESULT_TREES for d in t.iterdir()
                if d.is_dir() and not d.name.startswith("._")}
    assert {(r["results_tree"], r["run_dir"]) for r in rows} == expected
    canon = [r for r in rows if r["canonical"] == "Y"]
    assert len(canon) == 21
    assert sum(int(r["valid_decisions"]) for r in canon) >= 21 * 500
    for r in rows:
        assert r["canonical"] in ("Y", "N")
        if r["canonical"] == "N":
            assert r["reason_not_canonical"]
        assert r["input_file_hash"]
    q = [r for r in rows if "Qwen2.5_1.5B" in r["run_dir"]][0]
    assert q["records"] == "500" and q["valid_decisions"] == "0"
    assert "non-serverless" in q["reason_not_canonical"]


def test_original_reanalysis_labeled_exploratory(written):
    out, _ = written
    rows = read(out, "S_original_reanalysis")
    assert {r["analysis"] for r in rows} == {"exploratory"}
    types = [r for r in rows if r["stratum"] == "sham_type"]
    models = [r for r in rows if r["stratum"] == "model"]
    assert len(types) == 10 and len(models) == 21
    miss = [r for r in types if r["level"] == "missing_warning"][0]
    assert miss["discriminable_n"] == "0"
    assert miss["all_n"] == "840"


def test_modifications_by_type(written):
    out, _ = written
    rows = read(out, "S_modifications_by_type")
    assert len(rows) == 10
    assert sum(int(r["pairs_total"]) for r in rows) == 500
    assert sum(int(r["pending_physician_edit"]) for r in rows) == 23
    for r in rows:
        assert r["transform"] and r["fields_changed"]
        parts = sum(int(r[c]) for c in ("original_transform", "rule_typosquat",
                                        "rule_inversion_extended", "pending_physician_edit"))
        assert parts == int(r["pairs_total"])


@pytest.mark.parametrize("name", PENDING_NAMES)
def test_pending_tables_raise(name, tmp_path):
    with pytest.raises(st.PendingData) as exc:
        st.PENDING_TABLES[name](root=tmp_path)
    assert str(exc.value)


def test_write_all_reports_pending(written):
    out, res = written
    assert set(res["pending"]) == set(PENDING_NAMES)
    assert not any((out / f"{n}.csv").exists() for n in PENDING_NAMES)


def test_no_board_certified_anywhere(written):
    out, _ = written
    for p in out.glob("*.csv"):
        text = p.read_text().lower()
        assert "board-certified" not in text and "board certified" not in text
