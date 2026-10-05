import csv
import json
from collections import Counter

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
    assert sum(int(r["valid_decisions_scored"]) for r in canon) >= 21 * 500
    for r in rows:
        assert r["canonical"] in ("Y", "N")
        if r["canonical"] == "N":
            assert r["reason_not_canonical"]
        assert r["input_file_hash"]
    q = [r for r in rows if "Qwen2.5_1.5B" in r["run_dir"]][0]
    assert q["records"] == "500" and q["valid_decisions_scored"] == "0"
    assert "non-serverless" in q["reason_not_canonical"]


def test_inventory_reasons_attributed_and_not_redundant(written):
    out, _ = written
    for r in read(out, "S_run_inventory"):
        reason = r["reason_not_canonical"]
        if r["canonical"] == "N":
            assert "(per " in reason, r["run_dir"]
            if reason.count("; ") and "21-model" in reason:
                raise AssertionError(reason)
    assert (out / "README.txt").read_text().count("non-null") == 1


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


@pytest.mark.skipif(not LEGACY_EVAL.exists(), reason="legacy scripts not mounted")
def test_system_prompt_cites_point_at_the_prompt_string():
    import re
    for r in st.original_pipelines():
        m = re.match(r"(\S+?):(\d+)-(\d+)", r["system_prompt_source"])
        path, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        lines = (LEGACY_EVAL / path).read_text().splitlines()
        assert "You are a clinical decision support evaluator" in lines[lo - 1], r["profile"]
        assert lines[hi - 1].rstrip().endswith('"""'), r["profile"]
        assert not any("defended" in l.lower() or "POTENTIALLY MALICIOUS" in l for l in lines[lo - 1:hi]), r["profile"]


ALLOWED_SOURCES = {"manifest", "results.jsonl", "logged errors", "run logs", "author record"}


def test_inventory_reasons_use_plain_sources_not_internal_labels(written):
    import re
    out, _ = written
    for r in read(out, "S_run_inventory"):
        reason = r["reason_not_canonical"]
        assert "AUTHOR_FACTS" not in reason, r["run_dir"]
        assert set(re.findall(r"\(per ([^)]*)\)", reason)) <= ALLOWED_SOURCES, reason
    readme = (out / "README.txt").read_text()
    assert "AUTHOR_FACTS" not in readme
    assert "run logs" in readme and "author record" in readme


def test_quote_error_keeps_short_messages_whole():
    assert st._quote_error("Request timed out.") == "Request timed out."
    msg = "x " * 80  # exactly 160 characters
    assert st._quote_error(msg) == msg


def test_quote_error_cuts_long_messages_at_a_word_boundary():
    msg = "404 models/gemini-2.5-flash-lite-preview-06-17 is not found for API version v1beta, or is " \
          "not supported for generateContent. Call ListModels to see the list of available models and " \
          "their supported methods."
    q = st._quote_error(msg)
    assert q.endswith("…") and len(q) <= 161
    stem = q[:-1]
    assert msg.startswith(stem) and msg[len(stem)].isspace() and not stem[-1].isspace()
    assert q == msg[:msg.rfind(" ", 0, 161)] + "…"


def test_quote_error_never_cuts_a_single_long_token():
    token = "a" * 200
    assert st._quote_error(token) == token


@pytest.mark.skipif(not all(t.exists() for t in RESULT_TREES), reason="legacy results tree not mounted")
def test_inventory_quotes_logged_errors_whole_or_at_a_word_boundary(written):
    out, _ = written
    seen = 0
    for r in read(out, "S_run_inventory"):
        reason = r["reason_not_canonical"]
        if "logged error: " not in reason:
            continue
        seen += 1
        quoted = reason.split("logged error: ", 1)[1].split(" (per logged errors)")[0]
        lines = (LEGACY_EVAL / r["results_tree"] / r["run_dir"] / "results.jsonl").read_text().splitlines()
        recs = [json.loads(l) for l in lines if l.strip()]
        fulls = Counter(str(x["errors"][0]) for x in recs if x.get("errors") and not
                        (x.get("model_decision") is not None and x.get("selected_tool_correct") is not None))
        full = fulls.most_common(1)[0][0]
        if quoted.endswith("…"):
            stem = quoted[:-1]
            assert len(full) > 160 and len(stem) <= 160, r["run_dir"]
            assert full.startswith(stem) and full[len(stem)].isspace(), r["run_dir"]
        else:
            assert quoted == full, r["run_dir"]
    assert seen >= 10
