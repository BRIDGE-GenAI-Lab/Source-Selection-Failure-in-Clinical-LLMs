import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from r2 import profiles
from r2.analysis import supp_tables as st
from r2.paths import AUDIT_DIR, LEGACY_EVAL

RESULT_TREES = [LEGACY_EVAL / "results", LEGACY_EVAL / "results_opensource"]
PENDING_NAMES = ["S_models_repeated", "S_physician_verification", "S_per_model_repeated",
                 "S_per_type_repeated", "S_gee_repeated", "S_calibration_repeated", "S_sensitivity"]


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    # the repeated-experiment inputs come from an empty sandbox root, so the pre-results expectations below
    # hold whether or not the real verification/ and analysis_repeated/ outputs exist yet (review M6)
    out = tmp_path_factory.mktemp("tables")
    res = st.write_all(out, root=tmp_path_factory.mktemp("no_repeated_inputs"))
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
        if "logged the error: " not in reason:
            continue
        seen += 1
        quoted = reason.split("logged the error: ", 1)[1].split("; ")[0]
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


# ---------------------------------------------------------------- invalid-reply handling and coerced choices

def test_original_pipelines_state_invalid_reply_handling(written):
    out, _ = written
    by = {r["profile"]: r for r in read(out, "S_original_pipelines")}
    assert all(by[p]["invalid_reply_handling"] for p in profiles.PROFILES)
    assert "unclear" in by["openai_fc"]["invalid_reply_handling"]
    for p in profiles.PROFILES:
        if p != "openai_fc":
            assert "records A" in by[p]["invalid_reply_handling"], p
    assert by["deepseek_reasoner_inline7"]["decoding"] == "max_tokens 8192; no temperature parameter (provider default)"


def test_invalid_reply_cites_point_at_the_fallback_code():
    import re
    for r in st.original_pipelines():
        text = r["invalid_reply_handling"]
        if r["profile"] == "openai_fc":
            lines = (st.REPO / "src/runner.py").read_text().splitlines()
            assert "raise ValueError" in lines[435 - 1]
            assert 'setdefault("selected_tool", "unclear")' in lines[448 - 1]
            continue
        path, lo, hi = re.search(r"(scripts/\S+\.py):(\d+)-(\d+)", text).groups()
        lines = (st.REPO / path).read_text().splitlines()
        assert 'json_start = response_text.find("{")' in lines[int(lo) - 1], r["profile"]
        assert 'decision["selected_tool"] = selected' in lines[int(hi) - 1], r["profile"]
        block = "\n".join(lines[int(lo) - 1:int(hi)])
        assert '"selected_tool": "A"' in block and 'selected = "A"' in block, r["profile"]


def test_coerced_choices_table_matches_the_audit_summary(written):
    out, _ = written
    rows = read(out, "S_coerced_choices")
    summary = json.loads((AUDIT_DIR / "coerced_choices_summary.json").read_text())
    assert len(rows) == 21 and sum(int(r["records"]) for r in rows) == summary["n_records"]
    tot = lambda col: sum(int(r[col]) for r in rows)
    assert tot("coerced_total") == summary["coerced_total"]
    assert tot("coerced_no_json") + tot("coerced_json_decode_error") == summary["class_counts"]["coerced_parse_failure"]
    assert tot("coerced_invalid_choice") == summary["class_counts"]["coerced_invalid_choice"]
    ident = summary["by_visible_status"]["identical"]
    assert tot("coerced_on_identical_pairs") == ident.get("coerced_parse_failure", 0) + ident.get("coerced_invalid_choice", 0)
    assert all(int(r["coerced_total"]) == 0 for r in rows if r["profile"] != "together_inline6")
    for r in rows:
        assert int(r["coerced_total"]) + int(r["not_coerced"]) == int(r["records"])
        assert int(r["parse_failures_at_token_limit"]) <= int(r["coerced_no_json"]) + int(r["coerced_json_decode_error"])


def test_coerced_parse_failures_at_the_token_limit(written):  # from audit/coerced_choices.csv; no volume needed
    out, _ = written
    rows = {r["model"]: r for r in read(out, "S_coerced_choices")}
    assert sum(int(r["parse_failures_at_token_limit"]) for r in rows.values()) == 381
    q = rows["Qwen/Qwen3-Next-80B-A3B-Thinking"]
    assert int(q["parse_failures_at_token_limit"]) == int(q["coerced_no_json"]) + int(q["coerced_json_decode_error"]) == 329
    assert rows["Qwen/Qwen3-Next-80B-A3B-Thinking"]["runner_max_tokens"] == "2048"
    assert rows["gpt-4.1-2025-04-14"]["runner_max_tokens"] == "not set"


def test_coerced_choices_reads_token_limit_from_the_audit_csv_not_raw_files(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "LEGACY_EVAL", tmp_path / "not-mounted")
    monkeypatch.setattr(st, "LEGACY_RESULTS", tmp_path / "not-mounted" / "results", raising=False)
    rows = {r["model"]: r for r in st.coerced_choices()}
    assert sum(r["parse_failures_at_token_limit"] for r in rows.values()) == 381
    assert rows["Qwen/Qwen3-Next-80B-A3B-Thinking"]["parse_failures_at_token_limit"] == 329
    assert rows["ServiceNow-AI/Apriel-1.6-15b-Thinker"]["parse_failures_at_token_limit"] == 37
    assert rows["deepseek-reasoner"]["runner_max_tokens"] == 8192
    assert rows["gemini-2.5-flash-lite"]["runner_max_tokens"] == "not set"


@pytest.mark.skipif(not all(t.exists() for t in RESULT_TREES), reason="legacy results tree not mounted")
def test_inventory_error_counts_come_from_the_logs(written):
    out, _ = written
    rows = {r["run_dir"]: r for r in read(out, "S_run_inventory")}
    for tree in RESULT_TREES:
        for d in tree.iterdir():
            if d.name.startswith("._") or not d.is_dir():
                continue
            recs = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()] \
                if (d / "results.jsonl").exists() else []
            n429 = sum(1 for r in recs if r.get("model_decision") is None
                       and str((r.get("errors") or [""])[0]).startswith("429"))
            row = rows[d.name]
            if n429:
                assert f"(run logs) {n429} of {len(recs)} rows returned 429 rate-limit errors" in \
                    row["reason_not_canonical"], d.name
            n_err = sum(1 for r in recs if r.get("model_decision") is None and r.get("errors"))
            if row["canonical"] == "N" and n_err:
                assert f"(run logs) {n_err} of {len(recs)} rows" in row["reason_not_canonical"], d.name
    g = rows["run_20260107_181353"]["reason_not_canonical"]
    assert "28 of 59 rows returned 429 rate-limit errors" in g and "no stop reason logged" in g
    assert "74 of 116 rows returned 429" in rows["run_20260107_181634"]["reason_not_canonical"]


# ---- Task 12b: the repeated-experiment tables, built from a synthetic run ----
REPEATED_COLUMNS = {
    "S_models_repeated": ["model", "requested_model_id", "served_model_id", "provider", "reasoning", "decoding",
                          "substituted_for", "availability", "ok", "non_responses", "errors"],
    "S_physician_verification": ["item", "question", "role", "agreement", "kappa", "pairs_excluded",
                                 "pairs_flagged", "pairs_reviewed"],
    "S_sensitivity": ["analysis", "description", "n", "n_models", "n_pairs", "position_averaged_accuracy",
                      "paa_lo", "paa_hi", "accuracy", "acc_lo", "acc_hi", "primacy", "primacy_lo", "primacy_hi",
                      "acc_sham_first", "acc_sham_first_lo", "acc_sham_first_hi", "acc_sham_second",
                      "acc_sham_second_lo", "acc_sham_second_hi", "excluded_models"],
    "S_per_type_repeated": ["stratum", "level", "category", "n", "n_pairs", "failures", "failure_rate", "lo", "hi"],
    "S_gee_repeated": ["term", "label", "OR", "lo", "hi", "p"],
    "S_per_model_repeated": ["model", "n", "acc_sham_first", "acc_sham_first_lo", "acc_sham_first_hi",
                             "n_sham_first", "acc_sham_second", "acc_sham_second_lo", "acc_sham_second_hi",
                             "n_sham_second", "primacy", "primacy_lo", "primacy_hi", "position_averaged_accuracy",
                             "paa_lo", "paa_hi", "n_pair_units", "invariant_correct", "position_driven",
                             "wrong_both", "mcnemar_n10", "mcnemar_n01", "mcnemar_p", "mcnemar_p_fdr"],
    "S_calibration_repeated": ["scope", "model", "n_conf", "ece", "mce", "brier", "auroc", "mean_conf",
                               "accuracy_conf"],
}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from r2.analysis import repeated, synthetic
    root = tmp_path_factory.mktemp("synth")
    world = synthetic.build_world(root)
    repeated.run(world["runs_main"], analysis_dir=world["analysis_dir"], pairs_path=world["pairs_path"],
                 verify_dir=world["verify_dir"], specs=world["specs"])
    return world


def _csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_repeated_tables_read_the_paths_their_writers_use():
    import inspect
    from r2.analysis import repeated
    from r2.paths import REPO
    from r2.run import check_complete, probe_models
    from r2.verify import score_review
    from r2 import paths
    assert st.ANALYSIS_REPEATED == repeated.OUT == paths.ANALYSIS_REPEATED == REPO / "analysis_repeated"
    assert st.VERIFY == score_review.VERIFY_DIR
    assert check_complete.ANALYSIS_REPEATED == probe_models.ANALYSIS_REPEATED == paths.ANALYSIS_REPEATED
    assert 'ANALYSIS_REPEATED / "completeness.csv"' in inspect.getsource(check_complete.main)
    assert 'ANALYSIS_REPEATED / "model_availability.csv"' in inspect.getsource(probe_models.main)
    for name in ("kappa.csv", "excluded_pairs.json", "flags.json", "final_review.csv"):
        assert f'"{name}"' in inspect.getsource(score_review)


@pytest.mark.parametrize("name", PENDING_NAMES)
def test_repeated_table_from_a_synthetic_run(name, synth):
    rows = st.PENDING_TABLES[name](root=synth["root"])
    assert rows and list(rows[0]) == REPEATED_COLUMNS[name]


def test_models_repeated_matches_specs_and_completeness(synth):
    from r2.run.harness import load_specs
    rows = st.PENDING_TABLES["S_models_repeated"](root=synth["root"])
    specs = load_specs()
    assert [r["model"] for r in rows] == [s["original_name"] for s in specs]
    comp = {r["model"]: r for r in _csv(synth["analysis_dir"] / "completeness.csv")}
    for r, s in zip(rows, specs):
        assert r["requested_model_id"] == s["model_id"] and r["served_model_id"] == comp[r["model"]]["model_served"]
        assert r["non_responses"] == int(comp[r["model"]]["non_response"])
        assert r["reasoning"] == ("Yes" if s["reasoning"] else "No")
        assert ("provider default" in r["decoding"]) == (s["temperature"] is None)


def test_physician_verification_counts(synth):
    rows = st.PENDING_TABLES["S_physician_verification"](root=synth["root"])
    excluded = json.loads((synth["verify_dir"] / "excluded_pairs.json").read_text())
    flags = {k: v for k, v in json.loads((synth["verify_dir"] / "flags.json").read_text()).items() if k != "synthetic"}
    by = {r["item"]: r for r in rows}
    assert by["any item"]["pairs_excluded"] == len(excluded["excluded"])
    assert by["any item"]["pairs_flagged"] == len(flags)
    assert by["any item"]["pairs_reviewed"] == len(_csv(synth["verify_dir"] / "final_review.csv"))
    assert by["q5_edit_visible"]["pairs_excluded"] == len(excluded["excluded"])
    assert by["q8_leakage"]["pairs_flagged"] == len(flags) and by["q8_leakage"]["pairs_excluded"] == ""
    kappa = {r["item"]: r for r in _csv(synth["verify_dir"] / "kappa.csv")}
    assert all(by[i]["kappa"] == float(kappa[i]["kappa"]) for i in kappa)


def test_sensitivity_table_has_primary_and_s1_to_s5(synth):
    rows = st.PENDING_TABLES["S_sensitivity"](root=synth["root"])
    assert [r["analysis"] for r in rows] == ["primary", "S1_non_response_incorrect", "S2_including_excluded",
                                            "S3_excluding_flagged", "S4_excluding_substituted",
                                            "S5_excluding_generator_family"]
    head = json.loads((synth["analysis_dir"] / "headline.json").read_text())
    assert rows[0]["position_averaged_accuracy"] == head["primary"]["position_averaged_accuracy"]
    assert rows[-1]["excluded_models"] == "gemini-2.5-flash-lite"
    for r in rows:
        assert r["paa_lo"] <= r["position_averaged_accuracy"] <= r["paa_hi"]


def test_gee_table_keeps_order_and_category_terms_only(synth):
    rows = st.PENDING_TABLES["S_gee_repeated"](root=synth["root"])
    assert [r["term"] for r in rows] == ["sham_first[T.True]",
                                        "C(category, Treatment('metadata'))[T.clinical_safety]",
                                        "C(category, Treatment('metadata'))[T.injection]",
                                        "C(category, Treatment('metadata'))[T.semantic]"]
    assert rows[0]["label"] == "Modified excerpt shown first (vs second)"


def test_per_type_and_calibration_tables_stack_their_sources(synth):
    t = st.PENDING_TABLES["S_per_type_repeated"](root=synth["root"])
    assert {r["stratum"] for r in t} == {"modification type", "category"}
    assert sum(r["stratum"] == "modification type" for r in t) == len(_csv(synth["analysis_dir"] / "per_type.csv"))
    cal = st.PENDING_TABLES["S_calibration_repeated"](root=synth["root"])
    assert cal[0]["scope"] == "overall" and cal[0]["model"] == ""
    assert len(cal) == 1 + len(_csv(synth["analysis_dir"] / "calibration_per_model.csv"))


def test_missing_single_input_names_it(synth, tmp_path):
    import shutil
    root = tmp_path / "partial"
    shutil.copytree(synth["root"] / "analysis_repeated", root / "analysis_repeated")
    (root / "analysis_repeated" / "calibration_per_model.csv").unlink()
    with pytest.raises(st.PendingData, match="calibration_per_model.csv"):
        st.PENDING_TABLES["S_calibration_repeated"](root=root)


def test_write_repeated_writes_every_table(synth, tmp_path):
    res = st.write_repeated(tmp_path, root=synth["root"])
    assert set(res["written"]) == set(PENDING_NAMES) and not res["pending"]
    for name in PENDING_NAMES:
        assert _csv(tmp_path / f"{name}.csv")



# ---- fix round 1, M4 / M5 ----
def test_sensitivity_table_fails_on_a_missing_key(synth, tmp_path):
    import shutil
    root = tmp_path / "r"
    shutil.copytree(synth["root"] / "analysis_repeated", root / "analysis_repeated")
    path = root / "analysis_repeated" / "sensitivity.json"
    sens = json.loads(path.read_text())
    del sens["S3_excluding_flagged"]["acc_sham_first_lo"]
    path.write_text(json.dumps(sens))
    with pytest.raises(KeyError, match="acc_sham_first_lo"):
        st.PENDING_TABLES["S_sensitivity"](root=root)


def test_write_csv_refuses_an_empty_table(tmp_path):
    with pytest.raises(ValueError, match="no rows"):
        st.write_csv(tmp_path / "S_empty.csv", [])
    assert not (tmp_path / "S_empty.csv").exists()


def test_expected_outputs_key_every_repeated_table_column():
    exp = json.loads((Path(st.__file__).parent / "expected_outputs.json").read_text())
    for name, cols in REPEATED_COLUMNS.items():
        spec = exp["outputs"][f"tables/{name}.csv"]
        assert spec["type"] == "csv" and list(spec["columns"]) == cols, name


def test_fabricated_citation_transform_discloses_the_self_label_token():
    """Final fix F1: the original DOI suffix read 'fake'; the corrected set replaces it."""
    text = st.TRANSFORMS["fabricated_citation"][0]
    assert "10.9999/fake.NNNN" in text and "10.9999/cg.<year>.NNNN" in text and "fake DOI" not in text
