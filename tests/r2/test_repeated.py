import json

import pandas as pd
import pytest

from r2.analysis import repeated
from r2.analysis.repeated import (_ok, _snippets, noticed, pair_level, per_model_table, repeated_frame,
                                  sensitivity_non_response)


def _df(rule):
    rows = []
    for i in range(10):
        for order in ("TS", "ST"):
            sham_first = order == "ST"
            sel_first = rule(sham_first)
            rows.append({"model": "m", "case_id": f"C{i}", "order": order, "status": "ok",
                         "sham_first": sham_first, "selected_first": sel_first,
                         "correct": int(sel_first != sham_first)})
    return pd.DataFrame(rows)


def test_pair_level_positional_vs_content():
    pos = pair_level(_df(lambda sf: True))
    assert pos.invariant_correct.mean() == 0 and pos.position_driven.mean() == 1
    good = pair_level(_df(lambda sf: not sf))
    assert good.invariant_correct.mean() == 1 and good.position_driven.mean() == 0


def test_non_response_sensitivity():
    df = _df(lambda sf: not sf)
    df = df.astype({"selected_first": object, "correct": object})  # pandas refuses None in bool/int columns
    df.loc[0, ["status", "correct", "selected_first"]] = ["non_response", None, None]
    s = sensitivity_non_response(df)
    assert s["n"] == 20 and abs(s["accuracy"] - 19 / 20) < 1e-9
    assert pd.isna(s["primacy"]) and "content_z" not in s and "content_p" not in s


def test_noticed():
    snippets = ["Most contraindications are relative in emergency settings"]
    assert noticed(["B says most contraindications are relative in emergencies"], snippets) is True
    assert noticed(["B has a different year"], snippets) is False
    # M10: a snippet shorter than 3 words cannot be matched by the 3-word overlap rule: not assessable
    assert noticed(["year 2010 vs 2019"], ["2010"]) is None


def test_noticed_three_word_rule_is_applied_consistently():
    # a 2-token snippet (e.g. "IDSA 2014") used to be unmatchable and scored "not noticed"
    assert noticed(["A cites IDSA 2014, B cites IDSA 2019"], ["IDSA 2014"]) is None
    assert noticed([], ["ESC 2020"]) is None
    # a 1-token snippet (a year) no longer matches any mention of that year
    assert noticed(["published in 2014"], ["2014", "2019"]) is None
    # exactly 3 tokens is assessable: "v1.0-archived" -> v1, 0, archived
    assert noticed(["B is labelled v1.0-archived"], ["v1.0-archived"]) is True
    assert noticed(["B is older"], ["v1.0-archived"]) is False
    # short snippets are ignored when a longer one makes the response assessable
    assert noticed(["no difference"], ["2014", "Historical guideline version for reference only"]) is False
    assert noticed(["B: historical guideline version"], ["2014", "Historical guideline version"]) is True
    # no differences listed for an assessable edit is "not noticed"
    assert noticed([], ["Most contraindications are relative"]) is False


def test_repeated_frame_tolerates_error_cells(tmp_path, monkeypatch):
    from r2.paths import DATA_V2
    cases = json.loads((DATA_V2 / "pairs_v2.json").read_text())["cases"]
    c1, c2 = cases[0]["case_id"], cases[1]["case_id"]
    model = "model_x"
    results = tmp_path / "results"
    results.mkdir()
    ok = {"original_name": model, "case_id": c1, "order": "TS", "labels": {"A": "T", "B": "S"},
          "status": "ok", "selected_tool": "A", "correct": 1,
          "decision": {"confidence": 0.9, "trust_rationale": "r", "differences_found": ["d"]},
          "model_served": "m"}
    err = {"original_name": model, "case_id": c2, "order": "ST", "labels": {"A": "S", "B": "T"},
           "status": "non_response", "selected_tool": None, "correct": None, "decision": None,
           "error": "boom", "model_served": None}
    (results / f"{model}.jsonl").write_text(json.dumps(ok) + "\n" + json.dumps(err) + "\n")
    verify = tmp_path / "verify"
    verify.mkdir()
    (verify / "excluded_pairs.json").write_text(json.dumps({"rule": "x", "excluded": [c2],
                                                            "reasons": {c2: ["q5_edit_visible"]}}))
    (verify / "flags.json").write_text(json.dumps({c1: ["q8_leakage"]}))
    monkeypatch.setattr(repeated, "VERIFY_DIR", verify)
    monkeypatch.setattr(repeated, "load_specs", lambda *a, **k: [{"original_name": model, "model_id": model}])
    df = repeated_frame(results)
    assert len(df) == 2
    row = df[df.case_id == c2].iloc[0]
    assert row.status == "non_response" and pd.isna(row.correct)
    assert bool(row.excluded) and not bool(row.flagged)
    assert pd.isna(row.noticed)  # no response: E1 is not assessable, never "not noticed"
    ok_row = df[df.case_id == c1].iloc[0]
    assert not bool(ok_row.excluded) and bool(ok_row.flagged)
    assert ok_row.noticed == 0.0  # "d" shares no 3-word sequence with the edit
    assert df.noticed.dtype == float
    assert len(_ok(df)) == 1


def test_snippets_mixed_str_and_list():
    case = {"edit_record": {"changes": [{"before": "abc def ghi",
                                         "after": ["abc def ghi", "new bullet text here"]}]}}
    sn = _snippets(case)
    assert "new bullet text here" in sn
    assert all(len(x) > 1 for x in sn)


def _model_frame(model, content_rule, n=40, conf=0.8):
    rows = []
    for i in range(n):
        for sf in (True, False):
            sel = content_rule(sf)
            rows.append({"model": model, "case_id": f"{model}{i}", "sham_first": sf,
                         "selected_first": sel, "correct": int(sel != sf), "confidence": conf})
    return pd.DataFrame(rows)


def test_per_model_table_nan_p_does_not_poison_fdr_and_calibration_drops_nan():
    good = _model_frame("good", lambda sf: not sf)
    bad = _model_frame("bad", lambda sf: True)
    one_arm = _model_frame("arm", lambda sf: True)
    one_arm = one_arm[one_arm.sham_first]
    nanconf = _model_frame("nc", lambda sf: not sf)
    nanconf.loc[nanconf.index[:10], "confidence"] = float("nan")
    allnan = _model_frame("an", lambda sf: not sf, conf=float("nan"))
    pm = per_model_table(pd.concat([good, bad, one_arm, nanconf, allnan])).set_index("model")
    assert pd.isna(pm.loc["arm", "mcnemar_p"]) and pd.isna(pm.loc["arm", "mcnemar_p_fdr"])
    assert pd.isna(pm.loc["bad", "mcnemar_p"]) and pd.isna(pm.loc["bad", "mcnemar_p_fdr"])
    assert pm.loc[["good", "nc", "an"], "mcnemar_p_fdr"].notna().all()
    assert abs(pm.loc["nc", "mean_conf"] - 0.8) < 1e-9
    assert pd.isna(pm.loc["an", "ece"]) and pd.isna(pm.loc["an", "mean_conf"])


def test_repeated_frame_uses_last_record_per_cell(tmp_path, monkeypatch):
    from r2.paths import DATA_V2
    c1 = json.loads((DATA_V2 / "pairs_v2.json").read_text())["cases"][0]["case_id"]
    results = tmp_path / "results"
    results.mkdir()
    err = {"original_name": "model_x", "case_id": c1, "order": "TS", "labels": {"A": "T", "B": "S"},
           "status": "non_response", "selected_tool": None, "correct": None, "decision": None,
           "error": "BadRequestError: boom", "attempts": [], "model_served": None}
    ok = {**err, "status": "ok", "selected_tool": "A", "correct": 1, "attempts": [{}],
          "decision": {"confidence": 0.9, "differences_found": []}}
    del ok["error"]
    (results / "model_x.jsonl").write_text(json.dumps(err) + "\n" + json.dumps(ok) + "\n")
    verify = tmp_path / "verify"
    verify.mkdir()
    (verify / "excluded_pairs.json").write_text(json.dumps({"rule": "x", "excluded": [], "reasons": {}}))
    (verify / "flags.json").write_text("{}")
    monkeypatch.setattr(repeated, "VERIFY_DIR", verify)
    monkeypatch.setattr(repeated, "load_specs", lambda *a, **k: [{"original_name": "model_x", "model_id": "model_x"}])
    df = repeated_frame(results)
    assert len(df) == 1 and df.iloc[0].status == "ok" and df.iloc[0].correct == 1


def test_repeated_frame_reads_only_current_spec_files(tmp_path, monkeypatch):
    from r2.paths import DATA_V2
    c1 = json.loads((DATA_V2 / "pairs_v2.json").read_text())["cases"][0]["case_id"]
    results = tmp_path / "results"
    results.mkdir()
    ok = {"original_name": "model_x", "case_id": c1, "order": "TS", "labels": {"A": "T", "B": "S"},
          "status": "ok", "selected_tool": "A", "correct": 1, "attempts": [{}],
          "decision": {"confidence": 0.9, "differences_found": []}, "model_served": "m"}
    (results / "model_x.jsonl").write_text(json.dumps(ok) + "\n")
    (results / "dropped.jsonl").write_text(json.dumps({**ok, "original_name": "model_x"}) + "\n")
    verify = tmp_path / "verify"
    verify.mkdir()
    (verify / "excluded_pairs.json").write_text(json.dumps({"rule": "x", "excluded": [], "reasons": {}}))
    (verify / "flags.json").write_text("{}")
    monkeypatch.setattr(repeated, "VERIFY_DIR", verify)
    monkeypatch.setattr(repeated, "load_specs", lambda *a, **k: [{"original_name": "model_x", "model_id": "model_x"},
                                                                 {"original_name": "y", "model_id": "missing"}])
    assert len(repeated_frame(results)) == 1


# ---- M12: paired (McNemar) per-model content test; I7: cluster-aware pooled outputs ----
def test_mcnemar_pure_content_model():
    pm = per_model_table(_model_frame("good", lambda sf: not sf, n=12)).set_index("model")
    row = pm.loc["good"]
    assert row.mcnemar_n10 == 12 and row.mcnemar_n01 == 0 and row.mcnemar_p < 0.001
    assert "content_z" not in pm.columns and "content_p" not in pm.columns


def test_mcnemar_positional_model_is_nan():
    pm = per_model_table(_model_frame("pos", lambda sf: True, n=12)).set_index("model")
    row = pm.loc["pos"]
    assert row.mcnemar_n10 == 0 and row.mcnemar_n01 == 0 and pd.isna(row.mcnemar_p)


def test_mcnemar_counts_wrong_in_both_orders():
    df = _model_frame("rev", lambda sf: sf, n=5)
    row = per_model_table(df).set_index("model").loc["rev"]
    assert row.mcnemar_n10 == 0 and row.mcnemar_n01 == 5 and row.mcnemar_p < 0.1


def _synthetic_frame():
    rows = []
    for m, rule in (("good", lambda sf: not sf), ("pos", lambda sf: True)):
        for i in range(30):
            for order in ("TS", "ST"):
                sf = order == "ST"
                sel = rule(sf) if i % 4 else (not rule(sf))
                rows.append({"model": m, "case_id": f"C{i}", "order": order, "status": "ok", "sham_first": sf,
                             "selected_first": sel, "correct": int(sel != sf), "confidence": 0.7,
                             "differences": [], "sham_type": "dosing_error" if i % 2 else "outdated_version",
                             "excluded": i == 0, "flagged": i == 1, "substituted": False, "noticed": False})
    df = pd.DataFrame(rows)
    df["category"] = df.sham_type.map(repeated.CATEGORY)
    return df


def test_run_headline_has_cluster_cis_and_no_pooled_z(monkeypatch):
    monkeypatch.setattr(repeated, "repeated_frame", lambda *a, **k: _synthetic_frame())
    head = repeated.run(write=False)
    prim = head["primary"]
    assert "content_z" not in prim and "content_p" not in prim
    assert prim["primacy_lo"] <= prim["primacy"] <= prim["primacy_hi"]
    for k in ("invariant_correct", "position_driven", "wrong_both"):
        assert head[f"{k}_lo"] <= head[k] <= head[f"{k}_hi"]
    sens = repeated.sensitivity_summaries(_synthetic_frame())
    for s in sens.values():
        assert "content_z" not in s and "content_p" not in s


def test_rate_table_uses_cluster_bootstrap():
    from r2.analysis.stats import wilson
    analysis = _ok(_synthetic_frame())
    t = repeated.rate_table(analysis, "sham_type").set_index("sham_type")
    for _, r in t.iterrows():
        _, wlo, whi = wilson(int(r.failures), int(r.n))
        assert r.lo <= r.failure_rate <= r.hi and (r.lo, r.hi) != (wlo, whi)
    assert "content_z" not in t.columns and "content_p" not in t.columns


# ---- deferred Task 11 minors ----
PAIR_COLS = ["model", "case_id", "invariant_correct", "position_driven", "wrong_both"]


def test_pair_level_empty_frame_and_no_complete_pairs():
    empty = pair_level(pd.DataFrame())
    assert list(empty.columns) == PAIR_COLS and empty.empty
    one_order = _df(lambda sf: True)
    one_order = one_order[one_order.order == "TS"]
    out = pair_level(one_order)
    assert list(out.columns) == PAIR_COLS and out.empty
    non_ok = _df(lambda sf: True).assign(status="non_response")
    assert pair_level(non_ok).empty


def test_write_json_turns_nan_and_inf_into_null(tmp_path):
    import numpy as np
    path = tmp_path / "x.json"
    repeated.write_json(path, {"a": float("nan"), "b": [np.float64("nan"), 1.5], "c": {"d": np.inf},
                               "e": np.int64(3), "f": np.bool_(True)})

    def refuse(token):
        raise ValueError(f"invalid JSON constant {token}")
    data = json.loads(path.read_text(), parse_constant=refuse)
    assert data == {"a": None, "b": [None, 1.5], "c": {"d": None}, "e": 3, "f": True}


def test_non_response_sensitivity_emits_no_warning():
    import warnings
    df = _df(lambda sf: not sf).astype({"selected_first": object, "correct": object})
    df.loc[0, ["status", "correct", "selected_first"]] = ["non_response", None, None]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = sensitivity_non_response(df)
    assert abs(s["accuracy"] - 19 / 20) < 1e-9


def _verify_dir(tmp_path, excluded="{\"rule\": \"x\", \"excluded\": [], \"reasons\": {}}", flags="{}"):
    verify = tmp_path / "verify"
    verify.mkdir(exist_ok=True)
    (verify / "excluded_pairs.json").write_text(excluded)
    (verify / "flags.json").write_text(flags)
    return verify


@pytest.mark.parametrize("flags", ['["CASE_001"]', '{"CASE_001": "q8_leakage"}', '{"CASE_001": []}',
                                   '{"CASE_001": ["q5_edit_visible"]}', '{"NOT_A_CASE": ["q8_leakage"]}'])
def test_flags_json_shape_is_validated(tmp_path, flags):
    from r2.paths import DATA_V2
    pairs = {c["case_id"] for c in json.loads((DATA_V2 / "pairs_v2.json").read_text())["cases"]}
    with pytest.raises(ValueError, match="flags.json"):
        repeated.load_verification(_verify_dir(tmp_path, flags=flags), pairs)


@pytest.mark.parametrize("excluded", ['{"excluded": "CASE_001"}', '{"rule": "x"}', '{"rule": "x", "excluded": ["NOPE"]}'])
def test_excluded_pairs_shape_is_validated(tmp_path, excluded):
    from r2.paths import DATA_V2
    pairs = {c["case_id"] for c in json.loads((DATA_V2 / "pairs_v2.json").read_text())["cases"]}
    with pytest.raises(ValueError, match="excluded_pairs.json"):
        repeated.load_verification(_verify_dir(tmp_path, excluded=excluded), pairs)


def test_flag_items_match_score_review():
    from r2.verify.score_review import flag_reasons
    every_flag = {"q1_vignette_consistent": "N", "q2_question_answerable": "N", "q3_authentic_concordant": "P",
                  "q8_leakage": "Y"}
    assert set(flag_reasons(every_flag)) == set(repeated.FLAG_ITEMS)


def test_run_writes_calibration_per_model_and_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(repeated, "repeated_frame", lambda *a, **k: _synthetic_frame())
    out = tmp_path / "analysis_repeated"
    repeated.run(write=True, analysis_dir=out)
    cal = pd.read_csv(out / "calibration_per_model.csv")
    assert sorted(cal.model) == ["good", "pos"]
    for col in ("n_conf", "ece", "mce", "brier", "auroc", "mean_conf", "accuracy_conf"):
        assert col in cal.columns, col

    def refuse(token):
        raise ValueError(token)
    for p in out.glob("*.json"):
        json.loads(p.read_text(), parse_constant=refuse)  # no NaN / Infinity tokens anywhere
    conc = json.loads((out / "exploratory_concordance_original.json").read_text())
    assert conc["n_models"] == 0 and conc["spearman_rho"] is None and conc["p"] is None
