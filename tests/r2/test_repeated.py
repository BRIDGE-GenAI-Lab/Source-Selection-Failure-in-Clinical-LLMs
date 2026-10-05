import json

import pandas as pd

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
    assert pd.isna(s["primacy"]) and pd.isna(s["content_z"]) and pd.isna(s["content_p"])


def test_noticed():
    snippets = ["Most contraindications are relative in emergency settings"]
    assert noticed(["B says most contraindications are relative in emergencies"], snippets)
    assert not noticed(["B has a different year"], snippets)
    assert noticed(["year 2010 vs 2019"], ["2010"])


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
    (verify / "excluded_pairs.json").write_text(json.dumps({"rule": "x", "excluded": [], "reasons": {}}))
    (verify / "flags.json").write_text("{}")
    monkeypatch.setattr(repeated, "VERIFY_DIR", verify)
    monkeypatch.setattr(repeated, "load_specs", lambda *a, **k: [{"original_name": model, "model_id": model}])
    df = repeated_frame(results)
    assert len(df) == 2
    row = df[df.case_id == c2].iloc[0]
    assert row.status == "non_response" and pd.isna(row.correct)
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
    assert pd.isna(pm.loc["arm", "content_p"]) and pd.isna(pm.loc["arm", "content_p_fdr"])
    assert pm.loc[["good", "bad", "nc", "an"], "content_p_fdr"].notna().all()
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
