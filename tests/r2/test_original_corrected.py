from r2.analysis.original_corrected import run


def test_headline_numbers():
    h = run(write=False)
    assert h["identical"]["n"] == 3507
    assert abs(h["identical"]["accuracy"] - 0.490) < 0.001
    assert abs(h["identical"]["primacy"] - 0.880) < 0.001
    assert h["discriminable"]["n"] == 6993
    assert abs(h["discriminable"]["accuracy"] - 0.6458) < 0.0005
    assert h["all"]["n"] == 10500
    assert abs(h["all"]["primacy"] - 0.7274) < 0.0005
    assert set(h["types_without_valid_pairs"]) == {"contraindication_violation", "missing_warning"}


def test_profile_interaction_is_estimable(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    import r2.analysis.original_corrected as oc

    monkeypatch.setattr(oc, "OUT", tmp_path / "out")  # never write into the committed analysis_original/
    oc.run(write=True)
    OUT = oc.OUT
    for name in ("all", "discriminable"):
        g = pd.read_csv(OUT / f"gee_profile_interaction_{name}.csv", index_col=0)
        row = g.loc["sham_first_x_inline"]
        assert np.isfinite([row["OR"], row["lo"], row["hi"]]).all()


def test_headline_uses_cluster_bootstrap_and_drops_pooled_z():
    from r2.analysis.stats import wilson
    h = run(write=False)
    for name in ("all", "discriminable", "identical"):
        s = h[name]
        assert "content_z" not in s and "content_p" not in s
        assert s["acc_lo"] < s["accuracy"] < s["acc_hi"]
        _, wlo, whi = wilson(round(s["accuracy"] * s["n"]), s["n"])
        assert (s["acc_hi"] - s["acc_lo"]) > 1.5 * (whi - wlo)


def test_per_type_and_by_profile_are_cluster_aware():
    from r2.analysis.original_corrected import _by_profile, _per_type, original_frame
    from r2.analysis.stats import wilson
    df = original_frame()
    sub = df[df.visible_status != "identical"]
    pt = _per_type(sub).set_index("sham_type")
    am = pt.loc["authority_mimicry"]
    _, wlo, whi = wilson(int(am.failures), int(am.n))
    assert am.lo < am.failure_rate < am.hi and (am.hi - am.lo) > 1.5 * (whi - wlo)
    bp = _by_profile(sub)
    assert "content_z" not in bp.columns and "content_p" not in bp.columns


_OLD = {
    "all": {"n": 10500, "accuracy": 0.5938095238095238, "primacy": 0.7274285714285714,
            "acc_sham_first": 0.3674163674163674, "acc_sham_second": 0.8237665578805913,
            "acc_lo": 0.5644738095238095, "acc_hi": 0.6225761904761904,
            "position_averaged_accuracy": 0.5955914626484793, "says_identical": 0.5743809523809524},
    "discriminable": {"n": 6993, "accuracy": 0.6457886457886458, "primacy": 0.6507936507936508,
                      "acc_sham_first": 0.49500998003992014, "acc_sham_second": 0.7974756167527252,
                      "position_averaged_accuracy": 0.6462427983963226, "says_identical": 0.45145145145145144},
    "identical": {"n": 3507, "primacy": 0.8802395209580839,
                  "acc_sham_first": 0.11659192825112108, "acc_sham_second": 0.8769587928032502,
                  "position_averaged_accuracy": 0.4967753605271856, "says_identical": 0.8195038494439693},
}


def test_pooled_position_rates_have_cluster_bootstrap_cis(tmp_path, monkeypatch):
    import r2.analysis.original_corrected as oc
    monkeypatch.setattr(oc, "OUT", tmp_path / "out")
    h = oc.run(write=False)
    for name in ("all", "discriminable", "identical"):
        s = h[name]
        for rate in ("primacy", "acc_sham_first", "acc_sham_second", "says_identical", "position_averaged_accuracy"):
            assert s[f"{rate}_lo"] < s[rate] < s[f"{rate}_hi"], (name, rate)
        for k, v in _OLD[name].items():
            assert s[k] == v, (name, k)


# Pre-Task-3b headline.json (commit 509ef05). Every key it held must keep its value byte for byte.
_HEADLINE_BEFORE_3B_SHA256 = "0a909c12fe953f29a3cbbaf7e3c27a27e1f786f1bcb5f27d0cdb73a793b4f9fb"
_MEASURES = ("accuracy", "primacy", "acc_sham_first", "acc_sham_second", "position_averaged_accuracy")


def test_no_existing_headline_value_changes():
    import hashlib
    import json
    h = run(write=False)
    old = {k: v for k, v in h.items() if k not in ("excluding_coerced", "fabricated_citation_cue_confounded",
                                                   "discriminable_excluding_fabricated_citation")}
    assert hashlib.sha256(json.dumps(old, indent=2).encode()).hexdigest() == _HEADLINE_BEFORE_3B_SHA256


def test_excluding_coerced_block():
    import csv

    from r2.paths import AUDIT_DIR
    with open(AUDIT_DIR / "coerced_choices.csv", newline="") as fh:
        coerced = [r for r in csv.DictReader(fh) if r["class"] != "not_coerced"]
    n_coerced = {"all": len(coerced),
                 "identical": sum(r["visible_status"] == "identical" for r in coerced),
                 "discriminable": sum(r["visible_status"] != "identical" for r in coerced)}
    h = run(write=False)
    ex = h["excluding_coerced"]
    for name in ("all", "discriminable", "identical"):
        b = ex[name]
        assert b["n_coerced_excluded"] == n_coerced[name], name
        assert b["n"] == h[name]["n"] - n_coerced[name], name
        for m in _MEASURES:
            assert b[f"{m}_lo"] < b[m] < b[f"{m}_hi"], (name, m)
    assert ex["all"]["n"] == 10500 - 781 and ex["identical"]["n"] == 3507 - 368
    assert abs(ex["identical"]["primacy"] - 0.866) < 0.0005  # review figure: 86.6%


# ---- Final fix F1: original fabricated_citation results are cue-confounded ----
CUE_KEY = "fabricated_citation_cue_confounded"
# analysis_original/headline.json at 1a9831f (before F1), as written (json.dumps(head, indent=2)).
_HEADLINE_BEFORE_F1_SHA256 = "3a6f4bdf01ba7b349c854c5fe93488790d25dc4245d8e64fdd6b9733efe1a467"


def test_f1_keeps_every_existing_headline_key_byte_for_byte():
    import hashlib
    import json
    h = run(write=False)
    # the 2026-10-18 sensitivity without fabricated_citation follows the cue block (tests/r2/test_original_additions.py)
    assert list(h)[-2:] == [CUE_KEY, "discriminable_excluding_fabricated_citation"]
    old = {k: v for k, v in h.items() if k not in (CUE_KEY, "discriminable_excluding_fabricated_citation")}
    assert hashlib.sha256(json.dumps(old, indent=2).encode()).hexdigest() == _HEADLINE_BEFORE_F1_SHA256


def test_fabricated_citation_cue_confounded_block():
    import json

    import pandas as pd

    from r2.paths import AUDIT_DIR, REPO
    b = run(write=False)[CUE_KEY]
    assert b["sham_type"] == "fabricated_citation" and "cue-confounded" in b["label"]
    assert "exploratory" in b["label"] and "10.9999/fake." in b["note"]
    assert b["n"] == 840 and b["n_cases"] == 40 and b["n_models"] == 21
    tokens = json.loads((AUDIT_DIR / "self_label_tokens.json").read_text(encoding="utf-8"))
    assert b["rows_with_token"] == tokens["modified_tool_payload"]["fake"]["rows"] == 840
    pt = pd.read_csv(REPO / "analysis_original" / "per_type_all.csv", float_precision="round_trip").set_index(
        "sham_type").loc["fabricated_citation"]
    assert b["failures"] == int(pt.failures) == 288
    assert b["failure_rate"] == pt.failure_rate and (b["failure_rate_lo"], b["failure_rate_hi"]) == (pt.lo, pt.hi)
    assert abs(b["accuracy"] + b["failure_rate"] - 1) < 1e-12
    for m in ("accuracy", "failure_rate", "primacy", "acc_sham_first", "acc_sham_second",
              "position_averaged_accuracy"):
        assert b[f"{m}_lo"] < b[m] < b[f"{m}_hi"], m
    assert b["ci"].startswith("case-cluster bootstrap")


def test_committed_headline_has_the_cue_block():
    import json

    from r2.paths import REPO
    committed = json.loads((REPO / "analysis_original" / "headline.json").read_text())
    assert committed == json.loads(json.dumps(run(write=False)))
