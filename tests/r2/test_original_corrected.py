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


def test_profile_interaction_is_estimable():
    import numpy as np
    import pandas as pd

    from r2.analysis.original_corrected import OUT

    run(write=True)
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
            "acc_lo": 0.5644738095238095, "acc_hi": 0.6225761904761904},
    "discriminable": {"n": 6993, "accuracy": 0.6457886457886458, "primacy": 0.6507936507936508,
                      "acc_sham_first": 0.49500998003992014, "acc_sham_second": 0.7974756167527252},
    "identical": {"n": 3507, "primacy": 0.8802395209580839,
                  "acc_sham_first": 0.11659192825112108, "acc_sham_second": 0.8769587928032502},
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
