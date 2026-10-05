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
