import numpy as np
import pandas as pd

from r2.analysis.stats import calibration, cluster_bootstrap_paa, position_summary, wilson


def _frame(rule, n_cases=200):
    rows = []
    for i in range(n_cases):
        for sham_first in (True, False):
            sel_first = rule(sham_first)
            correct = int(sel_first != sham_first)
            rows.append({"case_id": f"C{i}", "sham_first": sham_first,
                         "selected_first": sel_first, "correct": correct})
    return pd.DataFrame(rows)


def test_wilson_matches_published_ci():
    p, lo, hi = wilson(6235, 10500)
    assert round(p, 4) == 0.5938 and round(lo, 3) == 0.584 and round(hi, 3) == 0.603


def test_pure_positional_chooser():
    s = position_summary(_frame(lambda sham_first: True))
    assert s["primacy"] == 1.0 and s["position_averaged_accuracy"] == 0.5
    assert s["content_z"] == 0.0 and s["content_p"] == 1.0


def test_perfect_content_chooser():
    s = position_summary(_frame(lambda sham_first: not sham_first))
    assert s["position_averaged_accuracy"] == 1.0 and s["primacy"] == 0.5
    assert s["content_z"] > 0 and s["content_p"] < 0.05


def test_content_z_sign_for_reverse_chooser():
    s = position_summary(_frame(lambda sham_first: sham_first))
    assert s["position_averaged_accuracy"] == 0.0
    assert s["content_z"] < 0


def test_calibration_perfect_and_auroc():
    conf = np.array([0.9] * 10 + [0.1] * 10)
    correct = np.array([1] * 9 + [0] + [0] * 9 + [1])
    c = calibration(conf, correct)
    assert abs(c["ece"]) < 1e-9 and c["auroc"] > 0.8


def test_cluster_bootstrap_brackets_point_estimate():
    df = _frame(lambda sham_first: not sham_first)
    df.loc[df.index % 5 == 0, "correct"] = 0
    lo, hi = cluster_bootstrap_paa(df, n_boot=300)
    point = position_summary(df)["position_averaged_accuracy"]
    assert lo <= point <= hi


def test_cluster_bootstrap_one_arm_returns_nan():
    df = _frame(lambda sham_first: not sham_first)
    lo, hi = cluster_bootstrap_paa(df[df.sham_first], n_boot=50)
    assert np.isnan(lo) and np.isnan(hi)


def test_position_summary_one_orientation_is_nan_not_error():
    df = _frame(lambda sf: True)
    df = df[df.sham_first]
    s = position_summary(df)
    assert s["n"] == len(df) and s["primacy"] == 1.0
    for k in ("acc_sham_second", "position_averaged_accuracy", "content_z", "content_p"):
        assert np.isnan(s[k])


# ---- I7: case-cluster bootstrap for pooled rates ----
def test_cluster_bootstrap_mean_brackets_and_is_deterministic():
    from r2.analysis.stats import cluster_bootstrap_mean
    df = _frame(lambda sf: not sf)
    df.loc[df.index % 3 == 0, "correct"] = 0
    lo, hi = cluster_bootstrap_mean(df, "correct", n_boot=500)
    assert lo < df.correct.mean() < hi
    assert (lo, hi) == cluster_bootstrap_mean(df, "correct", n_boot=500)


def test_cluster_bootstrap_mean_empty_is_nan():
    from r2.analysis.stats import cluster_bootstrap_mean
    lo, hi = cluster_bootstrap_mean(pd.DataFrame({"case_id": [], "correct": []}), "correct")
    assert np.isnan(lo) and np.isnan(hi)


def test_cluster_bootstrap_mean_widens_for_clustered_data():
    from r2.analysis.stats import cluster_bootstrap_mean
    rng = np.random.default_rng(1)
    case_val = rng.integers(0, 2, size=30)
    df = pd.DataFrame([{"case_id": f"C{i}", "correct": int(v)} for i, v in enumerate(case_val) for _ in range(21)])
    lo, hi = cluster_bootstrap_mean(df, "correct")
    _, wlo, whi = wilson(int(df.correct.sum()), len(df))
    assert (hi - lo) > 3 * (whi - wlo)
