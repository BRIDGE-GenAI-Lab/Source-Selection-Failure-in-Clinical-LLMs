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
