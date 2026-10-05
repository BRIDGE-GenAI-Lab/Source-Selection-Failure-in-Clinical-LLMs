import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as st
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

CATEGORY = {
    "missing_warning": "clinical_safety", "allergy_ignorance": "clinical_safety",
    "dosing_error": "clinical_safety", "contraindication_violation": "clinical_safety",
    "wrong_population": "semantic", "subtle_inversion": "semantic", "authority_mimicry": "semantic",
    "prompt_injection": "injection",
    "fabricated_citation": "metadata", "outdated_version": "metadata",
}


def wilson(k, n):
    if n == 0:  # an empty arm (e.g. a model with no ok record in one order) has no estimate
        return float("nan"), float("nan"), float("nan")
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return k / n, lo, hi


def two_prop_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * st.norm.sf(abs(z))


def position_summary(df):
    sf, ss = df[df.sham_first], df[~df.sham_first]
    nan = float("nan")
    acc_sf = sf.correct.mean() if len(sf) else nan
    acc_ss = ss.correct.mean() if len(ss) else nan
    # content sensitivity: P(select first | sham second) vs P(select first | sham first)
    z, p = two_prop_z(int(ss.selected_first.sum()), len(ss), int(sf.selected_first.sum()), len(sf))
    return {
        "n": int(len(df)), "accuracy": float(df.correct.mean()),
        "primacy": float(df.selected_first.mean()),
        "acc_sham_first": float(acc_sf), "acc_sham_second": float(acc_ss),
        "position_averaged_accuracy": float((acc_sf + acc_ss) / 2),
        "content_z": float(z), "content_p": float(p),
    }


def fdr(pvals):
    return multipletests(list(pvals), alpha=0.05, method="fdr_bh")[1]


def gee_logit(df, formula, group="case_id"):
    res = smf.gee(formula, groups=group, data=df, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = res.conf_int()
    return pd.DataFrame({"OR": np.exp(res.params), "lo": np.exp(ci[0]),
                         "hi": np.exp(ci[1]), "p": res.pvalues})


def calibration(conf, correct, bins=10):
    conf = np.clip(np.asarray(conf, dtype=float), 0, 1)
    correct = np.asarray(correct, dtype=float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    ece, mce = 0.0, 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            gap = abs(conf[m].mean() - correct[m].mean())
            ece += m.mean() * gap
            mce = max(mce, gap)
    pos, neg = conf[correct == 1], conf[correct == 0]
    auroc = float("nan")
    if len(pos) and len(neg):
        auroc = st.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))
    return {"ece": float(ece), "mce": float(mce), "brier": float(np.mean((conf - correct) ** 2)),
            "auroc": float(auroc), "mean_conf": float(conf.mean()), "accuracy": float(correct.mean())}


def cluster_bootstrap_paa(df, n_boot=2000, seed=20261005):
    if df.sham_first.nunique() < 2:
        return float("nan"), float("nan")
    agg = df.groupby(["case_id", "sham_first"]).correct.agg(["sum", "count"]).unstack()
    cols = pd.MultiIndex.from_product([["sum", "count"], [True, False]])
    agg = agg.reindex(columns=cols).fillna(0)
    k1, n1 = agg[("sum", True)].to_numpy(), agg[("count", True)].to_numpy()
    k2, n2 = agg[("sum", False)].to_numpy(), agg[("count", False)].to_numpy()
    m = len(agg)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, m, size=(n_boot, m))
    w = np.stack([np.bincount(d, minlength=m) for d in draws])
    d1, d2 = w @ n1, w @ n2
    ok = (d1 > 0) & (d2 > 0)
    paa = np.full(n_boot, np.nan)
    paa[ok] = ((w @ k1)[ok] / d1[ok] + (w @ k2)[ok] / d2[ok]) / 2
    return float(np.nanpercentile(paa, 2.5)), float(np.nanpercentile(paa, 97.5))


def cluster_bootstrap_mean(df, col, n_boot=2000, seed=20261005):
    """95% percentile CI for the pooled mean of `col`, resampling case_id with replacement (every row
    of a drawn case enters the resample). NaN values of `col` are ignored."""
    if len(df) == 0:
        return float("nan"), float("nan")
    agg = df.groupby("case_id")[col].agg(["sum", "count"])
    k, n = agg["sum"].to_numpy(dtype=float), agg["count"].to_numpy(dtype=float)
    m = len(agg)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, m, size=(n_boot, m))
    w = np.stack([np.bincount(d, minlength=m) for d in draws])
    den = w @ n
    stat = np.full(n_boot, np.nan)
    stat[den > 0] = (w @ k)[den > 0] / den[den > 0]
    if np.isnan(stat).all():
        return float("nan"), float("nan")
    return float(np.nanpercentile(stat, 2.5)), float(np.nanpercentile(stat, 97.5))
