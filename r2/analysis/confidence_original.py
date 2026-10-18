"""Stated confidence in the original evaluations, with the R1 definitions, for all pairs and the non-identical pairs.

The R1 analysis code for confidence is not in this repository; the definitions are those of the R1 Methods and
Supplementary Table S10, and with them the all-pairs values reproduce every R1 value (tests/r2/
test_original_additions.py):
- rows without a stated confidence are left out (4 Llama-3.2-3B records; R1 N = 10,496);
- overconfidence = mean stated confidence minus accuracy;
- expected calibration error (ECE) over ten equal-width bins, bin = np.digitize(confidence, np.linspace(0, 1, 11)) - 1,
  clipped to 0-9 (this binning reproduces all 21 R1 per-model ECE values), maximum calibration error (MCE) over the
  same bins, Brier score;
- AUROC of confidence for correctness (Mann-Whitney U / (n1 n2), ties counted one half);
- per model, the two-sided Mann-Whitney U test of confidence in correct versus incorrect selections, with
  Benjamini-Hochberg FDR across the models; "higher confidence when correct" = FDR q < 0.05 and AUROC > 0.5;
- high-confidence errors = the share of failures with confidence >= 0.90.
Pooled values carry case-cluster bootstrap CIs (2,000 resamples, seed 20261005, the draws of r2.analysis.stats).
Confidence comes from data/results_10500.csv and must equal the value in the run records.
"""
import numpy as np
import pandas as pd
from scipy import stats as st

from r2.analysis.stats import fdr
from r2.paths import REPO

RESULTS_CSV = REPO / "data" / "results_10500.csv"
EDGES = np.linspace(0, 1, 11)
HIGH = 0.90
N_BOOT, SEED = 2000, 20261005
DEFINITIONS = (
    "R1 definitions (Methods; Supplementary Table S10). Rows without a stated confidence are left out. "
    "Overconfidence: mean stated confidence minus accuracy. ECE: ten equal-width bins "
    "(np.digitize(confidence, np.linspace(0, 1, 11)) - 1, clipped to 0-9); MCE over the same bins; Brier score. "
    "AUROC of confidence for correctness (Mann-Whitney U / (n1 n2)). Per model: two-sided Mann-Whitney U test of "
    "confidence in correct versus incorrect selections, Benjamini-Hochberg FDR across the models; higher confidence "
    "when correct = FDR q < 0.05 and AUROC > 0.5. High-confidence errors: share of failures with confidence >= 0.90. "
    "Pooled CIs: case-cluster bootstrap, 2,000 resamples, seed 20261005. Source: data/results_10500.csv.")
MEASURES = ("accuracy", "mean_confidence", "overconfidence", "ece", "mce", "brier", "auroc", "high_conf_error_share")
PER_MODEL_COLS = ("model", "profile", "n", "n_missing_confidence", "accuracy", "mean_confidence",
                  "mean_confidence_correct", "mean_confidence_incorrect", "overconfidence", "ece", "mce", "brier",
                  "auroc", "high_conf_error_share", "mw_u", "mw_p", "mw_p_fdr", "higher_confidence_when_correct")


def bins(conf):
    return np.clip(np.digitize(np.asarray(conf, dtype=float), EDGES) - 1, 0, 9)


def attach_confidence(df):
    """df with `conf` from data/results_10500.csv, joined on (model, case_id); refuses a value that differs
    from the run records' confidence."""
    csv = pd.read_csv(RESULTS_CSV, usecols=["model", "case_id", "confidence"])
    out = df.merge(csv.rename(columns={"confidence": "conf"}), on=["model", "case_id"], how="left", validate="1:1")
    a, b = out.confidence.astype(float), out.conf
    if len(out) != len(df) or not ((a == b) | (a.isna() & b.isna())).all():
        raise ValueError("confidence in data/results_10500.csv differs from the run records")
    return out


def _from_counts(levels, pos, neg):
    """Measures from counts per confidence level (last axis): pos = correct, neg = incorrect selections."""
    n1, n0 = pos.sum(-1), neg.sum(-1)
    n = n1 + n0
    tot = pos + neg
    sum_conf = (tot * levels).sum(-1)
    acc, mean_conf = n1 / n, sum_conf / n
    brier = (pos * (levels - 1) ** 2 + neg * levels ** 2).sum(-1) / n
    b = bins(levels)
    gaps, ece = [], 0
    for k in range(10):
        m = b == k
        if not m.any():
            continue
        nb = tot[..., m].sum(-1)
        diff = np.abs((tot[..., m] * levels[m]).sum(-1) - pos[..., m].sum(-1))
        ece = ece + diff / n
        with np.errstate(divide="ignore", invalid="ignore"):
            gaps.append(np.where(nb > 0, diff / np.where(nb > 0, nb, 1), -np.inf))
    below = np.cumsum(neg, axis=-1) - neg  # incorrect selections at lower confidence levels
    with np.errstate(divide="ignore", invalid="ignore"):
        auroc = (pos * (below + 0.5 * neg)).sum(-1) / (n1 * n0)
        high = neg[..., levels >= HIGH].sum(-1) / n0
    return {"accuracy": acc, "mean_confidence": mean_conf, "overconfidence": mean_conf - acc, "ece": ece,
            "mce": np.max(np.stack(gaps, -1), -1), "brier": brier, "auroc": auroc, "high_conf_error_share": high}


def _counts(rows, levels):
    """Per case, counts of correct and incorrect selections at each confidence level."""
    idx = np.searchsorted(levels, rows.conf.to_numpy(float))
    cases, ci = np.unique(rows.case_id.to_numpy(), return_inverse=True)
    pos = np.zeros((len(cases), len(levels)), dtype=np.int64)
    neg = np.zeros_like(pos)
    np.add.at(pos, (ci[rows.correct.to_numpy() == 1], idx[rows.correct.to_numpy() == 1]), 1)
    np.add.at(neg, (ci[rows.correct.to_numpy() == 0], idx[rows.correct.to_numpy() == 0]), 1)
    return pos, neg


def pooled(sub, n_boot=N_BOOT, seed=SEED):
    """Pooled measures over the rows with a stated confidence, each with its case-cluster bootstrap CI."""
    rows = sub[sub.conf.notna()]
    levels = np.unique(rows.conf.to_numpy(float))
    pos, neg = _counts(rows, levels)
    est = _from_counts(levels, pos.sum(0), neg.sum(0))
    m = len(pos)
    draws = np.random.default_rng(seed).integers(0, m, size=(n_boot, m))
    w = np.zeros((n_boot, m), dtype=np.int64)
    np.add.at(w, (np.repeat(np.arange(n_boot), m), draws.ravel()), 1)
    boot = _from_counts(levels, w @ pos, w @ neg)  # integer products: exact, no BLAS
    out = {"n": int(len(rows)), "n_missing_confidence": int(sub.conf.isna().sum()),
           "n_failures": int((rows.correct == 0).sum())}
    for k in MEASURES:
        stat = boot[k][np.isfinite(boot[k])]
        out[k] = float(est[k])
        out[f"{k}_lo"], out[f"{k}_hi"] = float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))
    return out


def per_model(sub):
    rows = []
    for model, g in sub.groupby("model"):
        r = g[g.conf.notna()]
        levels = np.unique(r.conf.to_numpy(float))
        pos, neg = _counts(r, levels)
        est = _from_counts(levels, pos.sum(0), neg.sum(0))
        c, w = r.conf[r.correct == 1], r.conf[r.correct == 0]
        mw = st.mannwhitneyu(c, w, alternative="two-sided")
        rows.append({"model": model, "profile": g.profile.iloc[0], "n": int(len(r)),
                     "n_missing_confidence": int(g.conf.isna().sum()), **{k: float(est[k]) for k in MEASURES},
                     "mean_confidence_correct": float(c.mean()), "mean_confidence_incorrect": float(w.mean()),
                     "mw_u": float(mw.statistic), "mw_p": float(mw.pvalue)})
    out = pd.DataFrame(rows)
    out["mw_p_fdr"] = fdr(out.mw_p)
    out["higher_confidence_when_correct"] = (out.mw_p_fdr < 0.05) & (out.auroc > 0.5)
    return out[list(PER_MODEL_COLS)].sort_values("auroc", ascending=False, kind="stable").reset_index(drop=True)


def summary(sub):
    """Pooled measures and the per-model FDR counts of one set."""
    pm = per_model(sub)
    return {**pooled(sub), "n_models": int(len(pm)),
            "n_models_fdr_significant": int((pm.mw_p_fdr < 0.05).sum()),
            "n_models_higher_confidence_when_correct": int(pm.higher_confidence_when_correct.sum())}


def confidence_block(subsets):
    """The headline `confidence` key: all and discriminable (non-identical) pairs, with and without the records
    whose choice the original runner set to A (their confidence is mostly the runner's default 0.5 or 0)."""
    return {"definitions": DEFINITIONS,
            **{name: summary(subsets[name]) for name in ("all", "discriminable")},
            "excluding_coerced": {name: summary(subsets[name][~subsets[name].coerced])
                                  for name in ("all", "discriminable")}}
