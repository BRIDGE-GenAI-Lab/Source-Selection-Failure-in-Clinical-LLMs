#!/usr/bin/env python3
"""Supplementary Table S11 -- full cluster-aware primary regression.

Outcome  = correct source selection (1/0).
Predictors = sham_first (sham document in position A), attack category
             (reference = metadata), and model fixed effects.

Reports odds ratios + 95% CI under three specifications:
  (a) naive logistic regression,
  (b) cluster-robust standard errors by vignette (case_id),
  (c) GEE with an exchangeable working correlation by vignette.

Naive reproduces the manuscript's position OR 0.087 and clinical-safety OR ~0.14;
the clustered specifications leave the point estimates unchanged and widen the
CIs (which still exclude 1). Writes data/outputs/r2_table_s11.{json,csv}.

Note (paths): like the other analysis scripts, this expects the raw per-run
`results/` directory of the evaluation harness (not distributed) so that
load_canonical.py can rebuild the canonical 21 x 500 set. The shipped
data/outputs/r2_table_s11.{json,csv} are the exact artifacts this script
produces, so the table can be inspected without re-running the experiment.
"""
import os, sys, json, csv, warnings
warnings.filterwarnings("ignore")

# import the canonical loader that sits alongside this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_canonical import load_canonical

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "outputs")

# 10 adversarial modification types -> 4 attack categories
CAT = {
    "missing_warning": "clinical_safety", "dosing_error": "clinical_safety",
    "contraindication_violation": "clinical_safety", "allergy_ignorance": "clinical_safety",
    "wrong_population": "semantic", "subtle_inversion": "semantic", "authority_mimicry": "semantic",
    "prompt_injection": "injection",
    "fabricated_citation": "metadata", "outdated_version": "metadata",
}

TERMS = {
    "sham_first": "sham_first",
    "C(category, Treatment('metadata'))[T.clinical_safety]": "clinical_safety",
    "C(category, Treatment('metadata'))[T.semantic]": "semantic",
    "C(category, Treatment('metadata'))[T.injection]": "injection",
}
FORMULA = "correct ~ sham_first + C(category, Treatment('metadata')) + C(model)"


def _or_ci(res):
    ci = res.conf_int()
    return {lab: [float(np.exp(res.params[k])),
                  float(np.exp(ci.loc[k, 0])), float(np.exp(ci.loc[k, 1]))]
            for k, lab in TERMS.items()}


def main():
    recs, _ = load_canonical()
    df = pd.DataFrame([
        dict(correct=int(bool(r.get("selected_tool_correct"))),
             sham_first=1 if (r.get("mapping") or {}).get("A") == "S" else 0,
             category=CAT[r["sham_trap_type"]],
             model=r["model_name"], case_id=r["case_id"])
        for r in recs
    ])

    naive = smf.glm(FORMULA, data=df, family=sm.families.Binomial()).fit()
    clust = smf.glm(FORMULA, data=df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": df["case_id"]})
    gee = smf.gee(FORMULA, groups="case_id", data=df, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    R = {"naive": _or_ci(naive), "cluster_robust": _or_ci(clust), "gee_exchangeable": _or_ci(gee)}

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "_description": ("Supplementary Table S11 -- full cluster-aware primary regression. "
                         "Outcome: correct source selection. Predictors: sham_first (sham in position A), "
                         "attack category (reference=metadata), and model fixed effects. Odds ratios with "
                         "95% CI under naive logistic, cluster-robust SE by vignette, and GEE exchangeable."),
        "n_evaluations": int(len(df)), "n_vignettes": int(df.case_id.nunique()),
        "n_models": int(df.model.nunique()), "reference_category": "metadata", "odds_ratios": R,
    }
    with open(os.path.join(OUT_DIR, "r2_table_s11.json"), "w") as f:
        json.dump(payload, f, indent=2)

    with open(os.path.join(OUT_DIR, "r2_table_s11.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "specification", "odds_ratio", "ci_low", "ci_high"])
        for spec, d in R.items():
            for term in ["sham_first", "clinical_safety", "semantic", "injection"]:
                o, lo, hi = d[term]
                w.writerow([term, spec, round(o, 4), round(lo, 4), round(hi, 4)])

    print(f"n={len(df)}  vignettes={df.case_id.nunique()}  models={df.model.nunique()}")
    for term in ["sham_first", "clinical_safety", "semantic", "injection"]:
        o = R["naive"][term]
        print(f"  {term:16} naive OR {o[0]:.3f} ({o[1]:.3f}-{o[2]:.3f})")
    print("wrote data/outputs/r2_table_s11.{json,csv}")


if __name__ == "__main__":
    main()
