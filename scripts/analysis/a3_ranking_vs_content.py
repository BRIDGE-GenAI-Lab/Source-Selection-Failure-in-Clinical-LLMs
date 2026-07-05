# a3_ranking_vs_content.py
"""Ranking-vs-content decomposition.

(1) Content-beyond-position test: under a pure positional heuristic,
    acc_sham_A + acc_sham_B = 1.0. Content use shows up as the sum exceeding 1.0.
    z-test vs the null sum=1, overall and per model.
(2) Multivariable logistic regression correct ~ C(sham_pos) + C(category) + C(model);
    report ORs (95% CI) for position and each attack category.
-> outputs/ranking_vs_content.json
"""
import json, os, numpy as np
import pandas as pd, statsmodels.formula.api as smf
from scipy.stats import norm
from load_canonical import load_canonical, sham_position

CAT = {  # trap -> category (from manuscript)
    "missing_warning": "safety", "allergy_ignorance": "safety", "dosing_error": "safety",
    "contraindication_violation": "safety", "wrong_population": "semantic",
    "subtle_inversion": "semantic", "authority_mimicry": "semantic",
    "prompt_injection": "injection", "fabricated_citation": "metadata",
    "outdated_version": "metadata",
}


def main():
    recs, by_model = load_canonical()
    df = pd.DataFrame([{
        "correct": int(bool(r["selected_tool_correct"])),
        "sham_pos": sham_position(r),
        "trap": r.get("sham_trap_type", "unknown"),
        "model": r["model_name"],
    } for r in recs])
    df["category"] = df["trap"].map(CAT)
    # All canonical traps map to one of the four categories; an unmapped trap
    # would signal a data/schema drift, so surface it rather than silently
    # bucketing it into a degenerate, unpopulated "other" level (which also
    # destabilises the logistic fit).
    if df["category"].isna().any():
        bad = sorted(df.loc[df["category"].isna(), "trap"].unique())
        raise ValueError(f"unmapped sham_trap_type(s): {bad}")

    # (1) content-beyond-position, overall
    a = df[df.sham_pos == "A"]; b = df[df.sham_pos == "B"]
    pa, na = a.correct.mean(), len(a)
    pb, nb = b.correct.mean(), len(b)
    s = pa + pb               # =1.0 under pure position
    se = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    z = (s - 1.0) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    content_signal = {"acc_A": round(pa * 100, 1), "acc_B": round(pb * 100, 1),
                      "sum": round(s, 3), "z": round(z, 2), "p": p,
                      "interpretation": "sum>1 => content used beyond position"}

    # per-model sum test
    per_model = {}
    for m, g in df.groupby("model"):
        ga, gb = g[g.sham_pos == "A"], g[g.sham_pos == "B"]
        if len(ga) == 0 or len(gb) == 0:
            continue
        pma, pmb = ga.correct.mean(), gb.correct.mean()
        sm = pma + pmb
        sem = np.sqrt(pma * (1 - pma) / len(ga) + pmb * (1 - pmb) / len(gb))
        zm = (sm - 1) / sem if sem > 0 else 0
        per_model[m] = {"sum": round(sm, 3), "z": round(zm, 2),
                        "p": 2 * (1 - norm.cdf(abs(zm))),
                        "content_above_position": bool(sm > 1 and 2 * (1 - norm.cdf(abs(zm))) < 0.05)}

    # (2) logistic regression
    fit = smf.logit(
        "correct ~ C(sham_pos, Treatment('B')) + C(category, Treatment('metadata')) + C(model)",
        df,
    ).fit(disp=0, maxiter=200)
    conf = fit.conf_int()

    def _exp(x):
        # guard against overflow from any perfectly-separated coefficient
        v = float(np.exp(x)) if abs(x) < 700 else float("inf")
        return round(v, 3) if np.isfinite(v) else None

    ors = {}
    for name in fit.params.index:
        if name.startswith("C(sham_pos") or name.startswith("C(category"):
            ci = conf.loc[name]
            ors[name] = {"OR": _exp(fit.params[name]),
                         "CI": [_exp(ci[0]), _exp(ci[1])],
                         "p": float(fit.pvalues[name])}
    out = {"content_signal_overall": content_signal,
           "content_signal_per_model": per_model,
           "logit_odds_ratios": ors,
           "n_models_content_above_position": sum(v["content_above_position"] for v in per_model.values()),
           "n_models_total": len(per_model)}
    os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "outputs", "ranking_vs_content.json"), "w"),
              indent=2, default=float)
    print(json.dumps(content_signal, indent=2))
    print("models using content beyond position:",
          out["n_models_content_above_position"], "/", out["n_models_total"])


if __name__ == "__main__":
    main()
