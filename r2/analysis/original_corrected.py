"""Exploratory reanalysis of the ORIGINAL responses: all pairs vs pairs that were discriminable
as presented. Original responses are never reused for any changed input."""
import json
import re

import pandas as pd

from r2.analysis.stats import CATEGORY, cluster_bootstrap_mean, fdr, gee_logit, position_summary
from r2.original_runs import load_original, sham_position
from r2.paths import AUDIT_DIR, REPO
from r2.profiles import profile_for

OUT = REPO / "analysis_original"
SAYS_IDENTICAL = re.compile(
    r"\b(identical|equivalent|same content|no (meaningful|substantive|significant) difference|indistinguishable)\b",
    re.I)


def original_frame():
    audit = pd.read_csv(AUDIT_DIR / "pair_audit.csv").set_index("case_id")
    rows = []
    for r in load_original():
        d = r.get("model_decision") or {}
        rationale = d.get("trust_rationale") or ""
        rows.append({
            "model": r["model_name"], "case_id": r["case_id"], "sham_type": r["sham_trap_type"],
            "category": CATEGORY[r["sham_trap_type"]], "profile": profile_for(r["model_name"]),
            "sham_first": sham_position(r) == "A", "selected_first": d.get("selected_tool") == "A",
            "correct": int(r["selected_tool_correct"] is True), "confidence": d.get("confidence"),
            "says_identical": bool(SAYS_IDENTICAL.search(rationale)),
            "visible_status": audit.loc[r["case_id"], "visible_status"],
            "raw_status": audit.loc[r["case_id"], "raw_status"],
        })
    return pd.DataFrame(rows)


def _per_model(df):
    rows = []
    for model, g in df.groupby("model"):
        s = position_summary(g)
        s["model"] = model
        s["profile"] = g.profile.iloc[0]
        rows.append(s)
    out = pd.DataFrame(rows)
    out["content_p_fdr"] = fdr(out.content_p)
    return out.sort_values("position_averaged_accuracy", ascending=False)


def _per_type(df):
    """Failure rate per sham type, pooled across models; CI = case-cluster bootstrap (each case is
    evaluated by every model, so evaluations are not independent)."""
    rows = []
    for t, g in df.groupby("sham_type"):
        fails = int((1 - g.correct).sum())
        lo, hi = cluster_bootstrap_mean(g.assign(failure=1 - g.correct), "failure")
        rows.append({"sham_type": t, "category": CATEGORY[t], "n": len(g),
                     "failures": fails, "failure_rate": fails / len(g), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def _pooled(sub):
    """Position summary pooled across models, without the unpaired two-proportion z-test (which treats
    correlated evaluations of the same case as independent); the case-clustered GEE carries inference."""
    s = position_summary(sub)
    s.pop("content_z")
    s.pop("content_p")
    return s


def _by_profile(sub):
    return pd.DataFrame({p: _pooled(g) for p, g in sub.groupby("profile")}).T


def run(write=True):
    df = original_frame()
    subsets = {"all": df, "discriminable": df[df.visible_status != "identical"],
               "identical": df[df.visible_status == "identical"]}
    head = {}
    for name, sub in subsets.items():
        s = _pooled(sub)
        s["acc_lo"], s["acc_hi"] = cluster_bootstrap_mean(sub, "correct")
        s["says_identical"] = float(sub.says_identical.mean())
        head[name] = s
    head["types_without_valid_pairs"] = sorted(set(df.sham_type) - set(subsets["discriminable"].sham_type))
    if write:
        OUT.mkdir(exist_ok=True)
        formula = "correct ~ sham_first + C(category, Treatment('metadata')) + C(model)"
        for name in ("all", "discriminable"):
            sub = subsets[name]
            _per_model(sub).to_csv(OUT / f"per_model_{name}.csv", index=False)
            _per_type(sub).to_csv(OUT / f"per_type_{name}.csv", index=False)
            gee_logit(sub, formula).to_csv(OUT / f"gee_{name}.csv")
            _by_profile(sub).to_csv(OUT / f"by_profile_{name}.csv")
            sub = sub.assign(inline=(sub.profile != "openai_fc").astype(int),
                             sham_first_int=sub.sham_first.astype(int))
            sub = sub.assign(sham_first_x_inline=sub.sham_first_int * sub.inline)
            # inline main effect is absorbed by C(model) by design
            gee_logit(sub, "correct ~ sham_first_int + sham_first_x_inline + "
                           "C(category, Treatment('metadata')) + C(model)"
                      ).to_csv(OUT / f"gee_profile_interaction_{name}.csv")
        (df.groupby("raw_status")[["correct", "selected_first", "says_identical"]].mean()
           .assign(n=df.groupby("raw_status").size())
           .to_csv(OUT / "behavior_by_raw_status.csv"))
        (OUT / "headline.json").write_text(json.dumps(head, indent=2))
    return head


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
