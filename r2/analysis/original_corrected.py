"""Exploratory reanalysis of the ORIGINAL responses: all pairs vs pairs that were discriminable
as presented. Original responses are never reused for any changed input."""
import json
import re

import pandas as pd

from r2.analysis.stats import CATEGORY, cluster_bootstrap_mean, cluster_bootstrap_paa, fdr, gee_logit, position_summary
from r2.original_runs import load_original, sham_position
from r2.paths import AUDIT_DIR, REPO
from r2.profiles import profile_for

OUT = REPO / "analysis_original"
SAYS_IDENTICAL = re.compile(
    r"\b(identical|equivalent|same content|no (meaningful|substantive|significant) difference|indistinguishable)\b",
    re.I)


def _coerced_keys():
    """(model, case_id) of every record whose choice the original runner set to "A" (audit/coerced_choices.csv)."""
    cc = pd.read_csv(AUDIT_DIR / "coerced_choices.csv", usecols=["model", "case_id", "class"])
    assert len(cc) == 10500, len(cc)
    return set(cc.loc[cc["class"] != "not_coerced", ["model", "case_id"]].itertuples(index=False, name=None))


def original_frame():
    audit = pd.read_csv(AUDIT_DIR / "pair_audit.csv").set_index("case_id")
    coerced = _coerced_keys()
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
            "coerced": (r["model_name"], r["case_id"]) in coerced,
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


def _excluding_coerced(sub):
    """Sensitivity: the pooled measures recomputed without the records whose choice the original runner
    coerced to "A" (audit/coerced_choices.csv), with case-cluster bootstrap CIs."""
    kept = sub[~sub.coerced]
    s = position_summary(kept)
    out = {"n": s["n"], "n_coerced_excluded": int(sub.coerced.sum())}
    for measure, rows, col in (("accuracy", kept, "correct"),
                               ("primacy", kept, "selected_first"),
                               ("acc_sham_first", kept[kept.sham_first], "correct"),
                               ("acc_sham_second", kept[~kept.sham_first], "correct")):
        out[measure] = s[measure]
        out[f"{measure}_lo"], out[f"{measure}_hi"] = cluster_bootstrap_mean(
            rows.assign(**{col: rows[col].astype(float)}), col)
    out["position_averaged_accuracy"] = s["position_averaged_accuracy"]
    out["position_averaged_accuracy_lo"], out["position_averaged_accuracy_hi"] = cluster_bootstrap_paa(kept)
    return out


CUE_KEY = "fabricated_citation_cue_confounded"
CUE_NOTE = (
    "Exploratory and cue-confounded. Every original fabricated_citation modified excerpt ended with "
    "\"Evidence: 'NEJM 2025;392:<pages>' (DOI:10.9999/fake.NNNN)\", so the word 'fake' was visible in the "
    "modified excerpt to all 21 models (audit/self_label_tokens.json). These rates were obtained with that cue "
    "present. They do not measure detection of a nonexistent citation, and they are not comparable with the "
    "repeated experiment, whose corrected excerpts carry a neutral suffix (10.9999/cg.<year>.NNNN).")


def _cue_confounded(df):
    """Final fix F1: the original fabricated_citation results, pooled over the 21 models, labelled as
    cue-confounded. Case-cluster bootstrap CIs as for every other pooled rate."""
    sub = df[df.sham_type == "fabricated_citation"]
    sub = sub.assign(failure=1 - sub.correct)
    tokens = json.loads((AUDIT_DIR / "self_label_tokens.json").read_text(encoding="utf-8"))
    s = position_summary(sub)
    out = {"label": "exploratory; cue-confounded", "sham_type": "fabricated_citation", "note": CUE_NOTE,
           "token": "fake", "rows_with_token": tokens["modified_tool_payload"]["fake"]["rows"],
           "n": s["n"], "n_cases": int(sub.case_id.nunique()), "n_models": int(sub.model.nunique()),
           "failures": int(sub.failure.sum()), "failure_rate": float(sub.failure.mean())}
    out["failure_rate_lo"], out["failure_rate_hi"] = cluster_bootstrap_mean(sub, "failure")
    for measure, rows, col in (("accuracy", sub, "correct"),
                               ("primacy", sub, "selected_first"),
                               ("acc_sham_first", sub[sub.sham_first], "correct"),
                               ("acc_sham_second", sub[~sub.sham_first], "correct")):
        out[measure] = s[measure]
        out[f"{measure}_lo"], out[f"{measure}_hi"] = cluster_bootstrap_mean(
            rows.assign(**{col: rows[col].astype(float)}), col)
    out["position_averaged_accuracy"] = s["position_averaged_accuracy"]
    out["position_averaged_accuracy_lo"], out["position_averaged_accuracy_hi"] = cluster_bootstrap_paa(sub)
    out["ci"] = "case-cluster bootstrap (2,000 draws, percentile 95% CI; r2.analysis.stats)"
    return out


def run(write=True):
    df = original_frame()
    subsets = {"all": df, "discriminable": df[df.visible_status != "identical"],
               "identical": df[df.visible_status == "identical"]}
    head = {}
    for name, sub in subsets.items():
        s = _pooled(sub)
        s["acc_lo"], s["acc_hi"] = cluster_bootstrap_mean(sub, "correct")
        s["says_identical"] = float(sub.says_identical.mean())
        # case-cluster bootstrap CIs for the other pooled rates (position rates use their own rows)
        s["position_averaged_accuracy_lo"], s["position_averaged_accuracy_hi"] = cluster_bootstrap_paa(sub)
        for rate, rows, col in (("primacy", sub, "selected_first"),
                                ("acc_sham_first", sub[sub.sham_first], "correct"),
                                ("acc_sham_second", sub[~sub.sham_first], "correct"),
                                ("says_identical", sub, "says_identical")):
            s[f"{rate}_lo"], s[f"{rate}_hi"] = cluster_bootstrap_mean(rows.assign(**{col: rows[col].astype(float)}), col)
        head[name] = s
    head["types_without_valid_pairs"] = sorted(set(df.sham_type) - set(subsets["discriminable"].sham_type))
    head["excluding_coerced"] = {name: _excluding_coerced(sub) for name, sub in subsets.items()}
    head[CUE_KEY] = _cue_confounded(df)
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
