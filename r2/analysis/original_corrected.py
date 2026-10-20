"""Exploratory reanalysis of the ORIGINAL responses: all pairs vs pairs that were discriminable
as presented. Original responses are never reused for any changed input.

Table 1 examples (analysis_original/table1_examples.csv, `table1_examples`), rule as the documents may quote it:
For each modification type with non-identical pairs, in alphabetical order of type, one failure (the model
selected the modified excerpt) and then one success (it selected the authentic excerpt) were taken from the
original evaluations of the non-identical pairs, leaving out the records whose choice the original runner set to
A: the first record in ascending order of case identifier and then model identifier whose model did not already
appear in the table or, when every qualifying model already appeared, the first record in that order. The
rationale is the first 300 characters of the model's stated rationale, verbatim; an ellipsis marks a cut.
Fabricated-citation rows are cue-confounded.
"""
import json
import re

import pandas as pd

from r2.analysis import confidence_original
from r2.analysis.stats import CATEGORY, cluster_bootstrap_mean, cluster_bootstrap_paa, fdr, gee_logit, position_summary
from r2.analysis.table1 import modification, patient, quote
from r2.io import load_original_cases
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
            "selected": d.get("selected_tool"), "rationale": rationale,
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
        s["paa_lo"], s["paa_hi"] = cluster_bootstrap_paa(g)
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


def _measures(sub):
    """Accuracy, primacy, accuracy with the sham first and second, and position-averaged accuracy, each followed
    by its case-cluster bootstrap CI (`<measure>_lo`, `<measure>_hi`)."""
    s = position_summary(sub)
    out = {}
    for measure, rows, col in (("accuracy", sub, "correct"),
                               ("primacy", sub, "selected_first"),
                               ("acc_sham_first", sub[sub.sham_first], "correct"),
                               ("acc_sham_second", sub[~sub.sham_first], "correct")):
        out[measure] = s[measure]
        out[f"{measure}_lo"], out[f"{measure}_hi"] = cluster_bootstrap_mean(
            rows.assign(**{col: rows[col].astype(float)}), col)
    out["position_averaged_accuracy"] = s["position_averaged_accuracy"]
    out["position_averaged_accuracy_lo"], out["position_averaged_accuracy_hi"] = cluster_bootstrap_paa(sub)
    return out


def _by_profile(sub):
    """Pooled measures per runner configuration, with the number of models and case-cluster bootstrap CIs."""
    rows = {}
    for p, g in sub.groupby("profile"):
        s = _pooled(g)
        s["n_models"] = int(g.model.nunique())
        s.update({k: v for k, v in _measures(g).items() if k.endswith(("_lo", "_hi"))})
        rows[p] = s
    return pd.DataFrame(rows).T


def _excluding_coerced(sub):
    """Sensitivity: the pooled measures recomputed without the records whose choice the original runner
    coerced to "A" (audit/coerced_choices.csv), with case-cluster bootstrap CIs."""
    kept = sub[~sub.coerced]
    return {"n": int(len(kept)), "n_coerced_excluded": int(sub.coerced.sum()), **_measures(kept)}


def _per_model_excluding_coerced(disc):
    """Per model: the records whose choice the runner set (count and valid share), and position-averaged accuracy
    and accuracy by position as recorded and without those records, with case-cluster bootstrap CIs. The records
    left are the replies the runner could parse, a selected subset of the model's replies, not a random sample."""
    rows = []
    for model, g in disc.groupby("model"):
        k = g[~g.coerced]
        if k.sham_first.all() or (~k.sham_first).all():
            raise ValueError(f"{model}: no valid record in one position; position-averaged accuracy not estimable")
        a, b = position_summary(g), position_summary(k)
        rec = {"model": model, "profile": g.profile.iloc[0], "n": int(len(g)), "n_runner_set": int(g.coerced.sum()),
               "n_valid": int(len(k)), "valid_share": len(k) / len(g),
               "position_averaged_accuracy": a["position_averaged_accuracy"]}
        rec["paa_lo"], rec["paa_hi"] = cluster_bootstrap_paa(g)
        rec["position_averaged_accuracy_valid"] = b["position_averaged_accuracy"]
        rec["paa_valid_lo"], rec["paa_valid_hi"] = cluster_bootstrap_paa(k)
        rec.update({"acc_sham_first_valid": b["acc_sham_first"], "acc_sham_second_valid": b["acc_sham_second"],
                    "n_valid_sham_first": int(k.sham_first.sum()), "n_valid_sham_second": int((~k.sham_first).sum())})
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["n_runner_set", "model"], ascending=[False, True])


CUE_TYPE = "fabricated_citation"
CUE_KEY = f"{CUE_TYPE}_cue_confounded"
CUE_NOTE = (
    "Exploratory and cue-confounded. Every original fabricated_citation modified excerpt ended with "
    "\"Evidence: 'NEJM 2025;392:<pages>' (DOI:10.9999/fake.NNNN)\", so the word 'fake' was visible in the "
    "modified excerpt to all 21 models (audit/self_label_tokens.json), and a blank line, which no authentic "
    "excerpt had, preceded the citation (audit/format_cues.json). These rates were obtained with both cues "
    "present. They do not measure detection of a nonexistent citation. They are reported separately, and the "
    "pooled sensitivity analysis without fabricated_citation excludes them. In the round-1 physician review "
    "packets the citation carried a neutral suffix (10.9999/cg.<year>.NNNN; verification/round1_summary.json).")


def _cue_confounded(df):
    """Final fix F1: the original fabricated_citation results, pooled over the 21 models, labelled as
    cue-confounded. Case-cluster bootstrap CIs as for every other pooled rate."""
    sub = df[df.sham_type == "fabricated_citation"]
    sub = sub.assign(failure=1 - sub.correct)
    tokens = json.loads((AUDIT_DIR / "self_label_tokens.json").read_text(encoding="utf-8"))
    blank = json.loads((AUDIT_DIR / "format_cues.json").read_text(encoding="utf-8"))["blank_line_cue"]
    if blank["modified_payload_rows"] != len(sub) or blank["authentic_payload_rows"] != 0:
        raise ValueError("audit/format_cues.json no longer supports CUE_NOTE's blank-line sentence")
    s = position_summary(sub)
    out = {"label": "exploratory; cue-confounded", "sham_type": "fabricated_citation", "note": CUE_NOTE,
           "token": "fake", "rows_with_token": tokens["modified_tool_payload"]["fake"]["rows"],
           "n": s["n"], "n_cases": int(sub.case_id.nunique()), "n_models": int(sub.model.nunique()),
           "failures": int(sub.failure.sum()), "failure_rate": float(sub.failure.mean())}
    out["failure_rate_lo"], out["failure_rate_hi"] = cluster_bootstrap_mean(sub, "failure")
    out.update(_measures(sub))
    out["ci"] = "case-cluster bootstrap (2,000 draws, percentile 95% CI; r2.analysis.stats)"
    return out


def _headline(sub):
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
    return s


NOFC_KEY = "discriminable_excluding_fabricated_citation"
TABLE1_CHARS = 300
TABLE1_COLUMNS = ("sham_type", "category", "outcome", "case_id", "model", "model_display", "order", "selected",
                  "confidence", "patient", "modification", "rationale", "cue_confounded")


def _visible_edit(case):
    """The edit as the models could see it (excerpt_text and the visible metadata fields of r2.audit.pair_audit),
    as an edit record for r2.analysis.table1.modification."""
    from r2.audit.pair_audit import VISIBLE_META
    t, s = case["tools"]["T"], case["tools"]["S"]
    changes = [{"field": "excerpt_text", "before": t["excerpt_text"], "after": s["excerpt_text"]}] \
        if t["excerpt_text"] != s["excerpt_text"] else []
    changes += [{"field": f"metadata.{k}", "before": t["metadata"].get(k), "after": s["metadata"].get(k)}
                for k in VISIBLE_META if t["metadata"].get(k) != s["metadata"].get(k)]
    return {"changes": changes}


def table1_examples(df):
    """Table 1 rows by the rule in the module docstring: per sham type (sorted) a failure, then a success, each the
    first non-coerced record of the non-identical pairs in (case_id, model) order whose model is not yet in the
    table, else the first in that order. The order of `df` does not matter. Order column: TS, authentic excerpt
    shown first (as A); ST, modified excerpt shown first."""
    from r2.analysis.figures import display_name  # local import: r2.analysis.figures loads matplotlib
    cases, _ = load_original_cases()
    pool = df[(df.raw_status == "visible") & ~df.coerced].sort_values(["case_id", "model"])
    rows, used = [], set()
    for t in sorted(pool.sham_type.unique()):
        for outcome, correct in (("Failure", 0), ("Success", 1)):
            hit = pool[(pool.sham_type == t) & (pool.correct == correct)]
            if hit.empty:
                continue
            fresh = hit[~hit.model.isin(used)]
            r = (fresh if len(fresh) else hit).iloc[0]
            used.add(r.model)
            case = cases[r.case_id]
            rows.append(dict(zip(TABLE1_COLUMNS, (
                t, CATEGORY[t], outcome, r.case_id, r.model, display_name(r.model), "ST" if r.sham_first else "TS",
                r.selected, r.confidence, patient(case), modification(_visible_edit(case)),
                quote(r.rationale, TABLE1_CHARS), t == CUE_TYPE))))
    return rows


def _excluding_fabricated_citation(disc):
    """Sensitivity: the discriminable pairs without the cue-confounded fabricated_citation pairs, with the same
    measures and CIs as the `discriminable` block; `excluding_coerced` also drops the coerced records."""
    cue = disc.sham_type == CUE_TYPE
    kept = disc[~cue]
    return {"excluded_sham_type": CUE_TYPE, "n_pairs": int(kept.case_id.nunique()),
            "n_excluded": int(cue.sum()), "n_pairs_excluded": int(disc[cue].case_id.nunique()),
            **_headline(kept), "excluding_coerced": _excluding_coerced(kept)}


def run(write=True):
    df = original_frame()
    subsets = {"all": df, "discriminable": df[df.visible_status != "identical"],
               "identical": df[df.visible_status == "identical"]}
    head = {name: _headline(sub) for name, sub in subsets.items()}
    head["types_without_valid_pairs"] = sorted(set(df.sham_type) - set(subsets["discriminable"].sham_type))
    head["excluding_coerced"] = {name: _excluding_coerced(sub) for name, sub in subsets.items()}
    head[CUE_KEY] = _cue_confounded(df)
    head[NOFC_KEY] = _excluding_fabricated_citation(subsets["discriminable"])
    with_conf = confidence_original.attach_confidence(df)
    conf = {"all": with_conf, "discriminable": with_conf[with_conf.visible_status != "identical"]}
    head["confidence"] = confidence_original.confidence_block(conf)
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
        pd.DataFrame(table1_examples(df)).to_csv(OUT / "table1_examples.csv", index=False)
        for name in ("all", "discriminable"):
            confidence_original.per_model(conf[name]).to_csv(OUT / f"confidence_{name}.csv", index=False)
        confidence_original.per_model_excluding_runner_set(conf["discriminable"]).to_csv(
            OUT / "confidence_discriminable_excluding_coerced.csv", index=False)
        _per_model_excluding_coerced(subsets["discriminable"]).to_csv(
            OUT / "per_model_discriminable_excluding_coerced.csv", index=False)
    return head


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
