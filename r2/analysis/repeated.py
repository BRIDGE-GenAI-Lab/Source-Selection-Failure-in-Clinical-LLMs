"""Pre-specified analyses of the repeated experiment (docs/r2/ANALYSIS_PLAN.md).
Exploratory outputs are written with an 'exploratory_' prefix."""
import json
import re

import pandas as pd
from scipy.stats import spearmanr

from r2.analysis.stats import (CATEGORY, calibration, cluster_bootstrap_paa, fdr, gee_logit,
                               position_summary, wilson)
from r2.paths import DATA_V2, REPO, RUNS_V2, VERIFY_DIR
from r2.run.harness import final_records, load_specs, results_path

OUT = REPO / "analysis_repeated"


def _tokens(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def noticed(differences, snippets, k=3):
    grams = set()
    for s in snippets:
        t = _tokens(s)
        grams |= {tuple(t[i:i + k]) for i in range(len(t) - k + 1)}
        if 0 < len(t) < k:
            grams.add(tuple(t))
    for d in differences or []:
        t = _tokens(d)
        if any(tuple(t[i:i + k]) in grams for i in range(len(t) - k + 1)):
            return True
        if any((tok,) in grams for tok in t):
            return True
    return False


def _side(x):
    if x is None:
        return []
    if isinstance(x, str):
        return x.splitlines() if "\n" in x else [x]
    if isinstance(x, (list, tuple, set)):
        return [str(i) for i in x]
    return [str(x)]


def _snippets(case):
    out = []
    for ch in case["edit_record"]["changes"]:
        out += list(set(_side(ch["before"])) ^ set(_side(ch["after"])))
    return [s for s in out if s.strip()]


def repeated_frame(out_dir=RUNS_V2 / "main"):
    pairs = {c["case_id"]: c for c in json.loads((DATA_V2 / "pairs_v2.json").read_text())["cases"]}
    excluded = set(json.loads((VERIFY_DIR / "excluded_pairs.json").read_text())["excluded"])
    flagged = set(json.loads((VERIFY_DIR / "flags.json").read_text()))
    specs = load_specs()
    substituted = {s["original_name"] for s in specs if s.get("substituted_for")}
    rows = []
    for spec in specs:
        path = results_path(spec, out_dir)
        if not path.exists():
            continue
        for r in final_records(path):
            d = r.get("decision") or {}
            sel = r.get("selected_tool")
            rows.append({
                "model": r["original_name"], "case_id": r["case_id"], "order": r["order"],
                "status": r["status"], "sham_first": r["labels"]["A"] == "S",
                "selected_first": None if sel is None else sel == "A",
                "correct": r.get("correct"), "confidence": d.get("confidence"),
                "differences": d.get("differences_found") or [],
            })
    df = pd.DataFrame(rows)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["sham_type"] = df.case_id.map(lambda c: pairs[c]["eval"]["sham_trap_type"])
    df["category"] = df.sham_type.map(CATEGORY)
    df["excluded"] = df.case_id.isin(excluded)
    df["flagged"] = df.case_id.isin(flagged)
    df["substituted"] = df.model.isin(substituted)
    df["noticed"] = [noticed(diff, _snippets(pairs[c])) for diff, c in zip(df.differences, df.case_id)]
    return df


def pair_level(df):
    ok = df[df.status == "ok"]
    rows = []
    for (model, cid), g in ok.groupby(["model", "case_id"]):
        if len(g) != 2:
            continue
        rows.append({"model": model, "case_id": cid,
                     "invariant_correct": int(g.correct.sum() == 2),
                     "wrong_both": int(g.correct.sum() == 0),
                     "position_driven": int(g.selected_first.nunique() == 1)})
    return pd.DataFrame(rows)


def _ok(df):
    out = df[df.status == "ok"].copy()
    out["correct"] = out.correct.astype(int)
    out["selected_first"] = out.selected_first.astype(bool)
    return out


def sensitivity_non_response(df):
    """Non-responses scored incorrect. Selection is undefined for them, so only n, accuracy and
    position-averaged accuracy are meaningful; primacy and the content test are set to NaN."""
    d = df.copy()
    d["correct"] = d.correct.fillna(0).astype(int)
    d["selected_first"] = d.selected_first.fillna(False).astype(bool)
    s = position_summary(d)
    s.update({"primacy": float("nan"), "content_z": float("nan"), "content_p": float("nan")})
    return s


def per_model_table(analysis):
    nan = float("nan")
    per_model = []
    for model, g in analysis.groupby("model"):
        s = position_summary(g)
        s["paa_lo"], s["paa_hi"] = cluster_bootstrap_paa(g)
        conf = g.dropna(subset=["confidence"])
        cal = calibration(conf.confidence, conf.correct) if len(conf) else {
            k: nan for k in ("ece", "mce", "brier", "auroc", "mean_conf", "accuracy")}
        cal = {("accuracy_conf" if k == "accuracy" else k): v for k, v in cal.items()}
        s.update({"model": model, **cal})
        per_model.append(s)
    pm = pd.DataFrame(per_model)
    pm["content_p_fdr"] = float("nan")
    valid = pm.content_p.notna()
    if valid.any():
        pm.loc[valid, "content_p_fdr"] = fdr(pm.loc[valid, "content_p"])
    return pm


def _summary(df):
    s = position_summary(df)
    s["paa_lo"], s["paa_hi"] = cluster_bootstrap_paa(df)
    return s


def run(out_dir=RUNS_V2 / "main", write=True):
    df = repeated_frame(out_dir)
    analysis = _ok(df[~df.excluded])
    pairs = pair_level(df[~df.excluded])
    head = {
        "primary": _summary(analysis),
        "n_models": int(analysis.model.nunique()), "n_pairs": int(analysis.case_id.nunique()),
        "n_excluded_pairs": int(df[df.excluded].case_id.nunique()),
        "models_without_ok_cells": sorted(set(df.model) - set(analysis.model)),
        "non_response": int((df[~df.excluded].status == "non_response").sum()),
        "invariant_correct": float(pairs.invariant_correct.mean()),
        "position_driven": float(pairs.position_driven.mean()),
        "wrong_both": float(pairs.wrong_both.mean()),
    }
    sens = {
        "S1_non_response_incorrect": sensitivity_non_response(df[~df.excluded]),
        "S2_including_excluded": position_summary(_ok(df)),
        "S3_excluding_flagged": position_summary(_ok(df[~df.excluded & ~df.flagged])),
        "S4_excluding_substituted": position_summary(_ok(df[~df.excluded & ~df.substituted])),
    }
    if write:
        OUT.mkdir(exist_ok=True)
        pm = per_model_table(analysis)
        pm.sort_values("position_averaged_accuracy", ascending=False).to_csv(OUT / "per_model.csv", index=False)
        for col, name in (("sham_type", "per_type"), ("category", "per_category")):
            rows = []
            for key, g in analysis.groupby(col):
                fails = int((1 - g.correct).sum())
                p, lo, hi = wilson(fails, len(g))
                rows.append({col: key, "n": len(g), "failures": fails, "failure_rate": p, "lo": lo, "hi": hi,
                             **{k: v for k, v in position_summary(g).items() if k != "n"}})
            pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
        gee_logit(analysis, "correct ~ sham_first + C(category, Treatment('metadata')) + C(model)"
                  ).to_csv(OUT / "gee.csv")
        conf = analysis.dropna(subset=["confidence"])
        pd.DataFrame([calibration(conf.confidence, conf.correct)]).to_csv(OUT / "calibration_overall.csv", index=False)
        pairs.to_csv(OUT / "pair_level.csv", index=False)
        (OUT / "sensitivity.json").write_text(json.dumps(sens, indent=2))
        (analysis.groupby(["model", "sham_type"]).noticed.mean().unstack()
                 .to_csv(OUT / "exploratory_noticed.csv"))
        orig = pd.read_csv(REPO / "analysis_original" / "per_model_discriminable.csv").set_index("model")
        rep = pm.set_index("model")
        common = sorted(set(orig.index) & set(rep.index))
        rho, p = spearmanr(orig.loc[common, "position_averaged_accuracy"],
                           rep.loc[common, "position_averaged_accuracy"])
        (OUT / "exploratory_concordance_original.json").write_text(
            json.dumps({"n_models": len(common), "spearman_rho": rho, "p": p}, indent=2))
        (OUT / "headline.json").write_text(json.dumps(head, indent=2))
    return head


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
