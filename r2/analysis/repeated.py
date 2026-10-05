"""Pre-specified analyses of the repeated experiment (docs/r2/ANALYSIS_PLAN.md).
Exploratory outputs are written with an 'exploratory_' prefix."""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from r2.analysis.stats import (CATEGORY, calibration, cluster_bootstrap_mean, cluster_bootstrap_paa, fdr,
                               gee_logit, position_summary)
from r2.paths import DATA_V2, REPO, RUNS_V2, VERIFY_DIR
from r2.run.harness import final_records, load_specs, results_path

OUT = REPO / "analysis_repeated"
# Items for which r2.verify.score_review.flag_reasons flags a pair (flags.json values).
FLAG_ITEMS = ("q1_vignette_consistent", "q2_question_answerable", "q3_authentic_concordant", "q8_leakage")
PAIR_COLS = ["model", "case_id", "invariant_correct", "position_driven", "wrong_both"]


def _plain(x):
    """JSON-safe copy: NaN and +/-inf become None (JSON null); numpy scalars become Python scalars."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path, obj):
    Path(path).write_text(json.dumps(_plain(obj), indent=2, allow_nan=False) + "\n")


def load_verification(verify_dir, case_ids):
    """Read and validate the physician-verification outputs that r2.verify.score_review.apply_consensus
    writes: excluded_pairs.json {"rule": str, "excluded": [case_id], ...} and flags.json
    {case_id: [flag item, ...]}. Returns (excluded, flagged, rule)."""
    verify_dir = Path(verify_dir)
    ex = json.loads((verify_dir / "excluded_pairs.json").read_text())
    if not isinstance(ex, dict) or not isinstance(ex.get("rule"), str) or not isinstance(ex.get("excluded"), list) \
            or not all(isinstance(c, str) for c in ex["excluded"]):
        raise ValueError(f"{verify_dir}/excluded_pairs.json: expected {{'rule': str, 'excluded': [case_id]}}")
    unknown = sorted(set(ex["excluded"]) - set(case_ids))
    if unknown:
        raise ValueError(f"{verify_dir}/excluded_pairs.json: case_ids not in pairs_v2.json: {unknown[:10]}")
    flags = json.loads((verify_dir / "flags.json").read_text())
    if not isinstance(flags, dict):
        raise ValueError(f"{verify_dir}/flags.json: expected an object {{case_id: [flag item]}}")
    for cid, items in flags.items():
        if cid not in case_ids:
            raise ValueError(f"{verify_dir}/flags.json: {cid} is not in pairs_v2.json")
        if not isinstance(items, list) or not items or not set(items) <= set(FLAG_ITEMS):
            raise ValueError(f"{verify_dir}/flags.json: {cid} must map to a non-empty list of {FLAG_ITEMS}, "
                             f"got {items!r}")
    return set(ex["excluded"]), set(flags), ex["rule"]


def _tokens(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _grams(tokens, k):
    return {tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def noticed(differences, snippets, k=3):
    """E1 (exploratory): does any reported difference share a k-word sequence (k = 3, ANALYSIS_PLAN E1)
    with the edited content? Snippets shorter than k words cannot meet the rule and are ignored; when
    no snippet has k words the response is not assessable and the result is None (never False)."""
    grams = set()
    for s in snippets:
        grams |= _grams(_tokens(s), k)
    if not grams:
        return None
    return any(_grams(_tokens(d), k) & grams for d in differences or [])


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


FRAME_COLS = ["model", "case_id", "order", "status", "sham_first", "selected_first", "correct", "confidence",
              "differences", "rationale", "finished_at", "sham_type", "category", "excluded", "flagged",
              "substituted", "noticed"]


def repeated_frame(out_dir=None, *, pairs_path=None, verify_dir=None, specs=None):
    """One row per final record (last record per cell) of every current spec's result file.
    `noticed` (E1) is 1.0 / 0.0, or NaN when not assessable (no ok response, or no edit snippet of 3+ words)."""
    out_dir = RUNS_V2 / "main" if out_dir is None else out_dir
    pairs_path = DATA_V2 / "pairs_v2.json" if pairs_path is None else pairs_path
    pairs = {c["case_id"]: c for c in json.loads(Path(pairs_path).read_text())["cases"]}
    excluded, flagged, rule = load_verification(VERIFY_DIR if verify_dir is None else verify_dir, set(pairs))
    specs = load_specs() if specs is None else specs
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
                "rationale": d.get("trust_rationale") or "", "finished_at": r.get("finished_at"),
            })
    df = pd.DataFrame(rows, columns=FRAME_COLS[:11])
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["sham_type"] = df.case_id.map(lambda c: pairs[c]["eval"]["sham_trap_type"])
    df["category"] = df.sham_type.map(CATEGORY)
    df["excluded"] = df.case_id.isin(excluded)
    df["flagged"] = df.case_id.isin(flagged)
    df["substituted"] = df.model.isin(substituted)
    df["noticed"] = pd.Series([
        np.nan if status != "ok" or (v := noticed(diff, _snippets(pairs[c]))) is None else float(v)
        for diff, c, status in zip(df.differences, df.case_id, df.status)], index=df.index, dtype=float)
    df.attrs["exclusion_rule"] = rule
    return df


def pair_level(df):
    """One row per (model, pair) with an ok record in both orders; the three outcomes are exclusive."""
    if df.empty or "status" not in df:
        return pd.DataFrame(columns=PAIR_COLS)
    rows = []
    for (model, cid), g in df[df.status == "ok"].groupby(["model", "case_id"]):
        if len(g) != 2:
            continue
        rows.append({"model": model, "case_id": cid,
                     "invariant_correct": int(g.correct.sum() == 2),
                     "position_driven": int(g.selected_first.nunique() == 1),
                     "wrong_both": int(g.correct.sum() == 0)})
    return pd.DataFrame(rows, columns=PAIR_COLS)


def _ok(df):
    out = df[df.status == "ok"].copy()
    out["correct"] = out.correct.astype(int)
    out["selected_first"] = out.selected_first.astype(bool)
    return out


def _pooled(df):
    """Position summary pooled across models, without the unpaired two-proportion z-test (it treats the
    correlated evaluations of one pair as independent); the case-clustered GEE carries inference."""
    s = position_summary(df)
    s.pop("content_z")
    s.pop("content_p")
    return s


def sensitivity_non_response(df):
    """Non-responses scored incorrect. Selection is undefined for them, so only n, accuracy and
    position-averaged accuracy are meaningful; primacy is set to NaN."""
    d = df.copy()
    d["correct"] = pd.to_numeric(d.correct, errors="coerce").fillna(0).astype(int)
    d["selected_first"] = [bool(v) if isinstance(v, (bool, np.bool_)) else False for v in d.selected_first]
    s = _pooled(d)
    s["primacy"] = float("nan")
    return s


def mcnemar(g):
    """Exact McNemar test of content sensitivity on pairs with an ok record in both orders.
    n10: T selected in both orders (first in TS, second in ST); n01: S selected in both orders.
    Position-driven pairs (same slot in both orders) are concordant and carry no information."""
    counts = g.groupby("case_id").sham_first.agg(["sum", "count"])
    keep = counts.index[(counts["count"] == 2) & (counts["sum"] == 1)]
    both = g[g.case_id.isin(keep)]
    sel = both.set_index("case_id").selected_first.astype(bool)
    sham_first = both.sham_first.astype(bool).to_numpy()
    ts, st = sel[~sham_first], sel[sham_first].reindex(sel[~sham_first].index)
    n10, n01 = int((ts & ~st).sum()), int((~ts & st).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else float("nan")
    return n10, n01, float(p)


def per_model_table(analysis):
    nan = float("nan")
    per_model = []
    for model, g in analysis.groupby("model"):
        s = _pooled(g)
        s["paa_lo"], s["paa_hi"] = cluster_bootstrap_paa(g)
        s["mcnemar_n10"], s["mcnemar_n01"], s["mcnemar_p"] = mcnemar(g)
        conf = g.dropna(subset=["confidence"])
        cal = calibration(conf.confidence, conf.correct) if len(conf) else {
            k: nan for k in ("ece", "mce", "brier", "auroc", "mean_conf", "accuracy")}
        cal = {("accuracy_conf" if k == "accuracy" else k): v for k, v in cal.items()}
        s.update({"model": model, **cal})
        per_model.append(s)
    pm = pd.DataFrame(per_model)
    pm["mcnemar_p_fdr"] = float("nan")
    valid = pm.mcnemar_p.notna()
    if valid.any():
        pm.loc[valid, "mcnemar_p_fdr"] = fdr(pm.loc[valid, "mcnemar_p"])
    return pm


def _summary(df):
    s = _pooled(df)
    s["paa_lo"], s["paa_hi"] = cluster_bootstrap_paa(df)
    s["primacy_lo"], s["primacy_hi"] = cluster_bootstrap_mean(
        df.assign(first=df.selected_first.astype(float)), "first")
    return s


def sensitivity_summaries(df):
    return {
        "S1_non_response_incorrect": sensitivity_non_response(df[~df.excluded]),
        "S2_including_excluded": _pooled(_ok(df)),
        "S3_excluding_flagged": _pooled(_ok(df[~df.excluded & ~df.flagged])),
        "S4_excluding_substituted": _pooled(_ok(df[~df.excluded & ~df.substituted])),
    }


def rate_table(analysis, col):
    """Failure rate per level of `col`, pooled across models; CI = case-cluster bootstrap."""
    rows = []
    for key, g in analysis.groupby(col):
        fails = int((1 - g.correct).sum())
        lo, hi = cluster_bootstrap_mean(g.assign(failure=1 - g.correct), "failure")
        rows.append({col: key, "n": len(g), "failures": fails, "failure_rate": fails / len(g),
                     "lo": lo, "hi": hi, **{k: v for k, v in _pooled(g).items() if k != "n"}})
    return pd.DataFrame(rows)


CAL_COLS = ("ece", "mce", "brier", "auroc", "mean_conf", "accuracy")


def calibration_table(analysis, by=None):
    """Calibration of confidence for correctness (ANALYSIS_PLAN item 7) on rows with a confidence in [0, 1];
    overall, or one row per level of `by`. `accuracy_conf` is the accuracy of those rows."""
    groups = [(None, analysis)] if by is None else list(analysis.groupby(by))
    rows = []
    for key, g in groups:
        conf = g.dropna(subset=["confidence"])
        cal = calibration(conf.confidence, conf.correct) if len(conf) else {k: float("nan") for k in CAL_COLS}
        row = {} if by is None else {by: key}
        row.update({"n_conf": len(conf), **{("accuracy_conf" if k == "accuracy" else k): v for k, v in cal.items()}})
        rows.append(row)
    return pd.DataFrame(rows)


def concordance(per_model, original_csv):
    """E2 (exploratory): Spearman correlation of per-model position-averaged accuracy, original experiment
    (discriminable pairs) vs repeated experiment, over the models in both. Undefined (None) below 3 models."""
    orig = pd.read_csv(original_csv).set_index("model")
    rep = per_model.set_index("model")
    common = sorted(set(orig.index) & set(rep.index))
    rho = p = float("nan")
    if len(common) >= 3:
        rho, p = spearmanr(orig.loc[common, "position_averaged_accuracy"], rep.loc[common, "position_averaged_accuracy"])
    return {"n_models": len(common), "spearman_rho": float(rho), "p": float(p)}


def run(out_dir=None, write=True, *, analysis_dir=None, pairs_path=None, verify_dir=None, specs=None,
        original_csv=None):
    """Every path defaults to r2.paths at call time; a sandbox passes its own."""
    df = repeated_frame(out_dir, pairs_path=pairs_path, verify_dir=verify_dir, specs=specs)
    analysis = _ok(df[~df.excluded])
    pairs = pair_level(df[~df.excluded])
    head = {
        "primary": _summary(analysis),
        "n_models": int(analysis.model.nunique()), "n_pairs": int(analysis.case_id.nunique()),
        "n_excluded_pairs": int(df[df.excluded].case_id.nunique()),
        "models_without_ok_cells": sorted(set(df.model) - set(analysis.model)),
        "non_response": int((df[~df.excluded].status == "non_response").sum()),
    }
    for k in ("invariant_correct", "position_driven", "wrong_both"):
        head[k] = float(pairs[k].mean()) if len(pairs) else float("nan")
        head[f"{k}_lo"], head[f"{k}_hi"] = cluster_bootstrap_mean(pairs, k)
    sens = sensitivity_summaries(df)
    if write:
        out = Path(OUT if analysis_dir is None else analysis_dir)
        out.mkdir(parents=True, exist_ok=True)
        pm = per_model_table(analysis)
        pm.sort_values("position_averaged_accuracy", ascending=False).to_csv(out / "per_model.csv", index=False)
        for col, name in (("sham_type", "per_type"), ("category", "per_category")):
            rate_table(analysis, col).to_csv(out / f"{name}.csv", index=False)
        gee_logit(analysis, "correct ~ sham_first + C(category, Treatment('metadata')) + C(model)"
                  ).to_csv(out / "gee.csv")
        calibration_table(analysis).to_csv(out / "calibration_overall.csv", index=False)
        calibration_table(analysis, "model").to_csv(out / "calibration_per_model.csv", index=False)
        pairs.to_csv(out / "pair_level.csv", index=False)
        write_json(out / "sensitivity.json", sens)
        (analysis.groupby(["model", "sham_type"]).noticed.mean().unstack()
                 .to_csv(out / "exploratory_noticed.csv"))
        original_csv = REPO / "analysis_original" / "per_model_discriminable.csv" if original_csv is None \
            else original_csv
        write_json(out / "exploratory_concordance_original.json", concordance(pm, original_csv))
        write_json(out / "headline.json", head)
    return head


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
