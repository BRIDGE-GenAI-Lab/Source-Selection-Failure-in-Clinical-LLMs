"""Pre-specified analyses of the repeated experiment (docs/r2/ANALYSIS_PLAN.md).
Exploratory outputs are written with an 'exploratory_' prefix."""
import json
import math
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from r2.analysis.original_corrected import SAYS_IDENTICAL
from r2.analysis.stats import (CATEGORY, calibration, cluster_bootstrap_mean, cluster_bootstrap_paa, fdr,
                               gee_logit, position_summary, wilson)
from r2.paths import DATA_V2, REPO, RUNS_V2, VERIFY_DIR
from r2.run.harness import INCIDENTS, final_records, load_specs, read_records, results_path

OUT = REPO / "analysis_repeated"
# Items for which r2.verify.score_review.flag_reasons flags a pair (flags.json values).
FLAG_ITEMS = ("q1_vignette_consistent", "q2_question_answerable", "q3_authentic_concordant", "q8_leakage")
PAIR_COLS = ["model", "case_id", "invariant_correct", "position_driven", "wrong_both"]
# S5 (ruling 2026-10-05): the evaluated model of the same family as the vignette generator (Gemini 3 Pro),
# identified by its original_name in r2/run/models_v2.json.
S5_MODEL = "gemini-2.5-flash-lite"
SENSITIVITY = {
    "S1_non_response_incorrect": "Non-responses counted as incorrect",
    "S2_including_excluded": "Including the pairs excluded by the physicians",
    "S3_excluding_flagged": "Excluding the pairs flagged in physician verification",
    "S4_excluding_substituted": "Excluding substituted models",
    "S5_excluding_generator_family": "Excluding Gemini-2.5-Flash-Lite (same family as the vignette generator)",
}
# The documents name the bootstrap CI of the primary outcome primary.paa_lo / primary.paa_hi
# (PENDING_INDEX.md); analysis_original/headline.json names the same quantity
# position_averaged_accuracy_lo / _hi, which edits/numbers.py reads. Every summary written by this module
# carries both names: alias -> canonical key.
HEADLINE_ALIASES = {"position_averaged_accuracy_lo": "paa_lo", "position_averaged_accuracy_hi": "paa_hi"}


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


Verification = namedtuple("Verification", "excluded flagged rule pairs_v2_sha256 synthetic")


def load_verification(verify_dir, case_ids):
    """Read and validate the physician-verification outputs that r2.verify.score_review.apply_consensus
    writes: excluded_pairs.json {"rule": str, "excluded": [case_id], "pairs_v2_sha256": ...} and flags.json
    {case_id: [flag item, ...]}. A synthetic world (r2.analysis.synthetic) adds "synthetic": true to both;
    no other extra key is accepted in flags.json."""
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
    flags = dict(flags)
    marker = flags.pop("synthetic", None)
    if marker not in (None, True):
        raise ValueError(f"{verify_dir}/flags.json: 'synthetic' may only be true, got {marker!r}")
    for cid, items in flags.items():
        if cid not in case_ids:
            raise ValueError(f"{verify_dir}/flags.json: {cid} is not in pairs_v2.json")
        if not isinstance(items, list) or not items or not set(items) <= set(FLAG_ITEMS):
            raise ValueError(f"{verify_dir}/flags.json: {cid} must map to a non-empty list of {FLAG_ITEMS}, "
                             f"got {items!r}")
    return Verification(set(ex["excluded"]), set(flags), ex["rule"], ex.get("pairs_v2_sha256"),
                        ex.get("synthetic") is True or marker is True)


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


RECORD_COLS = ["model", "case_id", "order", "status", "sham_first", "selected_first", "correct", "confidence",
               "differences", "rationale", "finished_at"]
FRAME_COLS = RECORD_COLS + ["sham_type", "category", "excluded", "flagged", "substituted", "noticed"]


def _noticed_value(status, differences, case):
    """E1 per record: 1.0 / 0.0, or NaN when not assessable (no ok response, or no edit snippet of 3+ words)."""
    if status != "ok":
        return np.nan
    hit = noticed(differences, _snippets(case))
    return np.nan if hit is None else float(hit)


def repeated_frame(out_dir=None, *, pairs_path=None, verify_dir=None, specs=None):
    """One row per final record (last record per cell) of every current spec's result file.
    `noticed` (E1) is 1.0 / 0.0, or NaN when not assessable (no ok response, or no edit snippet of 3+ words)."""
    out_dir = RUNS_V2 / "main" if out_dir is None else out_dir
    pairs_path = DATA_V2 / "pairs_v2.json" if pairs_path is None else pairs_path
    doc = json.loads(Path(pairs_path).read_text())
    pairs = {c["case_id"]: c for c in doc["cases"]}
    ver = load_verification(VERIFY_DIR if verify_dir is None else verify_dir, set(pairs))
    excluded, flagged = ver.excluded, ver.flagged
    synthetic = doc.get("synthetic") is True or ver.synthetic
    specs = load_specs() if specs is None else specs
    substituted = {s["original_name"] for s in specs if s.get("substituted_for")}
    rows = []
    for spec in specs:
        path = results_path(spec, out_dir)
        if not path.exists():
            continue
        for r in final_records(path):
            synthetic = synthetic or r.get("synthetic") is True
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
    df = pd.DataFrame(rows, columns=RECORD_COLS)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["sham_type"] = df.case_id.map(lambda c: pairs[c]["eval"]["sham_trap_type"])
    df["category"] = df.sham_type.map(CATEGORY)
    df["excluded"] = df.case_id.isin(excluded)
    df["flagged"] = df.case_id.isin(flagged)
    df["substituted"] = df.model.isin(substituted)
    df["noticed"] = pd.Series([_noticed_value(status, diff, pairs[c])
                               for status, diff, c in zip(df.status, df.differences, df.case_id)],
                              index=df.index, dtype=float)
    df.attrs.update({"exclusion_rule": ver.rule, "verification": ver, "synthetic": synthetic})
    return df


def _pair_units(ok):
    """Model-pair units with an ok record in both orders; the three outcomes are exclusive and exhaustive."""
    rows = []
    for (model, cid), g in ok.groupby(["model", "case_id"]):
        if len(g) != 2 or g.sham_first.nunique() != 2:
            continue
        rows.append({"model": model, "case_id": cid,
                     "invariant_correct": int(g.correct.sum() == 2),
                     "position_driven": int(g.selected_first.nunique() == 1),
                     "wrong_both": int(g.correct.sum() == 0)})
    return pd.DataFrame(rows, columns=PAIR_COLS)


def pair_level(df):
    if df.empty or "status" not in df:
        return pd.DataFrame(columns=PAIR_COLS)
    return _pair_units(df[df.status == "ok"])


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


def _rate_ci(rows, col):
    """Case-cluster bootstrap 95% CI of the pooled mean of `col` over `rows`."""
    return cluster_bootstrap_mean(rows.assign(_rate=pd.to_numeric(rows[col], errors="coerce").astype(float)),
                                  "_rate")


def _summary(df, primacy=True):
    """Pooled position summary with a case-cluster bootstrap 95% CI (2,000 resamples of vignettes, seed
    20261005) for every rate: accuracy (acc_lo/acc_hi), primacy, accuracy with the modified excerpt first
    and second (each over that order's rows), and position-averaged accuracy (paa_lo/paa_hi)."""
    s = _pooled(df)
    s["acc_lo"], s["acc_hi"] = _rate_ci(df, "correct")
    if primacy:
        s["primacy_lo"], s["primacy_hi"] = _rate_ci(df, "selected_first")
    else:
        s["primacy"] = s["primacy_lo"] = s["primacy_hi"] = float("nan")
    for name, rows in (("acc_sham_first", df[df.sham_first]), ("acc_sham_second", df[~df.sham_first])):
        s[f"{name}_lo"], s[f"{name}_hi"] = _rate_ci(rows, "correct")
    s["paa_lo"], s["paa_hi"] = cluster_bootstrap_paa(df)
    for alias, key in HEADLINE_ALIASES.items():
        s[alias] = s[key]
    s["n_models"], s["n_pairs"] = int(df.model.nunique()), int(df.case_id.nunique())
    return s


def sensitivity_non_response(df):
    """Non-responses scored incorrect. Selection is undefined for them, so only n, accuracy and
    position-averaged accuracy are meaningful; primacy is set to NaN."""
    d = df.copy()
    d["correct"] = pd.to_numeric(d.correct, errors="coerce").fillna(0).astype(int)
    d["selected_first"] = [bool(v) if isinstance(v, (bool, np.bool_)) else False for v in d.selected_first]
    return _summary(d, primacy=False)


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


def _wilson_rate(flags):
    """Wilson 95% CI for a single proportion; NaN for an empty arm."""
    flags = pd.Series(flags, dtype=float)
    return wilson(int(flags.sum()), len(flags))


def per_model_table(analysis):
    """One row per model. Single proportions (accuracy with the modified excerpt first / second, primacy)
    carry Wilson 95% CIs; position-averaged accuracy carries a case-cluster bootstrap CI (ruling 13b I6);
    content sensitivity is the exact McNemar test (M12) with BH-FDR across models."""
    per_model = []
    cal = calibration_table(analysis, "model").set_index("model") if len(analysis) else None
    for model, g in analysis.groupby("model"):
        s = _pooled(g)
        sf, ss = g[g.sham_first], g[~g.sham_first]
        s["n_sham_first"], s["n_sham_second"] = len(sf), len(ss)
        for col, flags in (("acc_sham_first", sf.correct), ("acc_sham_second", ss.correct),
                           ("primacy", g.selected_first)):
            _, s[f"{col}_lo"], s[f"{col}_hi"] = _wilson_rate(flags)
        s["paa_lo"], s["paa_hi"] = cluster_bootstrap_paa(g)
        s["mcnemar_n10"], s["mcnemar_n01"], s["mcnemar_p"] = mcnemar(g)
        units = _pair_units(g)
        s["n_pair_units"] = len(units)
        for k in PAIR_COLS[2:]:
            s[k] = float(units[k].mean()) if len(units) else float("nan")
        s.update({"model": model, **cal.loc[model].to_dict()})
        per_model.append(s)
    pm = pd.DataFrame(per_model)
    pm["mcnemar_p_fdr"] = float("nan")
    valid = pm.mcnemar_p.notna()
    if valid.any():
        pm.loc[valid, "mcnemar_p_fdr"] = fdr(pm.loc[valid, "mcnemar_p"])
    return pm


def sensitivity_summaries(df):
    """S1-S5 of ANALYSIS_PLAN.md, each the primary outcome on a different analysis set, with bootstrap CIs."""
    keep = ~df.excluded
    sens = {
        "S1_non_response_incorrect": sensitivity_non_response(df[keep]),
        "S2_including_excluded": _summary(_ok(df)),
        "S3_excluding_flagged": _summary(_ok(df[keep & ~df.flagged])),
        "S4_excluding_substituted": _summary(_ok(df[keep & ~df.substituted])),
        "S5_excluding_generator_family": _summary(_ok(df[keep & (df.model != S5_MODEL)])),
    }
    sens["S4_excluding_substituted"]["excluded_models"] = sorted(set(df.model[df.substituted]))
    sens["S5_excluding_generator_family"]["excluded_models"] = [S5_MODEL]
    return sens


def rate_table(analysis, col):
    """Failure rate per level of `col`, pooled across models; CI = case-cluster bootstrap."""
    rows = []
    for key, g in analysis.groupby(col):
        fails = int((1 - g.correct).sum())
        lo, hi = cluster_bootstrap_mean(g.assign(failure=1 - g.correct), "failure")
        row = {col: key}
        if col == "sham_type":
            row["category"] = CATEGORY[key]
        row.update({"n": len(g), "n_pairs": int(g.case_id.nunique()), "failures": fails,
                    "failure_rate": fails / len(g), "lo": lo, "hi": hi,
                    **{k: v for k, v in _pooled(g).items() if k != "n"}})
        rows.append(row)
    return pd.DataFrame(rows)


def rationale_summary(analysis):
    """E3 (exploratory): rationales that call the two excerpts identical or equivalent, by the keyword match
    used for the original responses (r2.analysis.original_corrected.SAYS_IDENTICAL)."""
    says = analysis.rationale.fillna("").map(lambda t: bool(SAYS_IDENTICAL.search(t))).astype(float)
    lo, hi = cluster_bootstrap_mean(analysis.assign(_says=says), "_says")
    return {"n": len(analysis), "says_identical": float(says.mean()) if len(says) else float("nan"),
            "says_identical_lo": lo, "says_identical_hi": hi}


def noticed_summary(analysis):
    """E1 (exploratory), pooled: share of assessable ok responses whose differences_found shares a 3-word
    sequence with the edited content; responses whose edit has no 3-word snippet are not assessable."""
    lo, hi = cluster_bootstrap_mean(analysis, "noticed")
    n_ass = int(analysis.noticed.notna().sum())
    return {"n_ok": len(analysis), "n_assessable": n_ass, "n_not_assessable": len(analysis) - n_ass,
            "noticed": float(analysis.noticed.mean()) if n_ass else float("nan"), "noticed_lo": lo, "noticed_hi": hi}


CALL_DATES_BASIS = ("finished_at of every logged record in each current model's result file, including "
                    "superseded error attempts, and logged_at of every incidents.jsonl entry")


def _iso_range(stamps):
    ts = pd.to_datetime(pd.Series(stamps, dtype=object), utc=True, errors="coerce").dropna()
    if ts.empty:
        return {"first": None, "last": None}
    return {"first": ts.min().isoformat(), "last": ts.max().isoformat()}


def call_dates(out_dir, specs):
    """First and last call of the main run, overall and per model, from EVERY logged record (not only the
    final record per cell: an early API error that was retried later still dates the first request) and
    from the incidents log (a model-mismatch response is a paid call too)."""
    out_dir = Path(out_dir)
    stamps = {}
    for spec in specs:
        path = results_path(spec, out_dir)
        if path.exists():
            for r in read_records(path):
                stamps.setdefault(r.get("original_name", spec["original_name"]), []).append(r.get("finished_at"))
    incidents = out_dir / INCIDENTS
    current = {s["original_name"] for s in specs}
    if incidents.exists():
        for r in read_records(incidents):
            if r.get("original_name") in current:
                stamps.setdefault(r["original_name"], []).append(r.get("logged_at"))
    per_model = {m: _iso_range(v) for m, v in sorted(stamps.items())}
    per_model = {m: d for m, d in per_model.items() if d["first"]}
    overall = _iso_range([d[k] for d in per_model.values() for k in ("first", "last")])
    return {**overall, "per_model": per_model, "basis": CALL_DATES_BASIS}


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
    """Run every pre-specified analysis and the exploratory ones. Inputs default to this module's RUNS_V2 / "main",
    DATA_V2, VERIFY_DIR and load_specs() (from r2.paths), outputs to OUT; a sandbox passes its own paths."""
    specs = load_specs() if specs is None else specs
    if S5_MODEL not in {sp["original_name"] for sp in specs}:
        raise ValueError(f"S5 model {S5_MODEL!r} is not an original_name in the model specs")
    df = repeated_frame(out_dir, pairs_path=pairs_path, verify_dir=verify_dir, specs=specs)
    kept = df[~df.excluded]
    analysis = _ok(kept)
    pairs = pair_level(kept)
    head = {
        "primary": _summary(analysis),
        "n_models": int(analysis.model.nunique()), "n_pairs": int(analysis.case_id.nunique()),
        "n_types": int(analysis.sham_type.nunique()),
        "n_excluded_pairs": int(df[df.excluded].case_id.nunique()),
        "n_flagged_pairs": int(kept[kept.flagged].case_id.nunique()),
        "n_cells": int(len(kept)),
        "models_without_ok_cells": sorted(set(df.model) - set(analysis.model)),
        "non_response": int((kept.status == "non_response").sum()),
        "n_pair_units": int(len(pairs)),
        "rep_call_dates": call_dates(RUNS_V2 / "main" if out_dir is None else out_dir, specs),
        "exclusion_rule": df.attrs.get("exclusion_rule"),
    }
    for k in ("invariant_correct", "position_driven", "wrong_both"):
        head[k] = float(pairs[k].mean()) if len(pairs) else float("nan")
        head[f"{k}_lo"], head[f"{k}_hi"] = cluster_bootstrap_mean(pairs, k)
    synthetic = bool(df.attrs.get("synthetic", False))
    head["synthetic"] = synthetic
    sens = sensitivity_summaries(df)
    if write:
        out = Path(OUT if analysis_dir is None else analysis_dir)
        if synthetic:  # synthetic numbers never land in a real output location
            from r2.analysis.synthetic import assert_sandbox
            assert_sandbox(out)
        out.mkdir(parents=True, exist_ok=True)

        def to_csv(frame, name, **kw):
            (frame.assign(synthetic=True) if synthetic else frame).to_csv(out / name, **kw)

        def to_json(name, obj):
            write_json(out / name, {**obj, "synthetic": synthetic})
        pm = per_model_table(analysis)
        to_csv(pm.sort_values("position_averaged_accuracy", ascending=False), "per_model.csv", index=False)
        for col, name in (("sham_type", "per_type"), ("category", "per_category")):
            to_csv(rate_table(analysis, col), f"{name}.csv", index=False)
        to_csv(gee_logit(analysis, "correct ~ sham_first + C(category, Treatment('metadata')) + C(model)"), "gee.csv")
        to_csv(calibration_table(analysis), "calibration_overall.csv", index=False)
        to_csv(calibration_table(analysis, "model"), "calibration_per_model.csv", index=False)
        to_csv(pairs, "pair_level.csv", index=False)
        to_json("sensitivity.json", sens)
        to_csv(analysis.groupby(["model", "sham_type"]).noticed.mean().unstack(), "exploratory_noticed.csv")
        original_csv = REPO / "analysis_original" / "per_model_discriminable.csv" if original_csv is None \
            else original_csv
        to_json("exploratory_concordance_original.json", concordance(pm, original_csv))
        to_json("exploratory_noticed_summary.json", noticed_summary(analysis))
        to_json("exploratory_rationale.json", rationale_summary(analysis))
        to_json("headline.json", head)
    return head


if __name__ == "__main__":
    print(json.dumps(_plain(run()), indent=2))
