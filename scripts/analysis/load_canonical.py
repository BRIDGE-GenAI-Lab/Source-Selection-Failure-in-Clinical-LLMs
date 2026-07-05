# safe-guideline-tooling-eval/scripts/revision_R1/load_canonical.py
"""
Canonical data loader -- single source of truth for the npj R1 re-analysis.

The results/ directory holds duplicate and partial runs per model. This loader
reconstructs the published canonical set used in the manuscript:

  * one canonical run per model (the run with the most valid records;
    ties -> later directory name, which sorts lexically by timestamp),
  * only models with a *complete* run (>= 500 valid records) are retained --
    partial/abandoned runs (e.g. crashed mid-experiment) are dropped,
  * one non-canonical model (Qwen/Qwen2.5-1.5B-Instruct) that was run but never
    entered the published 21-model table (appendix_stats_10500.json / table_s1)
    is excluded,
  * within the chosen run, records are deduplicated by case_id (some runs append
    a few retry rows with repeated case_ids, e.g. Qwen3-VL-8B at 503 rows),
    yielding exactly 500 unique cases per model.

Result: 21 models x 500 = 10,500 records, reproducing the published aggregates
(overall accuracy ~= 59.4%, sham-in-A ~= 5,29x, sham-in-B ~= 5,21x). This is the
GATE asserted in __main__.
"""
import json, glob, os
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results")

# Models that were run but are NOT part of the published canonical 21-model set
# (absent from appendix_stats_10500.json -> table_s1).
NON_CANONICAL_MODELS = {"Qwen/Qwen2.5-1.5B-Instruct"}

# A complete run for this experiment is 500 cases. Runs below this are partial
# (crashed / abandoned re-runs) and are not part of the canonical set.
MIN_COMPLETE = 500


def _records(run_dir):
    f = os.path.join(run_dir, "results.jsonl")
    if not os.path.exists(f):
        return []
    out = []
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _is_valid(r):
    """A record is valid if it carries a model decision and a scored outcome."""
    return r.get("model_decision") is not None and r.get("selected_tool_correct") is not None


def _dedup_by_case_id(recs):
    """One record per case_id. When a case_id repeats (an appended retry), keep
    a valid (non-error) record over an error stub; otherwise keep the first."""
    seen = {}
    for r in recs:
        cid = r.get("case_id")
        if cid not in seen:
            seen[cid] = r
        elif not _is_valid(seen[cid]) and _is_valid(r):
            # replace an earlier error stub with the successful retry
            seen[cid] = r
    return list(seen.values())


def load_canonical():
    """Return (all_records, by_model) deduped to one run per model, 21 x 500."""
    best = {}  # model -> (n_records, run_dir, records)
    for run_dir in sorted(glob.glob(os.path.join(RESULTS, "run_*"))):
        if os.path.basename(run_dir).startswith("._"):
            continue
        cfg = os.path.join(run_dir, "config.json")
        model = os.path.basename(run_dir)
        if os.path.exists(cfg):
            try:
                model = json.load(open(cfg)).get("model", model)
            except Exception:
                pass
        recs = _records(run_dir)
        # keep run with most records; tie -> later dir name (timestamp sorts lexically)
        if model not in best or len(recs) >= best[model][0]:
            best[model] = (len(recs), run_dir, recs)

    by_model = {}
    all_records = []
    for model, (n, run_dir, recs) in best.items():
        if model in NON_CANONICAL_MODELS:
            continue
        if n < MIN_COMPLETE:
            # partial / abandoned run -> not part of the canonical set
            continue
        recs = _dedup_by_case_id(recs)
        for r in recs:
            r["model_name"] = model
        by_model[model] = recs
        all_records.extend(recs)
    return all_records, by_model


def sham_position(r):
    """Return 'A' or 'B' = position holding the sham ('S')."""
    m = r.get("mapping", {})
    for pos, kind in m.items():
        if kind == "S":
            return pos
    return None


if __name__ == "__main__":
    import numpy as np

    recs, by_model = load_canonical()
    n = len(recs)
    n_models = len(by_model)
    correct = sum(1 for r in recs if r.get("selected_tool_correct") is True)
    acc = correct / n * 100
    sham_a = sum(1 for r in recs if sham_position(r) == "A")
    sham_b = sum(1 for r in recs if sham_position(r) == "B")
    print(f"models={n_models} total={n} acc={acc:.1f}% sham_A={sham_a} sham_B={sham_b}")

    # GATE: published aggregates
    assert n_models == 21, f"expected 21 models, got {n_models}"
    assert n == 10500, f"expected 10500 records, got {n}"
    assert abs(acc - 59.4) < 1.0, f"accuracy {acc:.1f} != published 59.4"
    print("GATE PASS: reproduces published aggregates")

    # Authoritative headline numbers (resolve QC items 4-7).
    def sel_A(r):
        return r.get("model_decision", {}).get("selected_tool") == "A"

    primacy = np.mean([sel_A(r) for r in recs]) * 100
    acc_sham_A = np.mean([r["selected_tool_correct"] for r in recs if sham_position(r) == "A"]) * 100
    acc_sham_B = np.mean([r["selected_tool_correct"] for r in recs if sham_position(r) == "B"]) * 100
    os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
    json.dump(
        {
            "primacy_pct": round(primacy, 2),
            "acc_sham_A": round(acc_sham_A, 1),
            "acc_sham_B": round(acc_sham_B, 1),
            "swing_pp": round(acc_sham_B - acc_sham_A, 1),
        },
        open(os.path.join(os.path.dirname(__file__), "outputs", "canonical_headline.json"), "w"),
        indent=2,
    )
    print(
        f"primacy={primacy:.2f}% acc_sham_A={acc_sham_A:.1f}% "
        f"acc_sham_B={acc_sham_B:.1f}% swing={acc_sham_B-acc_sham_A:.1f}pp"
    )
