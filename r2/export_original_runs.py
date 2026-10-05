"""Export the 21 canonical original runs (selection rule of load_canonical.py) into the repo."""
import glob
import hashlib
import json
import os
import sys

from r2.paths import LEGACY_EVAL, LEGACY_RESULTS, ORIGINAL_RUNS_DIR


def _legacy_loader():
    sys.path.insert(0, str(LEGACY_EVAL / "scripts" / "revision_R1"))
    import load_canonical
    return load_canonical


def select_canonical_runs(lc):
    """Same rule as load_canonical.load_canonical(), but keeps the run directory."""
    best = {}
    for run_dir in sorted(glob.glob(str(LEGACY_RESULTS / "run_*"))):
        if os.path.basename(run_dir).startswith("._"):
            continue
        cfg = os.path.join(run_dir, "config.json")
        model = os.path.basename(run_dir)
        if os.path.exists(cfg):
            model = json.load(open(cfg)).get("model", model)
        recs = lc._records(run_dir)
        if model not in best or len(recs) >= best[model][0]:
            best[model] = (len(recs), run_dir, recs)
    out = {}
    for model, (n, run_dir, recs) in best.items():
        if model in lc.NON_CANONICAL_MODELS or n < lc.MIN_COMPLETE:
            continue
        out[model] = (run_dir, lc._dedup_by_case_id(recs))
    return out


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    lc = _legacy_loader()
    runs = select_canonical_runs(lc)
    reference, _ = lc.load_canonical()
    ref_keys = {(r["model_name"], r["case_id"], json.dumps(r["model_decision"], sort_keys=True)) for r in reference}
    ORIGINAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"selection_rule": lc.__doc__.strip(), "runs": []}
    got_keys = set()
    for model in sorted(runs):
        run_dir, recs = runs[model]
        name = os.path.basename(run_dir)
        dest = ORIGINAL_RUNS_DIR / (name + ".results.jsonl")
        with open(dest, "w") as fh:
            for r in recs:
                r = {k: v for k, v in r.items() if k != "model_name"}
                fh.write(json.dumps(r, sort_keys=True) + "\n")
                got_keys.add((model, r["case_id"], json.dumps(r["model_decision"], sort_keys=True)))
        cfg_path = os.path.join(run_dir, "config.json")
        manifest["runs"].append({
            "model": model,
            "run_dir": name,
            "file": dest.name,
            "n_records": len(recs),
            "source_results_sha256": _sha256(os.path.join(run_dir, "results.jsonl")),
            "config": json.load(open(cfg_path)) if os.path.exists(cfg_path) else {},
        })
    assert got_keys == ref_keys, "export differs from load_canonical() output"
    (ORIGINAL_RUNS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"exported {len(manifest['runs'])} runs, {len(got_keys)} records")


if __name__ == "__main__":
    main()
