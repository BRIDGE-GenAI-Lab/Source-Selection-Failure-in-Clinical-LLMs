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
            model = _read_json(cfg).get("model", model)
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


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_export(lc):
    """Everything main() writes, computed in memory: (manifest, {file name: bytes}, got_keys)."""
    runs = select_canonical_runs(lc)
    reference, _ = lc.load_canonical()
    ref_keys = {(r["model_name"], r["case_id"], json.dumps(r["model_decision"], sort_keys=True)) for r in reference}
    manifest = {"selection_rule": lc.__doc__.strip(), "runs": []}
    files, got_keys = {}, set()
    for model in sorted(runs):
        run_dir, recs = runs[model]
        name = os.path.basename(run_dir)
        dest_name = name + ".results.jsonl"
        lines = []
        for r in recs:
            r = {k: v for k, v in r.items() if k != "model_name"}
            lines.append(json.dumps(r, sort_keys=True) + "\n")
            got_keys.add((model, r["case_id"], json.dumps(r["model_decision"], sort_keys=True)))
        data = "".join(lines).encode("utf-8")
        files[dest_name] = data
        cfg_path = os.path.join(run_dir, "config.json")
        manifest["runs"].append({
            "model": model,
            "run_dir": name,
            "file": dest_name,
            "n_records": len(recs),
            "source_results_sha256": _sha256(os.path.join(run_dir, "results.jsonl")),
            "exported_sha256": hashlib.sha256(data).hexdigest(),
            "config": _read_json(cfg_path) if os.path.exists(cfg_path) else {},
        })
    if got_keys != ref_keys:
        raise RuntimeError("export differs from load_canonical() output: "
                           f"{len(got_keys - ref_keys)} extra and {len(ref_keys - got_keys)} missing record keys")
    return manifest, files, got_keys


def main():
    manifest, files, got_keys = build_export(_legacy_loader())  # raises before anything is written
    ORIGINAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (ORIGINAL_RUNS_DIR / name).write_bytes(data)
    (ORIGINAL_RUNS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"exported {len(manifest['runs'])} runs, {len(got_keys)} records")


if __name__ == "__main__":
    main()
