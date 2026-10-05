import json

from r2.paths import ORIGINAL_RUNS_DIR


def load_original():
    manifest = json.loads((ORIGINAL_RUNS_DIR / "manifest.json").read_text())
    records = []
    for entry in manifest["runs"]:
        with open(ORIGINAL_RUNS_DIR / entry["file"]) as fh:
            for line in fh:
                r = json.loads(line)
                r["model_name"] = entry["model"]
                r["run_dir"] = entry["run_dir"]
                records.append(r)
    return records


def sham_position(record):
    for pos, kind in record.get("mapping", {}).items():
        if kind == "S":
            return pos
    return None
