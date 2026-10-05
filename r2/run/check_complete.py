import csv
import json
import sys
from collections import Counter
from pathlib import Path

from r2.paths import DATA_V2, REPO, RUNS_V2
from r2.run.harness import ORDERS, iter_result_files, load_specs, load_v2_cases, results_path


def check(out_dir, specs, case_ids):
    report = []
    expected = {(c, o) for c in case_ids for o in ORDERS}
    for spec in specs:
        path = results_path(spec, out_dir)
        cells, statuses = Counter(), Counter()
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    cells[(r["case_id"], r["order"])] += 1
                    statuses[r["status"]] += 1
        report.append({"model": spec["original_name"], "cells": len(cells),
                       "missing": len(expected - set(cells)),
                       "duplicates": sum(1 for v in cells.values() if v > 1),
                       "unexpected": len(set(cells) - expected),
                       "ok": statuses["ok"], "non_response": statuses["non_response"]})
    return report


def export_flat(out_dir):
    dest = DATA_V2 / "results_repeated.csv"
    cols = ["original_name", "model_id", "model_served", "case_id", "order", "label_A", "label_B",
            "status", "selected_tool", "correct", "confidence", "trust_rationale", "differences_found",
            "harness_commit", "stimulus_sha256", "finished_at", "error"]
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for path in iter_result_files(out_dir):
            for line in path.read_text().splitlines():
                r = json.loads(line)
                d = r.get("decision") or {}
                w.writerow([r["original_name"], r["model_id"], r.get("model_served"), r["case_id"], r["order"],
                            r["labels"]["A"], r["labels"]["B"], r["status"], r.get("selected_tool"), r.get("correct"),
                            d.get("confidence"), d.get("trust_rationale"),
                            json.dumps(d.get("differences_found") or [], ensure_ascii=False),
                            r["harness_commit"], r["stimulus_sha256"], r["finished_at"], r.get("error", "")])
    return dest


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS_V2 / "main"
    cases, _ = load_v2_cases()
    report = check(out_dir, load_specs(), sorted(cases))
    dest = REPO / "analysis_repeated" / "completeness.csv"
    dest.parent.mkdir(exist_ok=True)
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    bad = [r for r in report if r["missing"] or r["duplicates"] or r["unexpected"]]
    for r in report:
        print(r)
    if bad:
        raise SystemExit(f"{len(bad)} models incomplete or duplicated")
    print("exported", export_flat(out_dir))


if __name__ == "__main__":
    main()
