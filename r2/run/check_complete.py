import csv
import json
import sys
from collections import Counter
from pathlib import Path

from r2.paths import DATA_V2, REPO, RUNS_V2
from r2.run.harness import (ORDERS, final_records, is_error_only, iter_result_files, load_specs,
                            load_v2_cases, read_records, results_path)


def _cell_counts(records):
    """Per cell: records that count (all but superseded error-only records), plus the number of
    superseded error-only records (an API error followed by a later record for the same cell)."""
    counted, superseded, seen_later = Counter(), 0, set()
    for r in reversed(records):
        key = (r["case_id"], r["order"])
        if key in seen_later and is_error_only(r):
            superseded += 1
        else:
            counted[key] += 1
        seen_later.add(key)
    return counted, superseded


class OrphanResults(RuntimeError):
    pass


EXPECTED_LABELS = {"TS": {"A": "T", "B": "S"}, "ST": {"A": "S", "B": "T"}}


def _invalid_ok_records(final):
    """Final ok records whose labels do not follow the order, whose `correct` does not follow from the
    labels and the selection, or whose stimulus_sha256 differs from the file's prevailing value."""
    ok = [r for r in final if r.get("status") == "ok"]
    shas = Counter(r.get("stimulus_sha256") for r in ok)
    prevailing = shas.most_common(1)[0][0] if shas else None
    bad = 0
    for r in ok:
        labels = r.get("labels")
        valid = (labels == EXPECTED_LABELS.get(r.get("order"))
                 and r.get("selected_tool") in ("A", "B")
                 and r.get("correct") == int(labels[r["selected_tool"]] == "T")
                 and r.get("stimulus_sha256") == prevailing)
        bad += not valid
    return bad


def _distinct(final, field):
    return ";".join(sorted({str(r[field]) for r in final if r.get(field) is not None}))


def orphan_files(out_dir, specs):
    """Result files in out_dir that belong to no current spec (e.g. a dropped or substituted model).
    Incident files (incidents.jsonl, *.incidents.jsonl) are evidence, not results, and are ignored."""
    expected = {results_path(s, out_dir).name for s in specs}
    return [p.name for p in iter_result_files(out_dir) if p.name not in expected]


def check(out_dir, specs, case_ids):
    """One completeness row per spec. Error cells may lack model_served (None or absent)."""
    if not specs:
        raise ValueError("no model specs to check (an empty spec list)")
    orphans = orphan_files(out_dir, specs)
    if orphans:
        raise OrphanResults(f"{out_dir} holds result files for no current spec: {orphans}")
    report = []
    expected = {(c, o) for c in case_ids for o in ORDERS}
    for spec in specs:
        path = results_path(spec, out_dir)
        records = read_records(path) if path.exists() else []
        cells, superseded = _cell_counts(records)
        final = list({(r["case_id"], r["order"]): r for r in records}.values())
        statuses = Counter(r["status"] for r in final)
        report.append({"model": spec["original_name"], "cells": len(cells),
                       "missing": len(expected - set(cells)),
                       "duplicates": sum(1 for v in cells.values() if v > 1),
                       "unexpected": len(set(cells) - expected),
                       "ok": statuses["ok"], "non_response": statuses["non_response"],
                       "errors": sum(1 for r in final if "error" in r),
                       "superseded_errors": superseded,
                       "invalid_records": _invalid_ok_records(final),
                       "harness_commit": _distinct(final, "harness_commit"),
                       "model_served": _distinct(final, "model_served"),
                       "system_fingerprint": _distinct(final, "system_fingerprint")})
    return report


def export_flat(out_dir, specs=None):
    dest = DATA_V2 / "results_repeated.csv"
    cols = ["original_name", "model_id", "model_served", "case_id", "order", "label_A", "label_B",
            "status", "selected_tool", "correct", "confidence", "trust_rationale", "differences_found",
            "harness_commit", "stimulus_sha256", "finished_at", "error"]
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for spec in (load_specs() if specs is None else specs):
            path = results_path(spec, out_dir)
            if not path.exists():
                continue
            for r in final_records(path):
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
    specs = load_specs()
    try:
        report = check(out_dir, specs, sorted(cases))
    except (OrphanResults, ValueError) as exc:
        raise SystemExit(f"gate failed: {exc}")
    dest = REPO / "analysis_repeated" / "completeness.csv"
    dest.parent.mkdir(exist_ok=True)
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    bad = [r for r in report if r["missing"] or r["duplicates"] or r["unexpected"] or r["invalid_records"]]
    for r in report:
        print(r)
    if bad:
        raise SystemExit(f"{len(bad)} models incomplete, duplicated or with invalid records")
    print("exported", export_flat(out_dir, specs))


if __name__ == "__main__":
    main()
