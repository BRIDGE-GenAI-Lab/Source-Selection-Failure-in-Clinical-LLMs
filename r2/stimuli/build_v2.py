"""Build the corrected stimulus set: every pair carries a modification that is visible in the
rendered model input. Pairs that were already edited keep their original modified record."""
import argparse
import copy
import csv
import json

import pandas as pd

from r2.io import load_original_cases
from r2.paths import AUDIT_DIR, DATA_V2, ORIGINAL_CASES_SHA256
from r2.stimuli.render import visible_content
from r2.stimuli.transforms import TransformError, apply_manual_edit, propose_inversion, typosquat


class UnresolvedEdits(RuntimeError):
    def __init__(self, case_ids):
        super().__init__(f"{len(case_ids)} pairs still lack a visible edit: {case_ids}")
        self.case_ids = case_ids


def edit_record(t, s, source):
    rt, rs = visible_content(t), visible_content(s)
    changes = [{"field": k, "before": rt[k], "after": rs[k]} for k in rt if rt[k] != rs[k]]
    return {"edit_source": source, "fields_changed": [c["field"] for c in changes], "changes": changes}


def build_v2(manual_edits=None, strict=True):
    manual_edits = manual_edits or {}
    cases, _ = load_original_cases()
    audit = pd.read_csv(AUDIT_DIR / "pair_audit.csv").set_index("case_id")
    out, unresolved = [], []
    for cid in sorted(cases):
        case = copy.deepcopy(cases[cid])
        t, s = case["tools"]["T"], case["tools"]["S"]
        sham_type = case["eval"]["sham_trap_type"]
        status = audit.loc[cid, "raw_status"]
        source = "original_transform"
        if cid in manual_edits:
            if status != "no_edit" and not manual_edits[cid].get("override"):
                raise ValueError(f"{cid}: manual_edits.json has an entry for a pair that already carries an "
                                 f"original edit (raw_status {status!r}); add \"override\": true to replace it")
            s = apply_manual_edit(s, manual_edits[cid])
            source = "physician_manual"
        elif status == "no_edit" and sham_type == "authority_mimicry":
            s = copy.deepcopy(s)
            try:
                s["metadata"]["organization"] = typosquat(t["metadata"]["organization"])
            except TransformError:
                unresolved.append(cid)
                continue
            source = "rule_typosquat"
        elif status == "no_edit" and sham_type == "subtle_inversion":
            proposal = propose_inversion(t["excerpt_text"])
            if proposal is None:
                unresolved.append(cid)
                continue
            s = copy.deepcopy(s)
            s["excerpt_text"] = proposal[0]
            source = "rule_inversion_extended"
        elif status == "no_edit":
            unresolved.append(cid)
            continue
        case["tools"]["S"] = s
        if visible_content(t) == visible_content(s):
            raise AssertionError(f"{cid}: rendered authentic and modified excerpts are identical")
        case["edit_record"] = edit_record(t, s, source)
        out.append(case)
    if unresolved and strict:
        raise UnresolvedEdits(unresolved)
    return out, unresolved


QUEUE_COLUMNS = ["case_id", "sham_type", "authentic_organization", "clinical_question", "authentic_excerpt_text"]


def write_outputs(cases, unresolved, originals=None):
    """Write pairs_v2.json and keep manual_edit_queue.csv in step: rewritten while pairs are
    unresolved, removed (never left stale) once the queue is empty."""
    DATA_V2.mkdir(parents=True, exist_ok=True)
    queue_path = DATA_V2 / "manual_edit_queue.csv"
    if unresolved:
        if originals is None:
            originals, _ = load_original_cases()
        with open(queue_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(QUEUE_COLUMNS)
            for cid in unresolved:
                c = originals[cid]
                w.writerow([cid, c["eval"]["sham_trap_type"], c["tools"]["T"]["metadata"]["organization"],
                            c.get("clinical_question"), c["tools"]["T"]["excerpt_text"]])
        print(f"{len(unresolved)} pairs need a physician-authored edit: data/v2/manual_edit_queue.csv")
    elif queue_path.exists():
        queue_path.unlink()
        print("no unresolved pairs: removed the stale data/v2/manual_edit_queue.csv")
    payload = {"derived_from_sha256": ORIGINAL_CASES_SHA256, "n_pairs": len(cases),
               "unresolved": unresolved, "cases": cases}
    (DATA_V2 / "pairs_v2.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(cases)} pairs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-unresolved", action="store_true")
    args = parser.parse_args()
    manual_path = DATA_V2 / "manual_edits.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else {}
    cases, unresolved = build_v2(manual, strict=not args.allow_unresolved)
    write_outputs(cases, unresolved)


if __name__ == "__main__":
    main()
