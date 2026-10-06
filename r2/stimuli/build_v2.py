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
from r2.stimuli.transforms import (FAKE_DOI, TransformError, apply_manual_edit, neutralize_citation_doi,
                                   propose_inversion, self_label_tokens, typosquat)


class UnresolvedEdits(RuntimeError):
    def __init__(self, case_ids):
        super().__init__(f"{len(case_ids)} pairs still lack a visible edit: {case_ids}")
        self.case_ids = case_ids


def edit_record(t, s, source):
    rt, rs = visible_content(t), visible_content(s)
    changes = [{"field": k, "before": rt[k], "after": rs[k]} for k in rt if rt[k] != rs[k]]
    return {"edit_source": source, "fields_changed": [c["field"] for c in changes], "changes": changes}


def self_labels_in(tool):
    """Self-labelling tokens in the tool as shown to the model (every rendered field, read as plain text,
    not as JSON: an escaped newline would hide a word boundary)."""
    parts = []
    for value in visible_content(tool).values():
        parts += [str(x) for x in value] if isinstance(value, list) else [str(value)]
    return self_label_tokens("\n".join(parts))


def neutralize_fabricated_citation(cid, s):
    """Final fix F1: the original fabricated-citation edit ends with DOI:10.9999/fake.NNNN, and the word
    "fake" tells the model which excerpt was modified. Replace the suffix with a neutral journal-style one
    (r2.stimuli.transforms.neutral_doi_suffix); everything else in the modified tool is kept."""
    s = copy.deepcopy(s)
    s["excerpt_text"], before, after = neutralize_citation_doi(s["excerpt_text"], cid)
    return s, {"rule": "neutral_doi_suffix", "field": "excerpt_text", "before": before, "after": after}


def before_self_label_fix(case):
    """The case as the build wrote it before F1: the neutral DOI put back and `self_label_fix` dropped.
    Used to prove that F1 changed nothing else."""
    fix = case["edit_record"]["self_label_fix"]
    text = json.dumps(case, ensure_ascii=False)
    if text.count(fix["after"]) != 3:  # S excerpt, the edit_record change, self_label_fix itself
        raise ValueError(f"{case['case_id']}: {fix['after']!r} occurs {text.count(fix['after'])} times, expected 3")
    out = json.loads(text.replace(fix["after"], fix["before"]))
    del out["edit_record"]["self_label_fix"]
    return out


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
        fix = None
        if sham_type == "fabricated_citation" and FAKE_DOI.search(s["excerpt_text"]):
            s, fix = neutralize_fabricated_citation(cid, s)
        case["tools"]["S"] = s
        if visible_content(t) == visible_content(s):
            raise AssertionError(f"{cid}: rendered authentic and modified excerpts are identical")
        labels = self_labels_in(s)
        if labels:
            raise AssertionError(f"{cid}: the modified excerpt carries self-labelling token(s) {labels}")
        case["edit_record"] = edit_record(t, s, source)
        if fix:
            case["edit_record"]["self_label_fix"] = fix
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
