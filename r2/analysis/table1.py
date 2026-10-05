"""Table 1: representative model responses in the repeated experiment (plan Task 12 Step 3).

For each modification type, one failure (the model selected the modified excerpt) and one success, drawn
with random.Random(seed) from the final ok records of the main run on verified pairs that are neither
excluded nor flagged. Every path defaults to r2.paths at call time, so a sandbox can redirect them.
CLI: python -m r2.analysis.table1 [out_dir]
"""
import csv
import json
import random
import sys
from pathlib import Path

from r2 import paths
from r2.analysis.repeated import _side, load_verification
from r2.run import harness

SEED = 20261005
MOD_CHARS = 120
QUOTE_CHARS = 160
CONTEXT_WORDS = 4
ELLIPSIS = "…"
KEYS = ("sham_type", "outcome", "case_id", "model", "order", "patient", "modification", "rationale_quote")
SEX = {"F": "woman", "M": "man"}

TABLE1_NOTE = (
    "Selection rule: for each modification type, one failure (the model selected the modified excerpt) and one "
    f"success (it selected the authentic excerpt), drawn with a fixed seed (random.Random({SEED})) from the valid "
    "responses of the repeated experiment on physician-verified pairs that were neither excluded nor flagged "
    "(verification/excluded_pairs.json, verification/flags.json). Modification: the edit as shown to the model, "
    "by field (Removed, Added or Changed text), each text cut at 120 characters. Rationale: the first 160 "
    "characters of the model's stated rationale, verbatim. Abbreviations: TS, authentic excerpt shown first "
    "(as A) and modified excerpt second; ST, modified excerpt shown first (as A)."
)


def _clip(text, n=MOD_CHARS):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n].rstrip() + ELLIPSIS


def quote(text, n=QUOTE_CHARS):
    """The first n characters of the rationale, verbatim, with an ellipsis when cut."""
    text = str(text).strip()
    return text if len(text) <= n else text[:n] + ELLIPSIS


def _focus(before, after):
    """Drop the shared opening words of a changed line (keeping a few for context) so the edit is visible."""
    a, b = str(before).split(), str(after).split()
    common = 0
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    cut = common - CONTEXT_WORDS
    if cut <= 0:
        return " ".join(a), " ".join(b)
    return ELLIPSIS + " ".join(a[cut:]), ELLIPSIS + " ".join(b[cut:])


def modification(edit_record):
    """The edit, by field: "Removed '…'", "Added '…'" or "Changed '…' → '…'"."""
    parts = []
    for ch in edit_record["changes"]:
        before = [x for x in _side(ch["before"]) if x.strip()]
        after = [x for x in _side(ch["after"]) if x.strip()]
        removed = [x for x in before if x not in after]
        added = [x for x in after if x not in before]
        field = ch["field"]
        if len(removed) == 1 and len(added) == 1:
            old, new = _focus(removed[0], added[0])
            parts.append(f"{field}: Changed '{_clip(old)}' → '{_clip(new)}'")
            continue
        actions = [f"{verb} {_and([_quoted(x) for x in items])}"
                   for verb, items in (("Removed", removed), ("Added", added)) if items]
        if actions:
            parts.append(f"{field}: {', '.join(actions)}")
    return "; ".join(parts)


def _quoted(text):
    return "'" + _clip(text) + "'"


def _and(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def patient(case):
    demo = case.get("demographics") or {}
    age = demo.get("age")
    who = f"{age}-year-old {SEX.get(demo.get('sex'), demo.get('sex') or 'patient')}" if age is not None \
        else f"Age not recorded, {SEX.get(demo.get('sex'), demo.get('sex') or 'patient')}"
    return f"{who}: {case.get('chief_complaint') or 'chief complaint not recorded'}"


def table1_rows(seed=SEED, *, pairs_path=None, verify_dir=None, runs_dir=None, specs=None):
    pairs_path = paths.DATA_V2 / "pairs_v2.json" if pairs_path is None else Path(pairs_path)
    verify_dir = paths.VERIFY_DIR if verify_dir is None else Path(verify_dir)
    runs_dir = paths.RUNS_V2 / "main" if runs_dir is None else Path(runs_dir)
    specs = harness.load_specs() if specs is None else specs
    pairs = {c["case_id"]: c for c in json.loads(pairs_path.read_text())["cases"]}
    excluded, flagged, _ = load_verification(verify_dir, set(pairs))
    pools = {}
    for spec in specs:
        path = harness.results_path(spec, runs_dir)
        if not path.exists():
            continue
        for r in harness.final_records(path):
            rationale = (r.get("decision") or {}).get("trust_rationale")
            if r.get("status") != "ok" or r["case_id"] in excluded | flagged or not isinstance(rationale, str) \
                    or not rationale.strip():
                continue
            sham_type = pairs[r["case_id"]]["eval"]["sham_trap_type"]
            outcome = "Success" if r["correct"] == 1 else "Failure"
            pools.setdefault((sham_type, outcome), []).append((r["original_name"], r["case_id"], r["order"], rationale))
    rng = random.Random(seed)
    rows = []
    for sham_type in sorted({t for t, _ in pools}):
        for outcome in ("Failure", "Success"):
            pool = sorted(pools.get((sham_type, outcome), []))
            if not pool:
                continue
            model, cid, order, rationale = rng.choice(pool)
            case = pairs[cid]
            rows.append(dict(zip(KEYS, (sham_type, outcome, cid, model, order, patient(case),
                                        modification(case["edit_record"]), quote(rationale)))))
    return rows


def write_table1(out_dir, **kwargs):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = table1_rows(**kwargs)
    with open(out_dir / "Table1.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(KEYS))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "Table1_note.txt").write_text(TABLE1_NOTE + "\n", encoding="utf-8")
    return rows


def main():
    from r2.analysis.supp_tables import OUT_DIR
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR
    rows = write_table1(out)
    print(f"wrote {out / 'Table1.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
