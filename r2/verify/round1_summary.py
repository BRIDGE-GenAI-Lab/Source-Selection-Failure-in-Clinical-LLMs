"""Descriptive summary of the round-1 physician review (A.G. and M.O., eight items), for all the reviewed pairs
and for the 333 pairs of the exclusion-only reanalysis (raw_status "visible" in audit/pair_audit.csv).

Reads the physicians' answers from verification/round1/review_AG.csv and review_MO.csv, as imported from their
returned workbooks (the workbooks are kept outside the repository; their SHA-256 is in round1_summary.json under
returned_workbooks). Per item it reports each physician's answer counts, the proportion of identical answers, the
unweighted Cohen kappa (three categories for Q3) and the number of pairs both physicians rated N (Q8: both rated
Y). A null kappa (blank in the CSV) means kappa is undefined: both physicians gave a single identical answer
category to every pair. It describes the review and excludes nothing. Writes verification/round1_summary.json and
.csv and the analyzed pairs' answers (verification/round1_analyzed_pairs.csv).
CLI: python -m r2.verify.round1_summary
"""
import csv
import json

import pandas as pd

from r2.io import load_original_cases
from r2.paths import AUDIT_DIR, VERIFY_DIR
from r2.verify.score_review import QUESTIONS, _load, allowed_answers, cohen_kappa

ROUND = "round1"
REVIEWER_NAMES = {"AG": "A.G.", "MO": "M.O."}
CONCERN = {q: ("Y" if q == "q8_leakage" else "N") for q, _ in QUESTIONS}
# what the models saw of each excerpt (r2.audit.pair_audit: excerpt_text and the four visible metadata fields),
# under the names the review forms use, with the case-file metadata key each one shows
SEEN = ("organization", "document_title", "version", "publication_year", "excerpt_text")
SEEN_METADATA = {"organization": "organization", "document_title": "guideline_title", "version": "version_label",
                 "publication_year": "publication_year"}
NOT_STATED = "not stated"  # how the forms show a missing metadata value
FC = "fabricated_citation"
KAPPA_UNDEFINED = ("undefined: both physicians gave one and the same answer category to every pair, so expected "
                   "agreement is 1 and kappa is 0/0")


def load_forms():
    """(A.G., M.O.): the round-1 forms indexed by case_id, answers checked (Y/N, Q3 also P; no blank)."""
    a, b = (_load(VERIFY_DIR / ROUND / f"review_{r}.csv") for r in REVIEWER_NAMES)
    if list(a.index) != list(b.index):
        raise ValueError("the two round-1 forms cover different pairs")
    return a, b


def analyzed_case_ids():
    """The 333 pairs of the reanalysis: authentic and modified excerpts differ in model-visible content."""
    audit = pd.read_csv(AUDIT_DIR / "pair_audit.csv")
    return sorted(audit.case_id[audit.raw_status == "visible"])


def item_summary(a, b, item):
    """One item: answer counts per physician, proportion identical, unweighted Cohen kappa, both rated concern."""
    a, b = list(a), list(b)
    out = {"item": item, "n_pairs": len(a)}
    for who, answers in (("AG", a), ("MO", b)):
        for v in allowed_answers(item):
            out[f"{who}_{v}"] = answers.count(v)
    concern = CONCERN[item]
    # kappa is undefined (0/0) when both physicians gave one and the same answer to every pair; cohen_kappa
    # returns 1.0 there, the summary records None
    single = len(set(a) | set(b)) == 1
    out.update({"agreement": sum(x == y for x, y in zip(a, b)) / len(a) if a else float("nan"),
                "kappa": None if single else cohen_kappa(a, b), "kappa_note": KAPPA_UNDEFINED if single else "",
                "concern_answer": concern, "both_concern": sum(x == y == concern for x, y in zip(a, b))})
    return out


def _seen(cell):
    tool = json.loads(cell)
    return {k: tool.get(k) for k in SEEN}


def _seen_original(tool):
    """The SEEN fields of a case-file tool, as the review forms show them."""
    meta = tool.get("metadata") or {}
    out = {k: NOT_STATED if meta.get(key) is None else meta.get(key) for k, key in SEEN_METADATA.items()}
    out["excerpt_text"] = tool.get("excerpt_text", "")
    return out


def _packets_vs_original(form, ids):
    """For the analyzed pairs, whether each excerpt as shown in round 1 equals, in the fields the models saw,
    the excerpt of the original as-run case file."""
    cases, _ = load_original_cases()
    same = {"authentic": 0, "modified": 0}
    differ = {}
    for cid in ids:
        for side, col, key in (("authentic", "authentic_as_shown_to_model", "T"),
                               ("modified", "modified_as_shown_to_model", "S")):
            if _seen(form.loc[cid, col]) == _seen_original(cases[cid]["tools"][key]):
                same[side] += 1
            else:
                differ.setdefault(side, {}).setdefault(form.loc[cid, "sham_type"], []).append(cid)
    return {"fields_compared": list(SEEN), "n_pairs": len(ids),
            "authentic_same_as_original_run": same["authentic"], "modified_same_as_original_run": same["modified"],
            "differing_by_sham_type": {side: {t: len(v) for t, v in sorted(d.items())} for side, d in differ.items()},
            "differing_case_ids": {side: sorted(c for v in d.values() for c in v) for side, d in differ.items()}}


def _fabricated_citation(form, ids):
    fc = form[form.sham_type == FC]
    mod = fc.modified_as_shown_to_model.map(lambda c: json.loads(c)["excerpt_text"])
    out = {"n_rows_all_forms": int(len(fc)), "n_rows_analyzed_pairs": int(fc.index.isin(ids).sum()),
           "rows_with_neutral_doi": int(mod.str.contains("DOI:10.9999/cg.", regex=False).sum()),
           "rows_with_fake_token": int(mod.str.contains("fake", case=False, regex=False).sum())}
    out["note"] = (f"The physicians saw the {out['n_rows_all_forms']} fabricated_citation pairs with the neutral DOI "
                   "rendering (DOI:10.9999/cg.<year>.NNNN), not the DOI:10.9999/fake.NNNN that the models saw in "
                   "the original runs; their answers on these pairs describe the neutral rendering.")
    return out


def summarize(a, b):
    """The summary of both subsets (all reviewed pairs; the analyzed pairs) from the two forms."""
    ids = analyzed_case_ids()
    missing = sorted(set(ids) - set(a.index))
    if missing:
        raise ValueError(f"analyzed pairs missing from the round-1 forms: {missing[:10]}")
    subsets = {}
    for name, rows in (("all_forms", list(a.index)), ("analyzed_pairs", ids)):
        subsets[name] = {"n_pairs": len(rows),
                         "items": [item_summary(a.loc[rows, q], b.loc[rows, q], q) for q, _ in QUESTIONS]}
    subsets["all_forms"]["definition"] = "every pair of the round-1 forms"
    subsets["analyzed_pairs"]["definition"] = ("the pairs whose authentic and modified excerpts differ in "
                                               "model-visible content (audit/pair_audit.csv raw_status visible)")
    return {"round": ROUND, "reviewers": list(REVIEWER_NAMES.values()),
            "forms": [f"verification/{ROUND}/review_{r}.csv" for r in REVIEWER_NAMES],
            "questions": dict(QUESTIONS), "concern_answer": CONCERN,
            "kappa": ("unweighted Cohen kappa (r2.verify.score_review.cohen_kappa); Q3 over three categories Y/N/P. "
                      "A null kappa (a blank cell in round1_summary.csv) means kappa is undefined because both "
                      "physicians gave a single identical answer category to every pair; kappa_note then says so"),
            "excludes_nothing": True, "use": "descriptive; no pair is excluded on these answers",
            "subsets": subsets, "fabricated_citation": _fabricated_citation(a, ids),
            "packets_vs_original_run": _packets_vs_original(a, ids)}


def analyzed_answers(a, b):
    """The analyzed pairs with both physicians' Q1-Q8 answers (no notes), one row per pair, by case_id."""
    ids = analyzed_case_ids()
    out = pd.DataFrame({"case_id": ids, "sham_type": a.loc[ids, "sham_type"].to_numpy()})
    for who, form in (("AG", a), ("MO", b)):
        for q, _ in QUESTIONS:
            out[f"{who}_{q}"] = form.loc[ids, q].to_numpy()
    return out


def _csv_rows(summary):
    keys = ["subset", "n_pairs", "item", "AG_Y", "AG_N", "AG_P", "MO_Y", "MO_N", "MO_P", "agreement", "kappa",
            "kappa_note", "concern_answer", "both_concern"]
    rows = [{"subset": name, **it} for name, s in summary["subsets"].items() for it in s["items"]]
    return keys, [{k: r.get(k, "") for k in keys} for r in rows]


def run():
    summary = summarize(*load_forms())
    prev = VERIFY_DIR / "round1_summary.json"  # keep the provenance of the import of the returned workbooks
    if prev.exists():
        summary["returned_workbooks"] = json.loads(prev.read_text(encoding="utf-8")).get("returned_workbooks")
    (VERIFY_DIR / "round1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    keys, rows = _csv_rows(summary)
    with open(VERIFY_DIR / "round1_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    a, b = load_forms()
    analyzed_answers(a, b).to_csv(VERIFY_DIR / "round1_analyzed_pairs.csv", index=False)
    return summary


if __name__ == "__main__":
    s = run()
    for name, sub in s["subsets"].items():
        print(name, sub["n_pairs"])
        for it in sub["items"]:
            kappa = "undefined" if it["kappa"] is None else f"{it['kappa']:.3f}"
            print(f"  {it['item']:26s} agree {it['agreement']:.3f} kappa {kappa} "
                  f"both {it['concern_answer']} {it['both_concern']}")
