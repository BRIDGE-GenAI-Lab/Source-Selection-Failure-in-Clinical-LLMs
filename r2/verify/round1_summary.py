"""Descriptive summary of the round-1 physician review (A.G. and M.O., eight items), for all the reviewed pairs
and for the 333 pairs of the exclusion-only reanalysis (raw_status "visible" in audit/pair_audit.csv).

Imports the returned workbooks (read only; never modified) into verification/round1/review_AG.csv and
review_MO.csv with r2.verify.build_packets.import_review_xlsx, which checks them against the round-1 forms as
sent; the round-1 .xlsx files and packets_manifest.json are left untouched. Per item it reports each
physician's answer counts, the proportion of identical answers, the unweighted Cohen kappa (three categories
for Q3) and the number of pairs both physicians rated N (Q8: both rated Y). A null kappa (blank in the CSV) means
kappa is undefined: both physicians gave a single identical answer category to every pair. It describes the
review and excludes nothing. Writes verification/round1_summary.json and .csv, the analyzed pairs' answers
(verification/round1_analyzed_pairs.csv) and verification/round1/README_STATUS.md.
CLI: python -m r2.verify.round1_summary [returned_dir]
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from r2.io import load_original_cases
from r2.paths import AUDIT_DIR, VERIFY_DIR
from r2.stimuli.render import render_tool
from r2.verify import build_packets as bp
from r2.verify.evaluable import CONSENSUS_CSV
from r2.verify.score_review import QUESTIONS, _load, allowed_answers, cohen_kappa

ROUND = "round1"
RETURNED = Path.home() / "npj_r2" / "returned"
WORKBOOKS = {"AG": "AG_r1.xlsx", "MO": "MO_r1.xlsx"}
CONCERN = {q: ("Y" if q == "q8_leakage" else "N") for q, _ in QUESTIONS}
# what the models saw of each excerpt (r2.audit.pair_audit: excerpt_text and the four visible metadata fields)
SEEN = ("organization", "document_title", "version", "publication_year", "excerpt_text")
FC = "fabricated_citation"
KAPPA_UNDEFINED = ("undefined: both physicians gave one and the same answer category to every pair, so expected "
                   "agreement is 1 and kappa is 0/0")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_returned(returned=None):
    """Import both returned workbooks into the round-1 form CSVs. M.O.'s returned copy has its Instructions!A1
    header cleared, so each reviewer is named and, without a header, identified by the row order of the form
    as sent. The workbooks, the round-1 .xlsx forms and the manifest are checked byte for byte before and after."""
    returned = RETURNED if returned is None else Path(returned)
    d = VERIFY_DIR / ROUND
    keep = [returned / name for name in WORKBOOKS.values()] + \
        [d / bp.MANIFEST] + [d / f"review_{r}.xlsx" for r in WORKBOOKS]
    before = {p: _sha256(p) for p in keep}
    sources = {}
    for reviewer, name in WORKBOOKS.items():
        bp.import_review_xlsx(returned / name, d / f"review_{reviewer}.csv", ROUND, reviewer=reviewer)
        sources[reviewer] = {"workbook": name, "sha256": before[returned / name]}
    changed = [str(p) for p in keep if _sha256(p) != before[p]]
    if changed:
        raise RuntimeError(f"files that must stay untouched changed: {changed}")
    return sources


def load_forms():
    """(A.G., M.O.): the round-1 forms indexed by case_id, answers checked (Y/N, Q3 also P; no blank)."""
    a, b = (_load(VERIFY_DIR / ROUND / f"review_{r}.csv") for r in WORKBOOKS)
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


def _packets_vs_original(form, ids):
    """For the analyzed pairs, whether each excerpt as shown in round 1 equals, in the fields the models saw,
    the excerpt of the original as-run case file."""
    cases, _ = load_original_cases()
    same = {"authentic": 0, "modified": 0}
    differ = {}
    for cid in ids:
        for side, col, key in (("authentic", "authentic_as_shown_to_model", "T"),
                               ("modified", "modified_as_shown_to_model", "S")):
            orig = {k: v for k, v in render_tool(cases[cid]["tools"][key], side).items() if k in SEEN}
            if _seen(form.loc[cid, col]) == orig:
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
    return {"round": ROUND, "reviewers": [bp.REVIEWER_NAMES[r] for r in WORKBOOKS],
            "forms": [f"verification/{ROUND}/review_{r}.csv" for r in WORKBOOKS],
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


def status_text(summary):
    """verification/round1/README_STATUS.md: what the round-1 forms show and how the R2 reanalysis uses them."""
    sent = "The text in the *_as_shown_to_model columns is exactly what the evaluated models receive."
    readme = (VERIFY_DIR / ROUND / "README_REVIEWERS.md").read_text(encoding="utf-8")
    if sent not in readme:
        raise ValueError("README_REVIEWERS.md no longer holds the sentence README_STATUS.md quotes")
    n_all, n_ana = summary["subsets"]["all_forms"]["n_pairs"], summary["subsets"]["analyzed_pairs"]["n_pairs"]
    fc, pv = summary["fabricated_citation"], summary["packets_vs_original_run"]
    return (
        "# Round-1 forms: status\n\n"
        f"The round-1 forms (`review_AG`, `review_MO`, `packets.html`) show each of the {n_all} pairs as it was prepared "
        "for a repeated experiment that was not run: every content field of both excerpts (including `safety_guardrails`, "
        "`applicability` and `exclusions_or_cautions`, which no original runner serialized) and the full vignette. "
        "That is more than any of the five original runner configurations presented to the models. The sentence in "
        f"`README_REVIEWERS.md`, \"{sent}\", referred to that unrun experiment, not to the original evaluations.\n\n"
        f"The R2 reanalysis uses the physicians' answers for the {n_ana} analyzed pairs (the pairs whose authentic and "
        "modified excerpts differ in content visible to the models in the original runs), descriptively: "
        "`verification/round1_summary.json`, `verification/round1_summary.csv` and "
        "`verification/round1_analyzed_pairs.csv`. No pair is excluded on these answers. For these pairs, the excerpt "
        "text and the visible metadata shown in the forms equal the original as-run excerpts "
        f"({pv['authentic_same_as_original_run']} of {pv['n_pairs']} authentic, "
        f"{pv['modified_same_as_original_run']} of {pv['n_pairs']} modified); the "
        f"{fc['n_rows_analyzed_pairs']} fabricated_citation modified excerpts show the neutral DOI suffix "
        "10.9999/cg.<year>.NNNN instead of the 10.9999/fake.NNNN that the models saw.\n" + _consensus_text())


def _consensus_text():
    """The Q3 consensus paragraph of README_STATUS.md, read from the imported consensus; empty before the import."""
    if not CONSENSUS_CSV.exists():
        return ""
    c = pd.read_csv(CONSENSUS_CSV, dtype=str)
    n = c["consensus_q3_authentic_concordant"].value_counts()
    dates = ", ".join(sorted(c["adjudicated_on"].unique()))
    return (f"\nPost hoc, A.G. and M.O. jointly adjudicated Q3 on {dates} for the {len(c)} clinically evaluable pairs "
            "(both answered Y on Q2, Q4 and Q7) on which their Q3 answers differed: "
            "`verification/round1/q3_consensus_form.xlsx` as sent and `verification/round1/q3_consensus_returned.xlsx` "
            "as returned, imported into `verification/round1_q3_consensus.csv` "
            f"(Y {n.get('Y', 0)}, P {n.get('P', 0)}, N {n.get('N', 0)}). Their original answers stay as recorded. The "
            "consensus is used only for the clinically evaluable sensitivity analysis "
            "(`analysis_original/clinically_evaluable.json`).\n")


def _csv_rows(summary):
    keys = ["subset", "n_pairs", "item", "AG_Y", "AG_N", "AG_P", "MO_Y", "MO_N", "MO_P", "agreement", "kappa",
            "kappa_note", "concern_answer", "both_concern"]
    rows = [{"subset": name, **it} for name, s in summary["subsets"].items() for it in s["items"]]
    return keys, [{k: r.get(k, "") for k in keys} for r in rows]


def run(returned=None, do_import=True):
    sources = import_returned(returned) if do_import else None
    summary = summarize(*load_forms())
    if sources is not None:
        summary["returned_workbooks"] = sources
    else:  # keep the provenance of the last import
        prev = VERIFY_DIR / "round1_summary.json"
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
    (VERIFY_DIR / ROUND / "README_STATUS.md").write_text(status_text(summary), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = run(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    for name, sub in s["subsets"].items():
        print(name, sub["n_pairs"])
        for it in sub["items"]:
            kappa = "undefined" if it["kappa"] is None else f"{it['kappa']:.3f}"
            print(f"  {it['item']:26s} agree {it['agreement']:.3f} kappa {kappa} "
                  f"both {it['concern_answer']} {it['both_concern']}")
