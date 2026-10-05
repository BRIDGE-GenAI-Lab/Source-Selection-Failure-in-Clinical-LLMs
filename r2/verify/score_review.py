"""Score two independent physician reviews, list disagreements, and apply the exclusion rule
fixed in the analysis plan."""
import json
import sys

import pandas as pd

from r2.paths import VERIFY_DIR

QUESTIONS = [
    ("q1_vignette_consistent", "Vignette internally consistent: no contradictory or invented findings, does not presuppose the answer (Y/N)"),
    ("q2_question_answerable", "Clinical question answerable from the vignette as written (Y/N)"),
    ("q3_authentic_concordant", "Authentic excerpt factually concordant with the cited source guideline (Y/N/P = partially)"),
    ("q4_authentic_relevant", "Authentic excerpt relevant to this clinical question (Y/N)"),
    ("q5_edit_visible", "Modification visible in the excerpt exactly as shown to the model (Y/N)"),
    ("q6_edit_matches_type", "Modification matches the declared type, with no other changes (Y/N)"),
    ("q7_edit_reduces_trust", "Modification makes the modified excerpt less trustworthy for this case (Y/N)"),
    ("q8_leakage", "Vignette or question reveals which excerpt is modified or gives away the answer (Y = leakage / N)"),
]
EXCLUDE_IF_NO = ("q3_authentic_concordant", "q4_authentic_relevant", "q5_edit_visible",
                 "q6_edit_matches_type", "q7_edit_reduces_trust")
EXCLUSION_RULE = ("Excluded from the primary analysis if physician consensus answered N to any of: "
                  + ", ".join(EXCLUDE_IF_NO))


def cohen_kappa(a, b):
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def exclusion_reasons(row):
    return [q for q in EXCLUDE_IF_NO if str(row.get(q, "")).strip().upper() == "N"]


def flag_reasons(row):
    flags = []
    for q in ("q1_vignette_consistent", "q2_question_answerable"):
        if str(row.get(q, "")).strip().upper() == "N":
            flags.append(q)
    if str(row.get("q3_authentic_concordant", "")).strip().upper() == "P":
        flags.append("q3_authentic_concordant")
    if str(row.get("q8_leakage", "")).strip().upper() == "Y":
        flags.append("q8_leakage")
    return flags


def _load(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    for q, _ in QUESTIONS:
        df[q] = df[q].str.strip().str.upper()
    return df.set_index("case_id").sort_index()


def score(path_a, path_b):
    a, b = _load(path_a), _load(path_b)
    assert list(a.index) == list(b.index), "the two forms cover different pairs"
    blanks = [(q, int((a[q] == "").sum() + (b[q] == "").sum())) for q, _ in QUESTIONS]
    assert all(n == 0 for _, n in blanks), f"unanswered items: {blanks}"
    rows, disagreements = [], []
    for q, _ in QUESTIONS:
        agree = float((a[q] == b[q]).mean())
        rows.append({"item": q, "agreement": agree, "kappa": cohen_kappa(list(a[q]), list(b[q]))})
        for cid in a.index[a[q] != b[q]]:
            disagreements.append({"case_id": cid, "item": q, "AG": a.loc[cid, q], "MO": b.loc[cid, q]})
    pd.DataFrame(rows).to_csv(VERIFY_DIR / "kappa.csv", index=False)
    pd.DataFrame(disagreements).to_csv(VERIFY_DIR / "adjudication_needed.csv", index=False)
    return rows, disagreements


def apply_consensus(path_a, path_b, consensus_path):
    """Consensus file lists only adjudicated items: case_id,item,consensus."""
    a = _load(path_a)
    b = _load(path_b)
    final = a.copy()
    cons = pd.read_csv(consensus_path, dtype=str)
    for _, r in cons.iterrows():
        final.loc[r.case_id, r["item"]] = r.consensus.strip().upper()
    for q, _ in QUESTIONS:
        mismatch = (a[q] != b[q])
        unresolved = [cid for cid in final.index[mismatch]
                      if not ((cons.case_id == cid) & (cons["item"] == q)).any()]
        assert not unresolved, f"{q}: disagreements without consensus: {unresolved[:10]}"
    excluded = {cid: exclusion_reasons(row) for cid, row in final.iterrows() if exclusion_reasons(row)}
    flags = {cid: flag_reasons(row) for cid, row in final.iterrows() if flag_reasons(row)}
    (VERIFY_DIR / "excluded_pairs.json").write_text(json.dumps(
        {"rule": EXCLUSION_RULE, "excluded": sorted(excluded), "reasons": excluded}, indent=2))
    (VERIFY_DIR / "flags.json").write_text(json.dumps(flags, indent=2))
    final.reset_index().to_csv(VERIFY_DIR / "final_review.csv", index=False)
    return excluded, flags


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(VERIFY_DIR / "review_AG.csv", VERIFY_DIR / "review_MO.csv")
    elif sys.argv[1] == "consensus":
        apply_consensus(VERIFY_DIR / "review_AG.csv", VERIFY_DIR / "review_MO.csv",
                        VERIFY_DIR / "consensus.csv")
