"""Score two independent physician reviews, list disagreements, and apply the exclusion rule
fixed in the analysis plan."""
import hashlib
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


def allowed_answers(item):
    return ("Y", "N", "P") if item == "q3_authentic_concordant" else ("Y", "N")


def _load(path):
    """Read one physician form. Every question cell must be Y or N (Q3 also P), after stripping and
    upper-casing; only the notes column may be blank."""
    df = pd.read_csv(path, dtype=str).fillna("")
    problems = []
    for q, _ in QUESTIONS:
        if q not in df.columns:
            raise ValueError(f"{path}: missing answer column {q}")
        df[q] = df[q].str.strip().str.upper()
        blank = df.case_id[df[q] == ""].tolist()
        if blank:
            problems.append(f"{q}: blank answers for {blank[:10]}")
        bad = df[(df[q] != "") & ~df[q].isin(allowed_answers(q))]
        if len(bad):
            problems.append(f"{q}: invalid answers {sorted(set(bad[q]))} for {bad.case_id.tolist()[:10]} "
                            f"(allowed {allowed_answers(q)})")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return df.set_index("case_id").sort_index()


def score(path_a, path_b):
    a, b = _load(path_a), _load(path_b)
    if list(a.index) != list(b.index):
        raise ValueError("the two forms cover different pairs")
    blanks = [(q, int((a[q] == "").sum() + (b[q] == "").sum())) for q, _ in QUESTIONS]
    if any(n for _, n in blanks):
        raise ValueError(f"unanswered items: {blanks}")
    rows, disagreements = [], []
    for q, _ in QUESTIONS:
        agree = float((a[q] == b[q]).mean())
        rows.append({"item": q, "agreement": agree, "kappa": cohen_kappa(list(a[q]), list(b[q]))})
        for cid in a.index[a[q] != b[q]]:
            disagreements.append({"case_id": cid, "item": q, "AG": a.loc[cid, q], "MO": b.loc[cid, q]})
    pd.DataFrame(rows).to_csv(VERIFY_DIR / "kappa.csv", index=False)
    pd.DataFrame(disagreements).to_csv(VERIFY_DIR / "adjudication_needed.csv", index=False)
    return rows, disagreements


def _check_packets(forms):
    """The forms must have been generated from the current data/v2/pairs_v2.json: the manifest written
    with the forms carries the file's SHA-256, and every form row's packet_sha256 must equal the hash
    recomputed from the current file. Returns the current file's SHA-256."""
    from r2.verify import build_packets  # local import: build_packets imports QUESTIONS from here
    raw = build_packets.pairs_path().read_bytes()
    current = hashlib.sha256(raw).hexdigest()
    manifest_path = VERIFY_DIR / build_packets.MANIFEST
    if not manifest_path.exists():
        raise ValueError(f"{manifest_path} is missing; the forms cannot be bound to pairs_v2.json")
    recorded = json.loads(manifest_path.read_text()).get("pairs_v2_sha256")
    if recorded != current:
        raise ValueError(f"pairs_v2_sha256 in {manifest_path.name} ({recorded}) differs from the current "
                         f"data/v2/pairs_v2.json ({current}); the reviewed packets are not the current stimuli")
    packets = {r["case_id"]: r["packet_sha256"] for r in build_packets.packet_rows(json.loads(raw)["cases"])}
    for name, form in forms.items():
        if "packet_sha256" not in form.columns:
            raise ValueError(f"{name}: no packet_sha256 column")
        if set(form.index) != set(packets):
            raise ValueError(f"{name}: form rows do not match the current packets "
                             f"(missing {sorted(set(packets) - set(form.index))[:10]}, "
                             f"extra {sorted(set(form.index) - set(packets))[:10]})")
        bad = [cid for cid in form.index if form.loc[cid, "packet_sha256"] != packets[cid]]
        if bad:
            raise ValueError(f"{name}: packet_sha256 differs from the current render for {bad[:10]}")
    return current


def apply_consensus(path_a, path_b, consensus_path):
    """Consensus file lists only adjudicated items: case_id,item,consensus."""
    a = _load(path_a)
    b = _load(path_b)
    if list(a.index) != list(b.index):
        raise ValueError("the two forms cover different case_ids")
    pairs_sha = _check_packets({str(path_a): a, str(path_b): b})
    qkeys = [q for q, _ in QUESTIONS]
    blanks = {q: int((a[q] == "").sum() + (b[q] == "").sum()) for q in qkeys}
    if any(blanks.values()):
        raise ValueError(f"blank answers in the forms: { {q: n for q, n in blanks.items() if n} }")
    cons = pd.read_csv(consensus_path, dtype=str).fillna("")
    for _, r in cons.iterrows():
        cid, item, val = r["case_id"], r["item"], str(r["consensus"]).strip().upper()
        if cid not in a.index:
            raise ValueError(f"consensus row for unknown case_id {cid}")
        if item not in qkeys:
            raise ValueError(f"consensus row for unknown item {item} ({cid})")
        allowed = allowed_answers(item)
        if val not in allowed:
            raise ValueError(f"consensus value {val!r} not allowed for {item} ({cid}); allowed {allowed}")
        if a.loc[cid, item] == b.loc[cid, item]:
            raise ValueError(f"consensus row for {cid} {item} but the reviewers already agree; "
                             "it would override agreement")
    final = a.copy()
    for _, r in cons.iterrows():
        final.loc[r["case_id"], r["item"]] = str(r["consensus"]).strip().upper()
    for q, _ in QUESTIONS:
        mismatch = (a[q] != b[q])
        unresolved = [cid for cid in final.index[mismatch]
                      if not ((cons.case_id == cid) & (cons["item"] == q)).any()]
        if unresolved:
            raise ValueError(f"{q}: disagreements without consensus: {unresolved[:10]}")
    excluded = {cid: exclusion_reasons(row) for cid, row in final.iterrows() if exclusion_reasons(row)}
    flags = {cid: flag_reasons(row) for cid, row in final.iterrows() if flag_reasons(row)}
    (VERIFY_DIR / "excluded_pairs.json").write_text(json.dumps(
        {"rule": EXCLUSION_RULE, "excluded": sorted(excluded), "reasons": excluded,
         "pairs_v2_sha256": pairs_sha}, indent=2))
    (VERIFY_DIR / "flags.json").write_text(json.dumps(flags, indent=2))
    final.reset_index().to_csv(VERIFY_DIR / "final_review.csv", index=False)
    return excluded, flags


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(VERIFY_DIR / "review_AG.csv", VERIFY_DIR / "review_MO.csv")
    elif sys.argv[1] == "consensus":
        apply_consensus(VERIFY_DIR / "review_AG.csv", VERIFY_DIR / "review_MO.csv",
                        VERIFY_DIR / "consensus.csv")
