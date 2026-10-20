"""The physician review items as worded on the review form, answer validation and Cohen's kappa
(used by r2.verify.round1_summary)."""
import pandas as pd

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


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


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
