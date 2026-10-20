import pandas as pd
import pytest

from r2.verify import score_review as sr
from r2.verify.score_review import cohen_kappa

QKEYS = [q for q, _ in sr.QUESTIONS]


def test_kappa_known_value():
    assert abs(cohen_kappa(["Y", "Y", "N", "N"], ["Y", "N", "N", "N"]) - 0.5) < 1e-9
    assert cohen_kappa(["Y", "Y"], ["Y", "Y"]) == 1.0


def test_kappa_of_empty_input_is_nan():
    import math
    assert math.isnan(cohen_kappa([], []))


def _form(path, overrides=None, cases=("C1", "C2", "C3")):
    rows = []
    for cid in cases:
        row = {"case_id": cid, **{q: "Y" for q in QKEYS}, "q8_leakage": "N", "notes": ""}
        row.update((overrides or {}).get(cid, {}))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---- I5: answer validation ----
@pytest.mark.parametrize("item,value", [("q5_edit_visible", "No"), ("q4_authentic_relevant", "n/a"),
                                        ("q6_edit_matches_type", "P"), ("q8_leakage", "maybe"),
                                        ("q1_vignette_consistent", ""), ("q3_authentic_concordant", "partial")])
def test_load_rejects_invalid_answers(tmp_path, item, value):
    bad = _form(tmp_path / "bad.csv", {"C2": {item: value}})
    with pytest.raises(ValueError, match=item):
        sr._load(bad)


def test_load_normalizes_case_and_allows_blank_notes(tmp_path):
    ok = _form(tmp_path / "ok.csv", {"C1": {"q3_authentic_concordant": " p ", "q2_question_answerable": "y", "notes": ""}})
    df = sr._load(ok)
    assert df.loc["C1", "q3_authentic_concordant"] == "P" and df.loc["C1", "q2_question_answerable"] == "Y"
