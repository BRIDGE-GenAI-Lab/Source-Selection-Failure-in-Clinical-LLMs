import json

import pytest

from r2.paths import DATA_V2, VERIFY_DIR
from r2.verify.score_review import cohen_kappa, exclusion_reasons, flag_reasons


def _pairs():
    return json.loads((DATA_V2 / "pairs_v2.json").read_text())


def test_kappa_known_value():
    assert abs(cohen_kappa(["Y", "Y", "N", "N"], ["Y", "N", "N", "N"]) - 0.5) < 1e-9
    assert cohen_kappa(["Y", "Y"], ["Y", "Y"]) == 1.0


def test_exclusion_rule():
    ok = {"q3_authentic_concordant": "Y", "q4_authentic_relevant": "Y", "q5_edit_visible": "Y",
          "q6_edit_matches_type": "Y", "q7_edit_reduces_trust": "Y"}
    assert exclusion_reasons(ok) == []
    assert exclusion_reasons({**ok, "q5_edit_visible": "N"}) == ["q5_edit_visible"]
    assert exclusion_reasons({**ok, "q3_authentic_concordant": "P"}) == []


def test_flags_do_not_exclude():
    row = {"q1_vignette_consistent": "N", "q2_question_answerable": "Y", "q3_authentic_concordant": "P",
           "q8_leakage": "Y"}
    assert set(flag_reasons(row)) == {"q1_vignette_consistent", "q3_authentic_concordant", "q8_leakage"}


def test_packets_contain_rendered_input_only():
    if _pairs()["unresolved"]:
        pytest.skip("blocked: pairs_v2.json has unresolved pairs awaiting physician-authored edits")
    from r2.io import load_original_cases
    from r2.verify.build_packets import packet_rows
    cases, _ = load_original_cases()
    rows = packet_rows()
    assert len(rows) == 500
    for r in rows:
        ctx = (cases[r["case_id"]].get("assessment_context") or "").strip()
        if ctx:
            assert ctx not in r["vignette_as_shown_to_model"]
        assert r["authentic_as_shown_to_model"] and r["modified_as_shown_to_model"]


def test_write_forms_refuses_incomplete_set():
    from r2.verify.build_packets import write_forms
    d = _pairs()
    if len(d["cases"]) == 500 and not d["unresolved"]:
        pytest.skip("pairs_v2.json is complete; refusal not applicable")
    before = sorted(p.name for p in VERIFY_DIR.iterdir())
    with pytest.raises(RuntimeError):
        write_forms()
    assert sorted(p.name for p in VERIFY_DIR.iterdir()) == before
