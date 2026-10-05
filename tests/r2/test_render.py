import pytest

from r2.io import load_original_cases
from r2.stimuli.render import render_messages, render_tool, visible_content

CASES, _ = load_original_cases()

# Explicit allow-list of (case, substring) leakage matches reviewed as legitimate clinical text.
# Recorded in verification/leakage_flags.json for physician Q8 review.
_GI = "gi_bleed"  # tool applicability text reads "Adults presenting with gi_bleed-related conditions"
ALLOWED = {
    "CASE_010": {_GI},
    "CASE_055": {_GI},
    "CASE_130": {_GI},
    "CASE_142": {_GI},
    "CASE_273": {_GI},
    "CASE_342": {_GI},
    "CASE_457": {_GI},
    "CASE_192": {"poison"},  # clinical question: carbon monoxide poisoning
    "CASE_229": {"poison"},  # clinical question: CO poisoning in pregnancy
    "CASE_387": {"poison"},  # HPI: Contact Dermatitis. Poison Ivy.
}


def test_hidden_fields_are_now_rendered_for_case022():
    msgs = render_messages(CASES["CASE_022"], "ST")
    assert "Most contraindications are relative in emergency settings" in msgs["user"]
    assert visible_content(CASES["CASE_022"]["tools"]["T"]) != visible_content(CASES["CASE_022"]["tools"]["S"])


@pytest.mark.parametrize("cid", sorted(CASES))
def test_no_leakage_fields(cid):
    case = CASES[cid]
    user = render_messages(case, "TS")["user"]
    allowed = ALLOWED.get(cid, set())
    ctx = (case.get("assessment_context") or "").strip()
    if ctx:
        assert ctx not in user
    for flag in case.get("case_flags") or []:
        if "_" in flag:
            assert flag in allowed or flag not in user
    for token in ("sham", "trap_type", "modification_log", "preferred_tool", "poison", "source_pdf_path"):
        assert token in allowed or token not in user.lower()


def test_orders_swap_payloads_exactly():
    c = CASES["CASE_001"]
    ts, st = render_messages(c, "TS"), render_messages(c, "ST")
    assert ts["labels"] == {"A": "T", "B": "S"} and st["labels"] == {"A": "S", "B": "T"}
    assert ts["payload_sha256"]["A"] != st["payload_sha256"]["A"]
    assert render_tool(c["tools"]["T"], "A") == render_tool(c["tools"]["T"], "A")


def test_full_vignette_fields_present():
    user = render_messages(CASES["CASE_022"], "TS")["user"]
    assert "CT Head: Old infarct, no acute bleed" in user
    assert "Clinical question" in user


def test_rejects_bad_order():
    with pytest.raises(ValueError):
        render_messages(CASES["CASE_001"], "AB")
