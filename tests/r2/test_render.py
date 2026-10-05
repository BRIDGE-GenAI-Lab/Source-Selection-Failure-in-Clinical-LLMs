import json
from pathlib import Path

import pytest

from r2.io import load_original_cases
from r2.stimuli.render import render_messages, render_tool, visible_content

CASES, _ = load_original_cases()
REPO_ROOT = Path(__file__).resolve().parents[2]
# The corrected set, including rule-generated and physician-authored edits, is leakage-tested too.
V2_CASES = {c["case_id"]: c for c in json.loads((REPO_ROOT / "data" / "v2" / "pairs_v2.json").read_text())["cases"]}
LEAKAGE_CASES = [("original", cid) for cid in sorted(CASES)] + [("v2", cid) for cid in sorted(V2_CASES)]

# Reviewed benign leakage-token matches: {case_id: [{substring, location, reason}]}.
FLAGS = json.loads((REPO_ROOT / "verification" / "leakage_flags.json").read_text())
VIGNETTE_FIELDS = ("setting", "chief_complaint", "history_present_illness", "clinical_question")
TOOL_KEYS = {"label", "organization", "document_title", "version", "publication_year", "excerpt_text",
             "safety_guardrails", "applicability", "exclusions_or_cautions"}


def _source_text(case, location):
    if location.startswith("tool_"):
        tool_id, field = location[5:].split(".", 1)
        val = render_tool(case["tools"][tool_id], "X")[field]
        return "\n".join(map(str, val)) if isinstance(val, list) else str(val)
    return str(case.get(location) or "")


def _allowed(cid):
    return {e["substring"] for e in FLAGS.get(cid, [])}


def _check_allowed(cid, case, user, sub):
    locs = [e["location"] for e in FLAGS[cid] if e["substring"] == sub]
    expected = sum(_source_text(case, loc).lower().count(sub) for loc in locs)
    assert user.lower().count(sub) == expected, (cid, sub, locs)


def test_hidden_fields_are_now_rendered_for_case022():
    msgs = render_messages(CASES["CASE_022"], "ST")
    assert "Most contraindications are relative in emergency settings" in msgs["user"]
    assert visible_content(CASES["CASE_022"]["tools"]["T"]) != visible_content(CASES["CASE_022"]["tools"]["S"])


@pytest.mark.parametrize("source,cid", LEAKAGE_CASES)
def test_no_leakage_fields(source, cid):
    case = (CASES if source == "original" else V2_CASES)[cid]
    user = render_messages(case, "TS")["user"]
    allowed = _allowed(cid)
    ctx = (case.get("assessment_context") or "").strip()
    if ctx:
        assert ctx not in user
    for flag in case.get("case_flags") or []:
        if "_" in flag:
            if flag in allowed:
                _check_allowed(cid, case, user, flag)
            else:
                assert flag not in user
    for token in ("sham", "trap_type", "modification_log", "preferred_tool", "poison", "source_pdf_path"):
        if token in allowed:
            _check_allowed(cid, case, user, token)
        else:
            assert token not in user.lower()


def test_orders_swap_payloads_exactly():
    c = CASES["CASE_001"]
    ts, st = render_messages(c, "TS"), render_messages(c, "ST")
    assert ts["labels"] == {"A": "T", "B": "S"} and st["labels"] == {"A": "S", "B": "T"}
    assert ts["payload_sha256"]["A"] != st["payload_sha256"]["A"]
    assert render_messages(c, "TS")["payload_sha256"] == ts["payload_sha256"]


def _strip(d):
    return {k: v for k, v in d.items() if k != "label"}


def test_swapped_orders_carry_identical_payloads_all_cases():
    for cid, c in CASES.items():
        ts, st = render_messages(c, "TS"), render_messages(c, "ST")
        t, s_ = c["tools"]["T"], c["tools"]["S"]
        assert _strip(render_tool(t, "A")) == _strip(render_tool(t, "B")) == visible_content(t)
        # T is A under TS and B under ST; same for S, and the digests follow the tool, not the slot
        assert ts["labels"]["A"] == st["labels"]["B"] == "T" and ts["labels"]["B"] == st["labels"]["A"] == "S"
        for order_msgs, la, lb in ((ts, t, s_), (st, s_, t)):
            assert json.dumps(render_tool(la, "A"), indent=2, ensure_ascii=False) in order_msgs["user"]
            assert json.dumps(render_tool(lb, "B"), indent=2, ensure_ascii=False) in order_msgs["user"]
        assert ts["system"] == st["system"]


def test_render_tool_key_whitelist_all_tools():
    for cid, c in CASES.items():
        for tid in ("T", "S"):
            assert set(render_tool(c["tools"][tid], "A")) == TOOL_KEYS, (cid, tid)


def test_full_vignette_fields_present():
    user = render_messages(CASES["CASE_022"], "TS")["user"]
    assert "CT Head: Old infarct, no acute bleed" in user
    assert "Clinical question" in user


def test_rejects_bad_order():
    with pytest.raises(ValueError):
        render_messages(CASES["CASE_001"], "AB")
