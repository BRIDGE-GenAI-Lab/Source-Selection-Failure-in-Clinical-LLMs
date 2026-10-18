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


# ---- Task DM item 5: robustness of render; none of these may change an existing packet ----
from r2.stimuli.render import render_vignette  # noqa: E402


def test_render_tool_handles_metadata_none_and_missing_fields():
    out = render_tool({"metadata": None, "excerpt_text": "x"}, "A")
    for k in ("organization", "document_title", "version", "publication_year"):
        assert out[k] == "not stated"
    out = render_tool({"metadata": {"organization": "Org", "publication_year": 2020}, "excerpt_text": "x"}, "A")
    assert out["organization"] == "Org" and out["publication_year"] == 2020
    assert out["document_title"] == "not stated" and out["version"] == "not stated"
    assert render_tool({"excerpt_text": "x"}, "A")["organization"] == "not stated"


@pytest.mark.parametrize("demo, expected", [
    ({"age": 50, "sex": "F"}, "Patient: 50-year-old F"),
    ({"sex": "F"}, "Patient: age not stated, F"),
    ({"age": 50}, "Patient: 50-year-old, sex not stated"),
    ({}, "Patient: age not stated, sex not stated"),
    (None, "Patient: age not stated, sex not stated"),
    ({"age": 50, "sex": "F", "weight_kg": 0}, "Patient: 50-year-old F, 0 kg"),
    ({"age": 50, "sex": "F", "weight_kg": 70}, "Patient: 50-year-old F, 70 kg"),
    ({"age": 50, "sex": "F", "weight_kg": None}, "Patient: 50-year-old F"),
])
def test_vignette_patient_line(demo, expected):
    assert expected in render_vignette({"demographics": demo}).splitlines()


def test_vitals_render_in_fixed_key_order():
    a = render_vignette({"vitals": {"spo2": 97, "hr": 80, "temp_c": 37.0}})
    b = render_vignette({"vitals": {"temp_c": 37.0, "spo2": 97, "hr": 80}})
    assert a == b
    assert "Vital signs: Temp (C) 37.0, HR 80, SpO2 (%) 97" in a.splitlines()


def test_render_changes_leave_every_round1_packet_unchanged_except_the_cue_fix_pairs():
    """Physicians review round-1 forms bound by packet_sha256; the manifest written with those forms must
    still equal the hashes recomputed with the current renderer, except for the 45 pairs whose stimulus the
    cue fixes changed on purpose (tests/r2/test_cue_fixes.py): round 2 re-reviews those."""
    from r2.verify.build_packets import packet_rows
    from tests.r2.test_cue_fixes import CUE_CHANGED_IDS
    manifest = json.loads((Path(__file__).resolve().parents[2] / "verification" / "round1"
                           / "packets_manifest.json").read_text())["packets"]
    current = {r["case_id"]: r["packet_sha256"] for r in packet_rows()}
    assert {cid: current[cid] for cid in manifest if cid not in CUE_CHANGED_IDS} == \
        {cid: h for cid, h in manifest.items() if cid not in CUE_CHANGED_IDS}
    assert sorted(cid for cid in manifest if current[cid] != manifest[cid]) == CUE_CHANGED_IDS
    assert len(manifest) == 477
