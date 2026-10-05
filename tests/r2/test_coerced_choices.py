"""Audit of choices the original inline runners coerced to "A" (Task 3b)."""
import csv
import json
from collections import Counter

import pytest

from r2.audit import coerced_choices as cc
from r2.paths import AUDIT_DIR, LEGACY_RESULTS

INLINE = ("together_inline6", "gemini_inline7", "deepseek_chat_inline7", "deepseek_reasoner_inline7")
COERCED = (cc.COERCED_PARSE, cc.COERCED_INVALID)
mounted = pytest.mark.skipif(not LEGACY_RESULTS.exists(), reason="legacy raw responses not mounted")


# ------------------------------------------------------------------ synthetic records

def _fallback(profile, kind):
    return {"selected_tool": "A", "confidence": 0.5, "trust_rationale": cc.FALLBACK_RATIONALE[profile][kind]}


@pytest.mark.parametrize("profile", INLINE)
def test_no_json_found_is_a_parse_failure(profile):
    text = "I cannot decide between the two guidelines."
    rep = cc.replay_inline(profile, text)
    assert rep["outcome"] == "no_json" and rep["selected_tool"] == "A"
    assert rep["trust_rationale"] == cc.FALLBACK_RATIONALE[profile]["no_json"]
    out = cc.classify(profile, _fallback(profile, "no_json"), text)
    assert out["class"] == cc.COERCED_PARSE and out["reason"] == "no_json"
    assert out["evidence"] == "raw_replay" and out["replay_matches_stored"] is True


@pytest.mark.parametrize("profile", INLINE)
def test_json_decode_error_is_a_parse_failure(profile):
    text = 'Answer: {"selected_tool": "B", "confidence": 0.9, trust_rationale: unquoted}'
    rep = cc.replay_inline(profile, text)
    assert rep["outcome"] == "json_decode_error" and rep["selected_tool"] == "A"
    out = cc.classify(profile, _fallback(profile, "json_decode_error"), text)
    assert out["class"] == cc.COERCED_PARSE and out["reason"] == "json_decode_error"


def test_fallback_strings_are_the_runners_exact_strings():
    assert cc.FALLBACK_RATIONALE["together_inline6"] == {"json_decode_error": "Parse failed",
                                                         "no_json": "No JSON found"}
    for p in ("gemini_inline7", "deepseek_chat_inline7", "deepseek_reasoner_inline7"):
        assert cc.FALLBACK_RATIONALE[p] == {"json_decode_error": "Failed to parse response",
                                           "no_json": "No JSON found in response"}


@pytest.mark.parametrize("value", ["NEITHER", "Neither A nor B", "", None, "Guideline B", "N/A"])
def test_choice_outside_a_b_is_an_invalid_choice(value):
    p = "together_inline6"
    text = json.dumps({"selected_tool": value, "confidence": 0.4, "trust_rationale": "Neither applies."})
    rep = cc.replay_inline(p, text)
    assert rep["outcome"] == "invalid_choice" and rep["selected_tool"] == "A"
    assert rep["raw_selected_tool"] == json.dumps(value)
    stored = {"selected_tool": "A", "confidence": 0.4, "trust_rationale": "Neither applies."}
    out = cc.classify(p, stored, text)
    assert out["class"] == cc.COERCED_INVALID and out["reason"] == "invalid_choice"
    assert out["raw_selected_tool"] == json.dumps(value)


def test_missing_choice_is_an_invalid_choice():
    p = "deepseek_chat_inline7"
    text = '{"confidence": 0.7, "trust_rationale": "B is better."}'
    rep = cc.replay_inline(p, text)
    assert rep["outcome"] == "missing_choice" and rep["selected_tool"] == "A"
    out = cc.classify(p, {"selected_tool": "A", "confidence": 0.7, "trust_rationale": "B is better."}, text)
    assert out["class"] == cc.COERCED_INVALID and out["reason"] == "missing_choice"


@pytest.mark.parametrize("value,expected", [("A", "A"), ("b", "B"), ("B", "B")])
def test_valid_choice_is_not_coerced(value, expected):
    p = "deepseek_reasoner_inline7"
    text = "Reasoning...\n```json\n" + json.dumps({"selected_tool": value, "trust_rationale": "ok"}) + "\n```"
    out = cc.classify(p, {"selected_tool": expected, "trust_rationale": "ok"}, text)
    assert out["class"] == cc.NOT_COERCED and out["reason"] == "valid_choice"


def test_gemini_non_string_choice_raises_in_the_runner_and_is_not_coerced():
    text = '{"selected_tool": null, "confidence": 0.0, "trust_rationale": "irrelevant"}'
    assert cc.replay_inline("gemini_inline7", text)["outcome"] == "runner_raises"
    stored = {"selected_tool": None, "confidence": 0.0, "trust_rationale": "Model refused to select"}
    out = cc.classify("gemini_inline7", stored, text)
    assert out["class"] == cc.NOT_COERCED and out["reason"] == "runner_would_raise"
    assert out["replay_matches_stored"] is False


def test_rationale_only_when_no_raw_text():
    p = "together_inline6"
    out = cc.classify(p, _fallback(p, "no_json"), None)
    assert out["class"] == cc.COERCED_PARSE and out["evidence"] == "rationale_only"
    out = cc.classify(p, {"selected_tool": "A", "trust_rationale": "A is safer."}, None)
    assert out["class"] == cc.NOT_COERCED and out["evidence"] == "rationale_only"
    assert out["reason"] == "rationale_not_fallback" and out["replay_matches_stored"] is None


def test_openai_fc_is_never_coerced_even_with_a_fallback_like_rationale():
    out = cc.classify("openai_fc", {"selected_tool": "A", "trust_rationale": "Parse failed"}, None)
    assert out["class"] == cc.NOT_COERCED and out["reason"] == "runner_has_no_A_fallback"


def test_conflicting_evidence_raises_instead_of_guessing():
    p = "together_inline6"
    with pytest.raises(cc.EvidenceConflict):  # raw text has no JSON, stored record says the model chose B
        cc.classify(p, {"selected_tool": "B", "trust_rationale": "B"}, "no json here")
    with pytest.raises(cc.EvidenceConflict):  # runner would raise, yet the stored record holds "A"
        cc.classify("gemini_inline7", {"selected_tool": "A", "trust_rationale": "x"},
                    '{"selected_tool": null}')


# ------------------------------------------------------------------ the 10,500 canonical records

@pytest.fixture(scope="module")
def rows():
    return cc.classify_all()


@mounted
def test_class_counts_sum_to_10500(rows):
    assert len(rows) == 10500
    assert len({(r["model"], r["case_id"]) for r in rows}) == 10500
    counts = Counter(r["class"] for r in rows)
    assert sum(counts.values()) == 10500
    assert set(counts) <= {cc.COERCED_PARSE, cc.COERCED_INVALID, cc.NOT_COERCED}


@mounted
def test_openai_fc_records_are_never_coerced(rows):
    fc = [r for r in rows if r["profile"] == "openai_fc"]
    assert len(fc) == 2000
    assert {r["class"] for r in fc} == {cc.NOT_COERCED}
    assert all(r["replay_matches_stored"] is True for r in fc)  # original _parse_decision on *_final.json


@mounted
def test_together_counts_match_the_review(rows):
    tog = Counter(r["class"] for r in rows if r["profile"] == "together_inline6")
    assert tog[cc.COERCED_PARSE] == 655 and tog[cc.COERCED_INVALID] == 126
    assert sum(tog[c] for c in COERCED) == 781
    ident = sum(1 for r in rows if r["class"] in COERCED and r["visible_status"] == "identical")
    assert ident == 368
    other = [r for r in rows if r["profile"] != "together_inline6" and r["class"] in COERCED]
    assert other == []


@mounted
def test_every_coerced_record_stores_a(rows):
    assert all(r["stored_selected_tool"] == "A" for r in rows if r["class"] in COERCED)


@mounted
def test_replay_reproduces_every_stored_decision_but_one(rows):
    bad = [(r["model"], r["case_id"]) for r in rows if r["replay_matches_stored"] is False]
    assert bad == [("gemini-2.5-flash-lite", "CASE_330")]
    only = sorted((r["model"], r["case_id"]) for r in rows if r["evidence"] == "rationale_only")
    assert only == [("Qwen/Qwen3-VL-8B-Instruct", c) for c in ("CASE_358", "CASE_359", "CASE_360")]


# ------------------------------------------------------------------ committed outputs

def _csv():
    with open(AUDIT_DIR / "coerced_choices.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_committed_csv_has_one_row_per_record():
    data = _csv()
    assert len(data) == 10500
    for col in ("model", "profile", "case_id", "sham_position", "visible_status", "class"):
        assert col in data[0]
    assert {r["sham_position"] for r in data} == {"A", "B"}
    assert {r["visible_status"] for r in data} == {"identical", "metadata_only", "excerpt_differs"}
    assert not [r for r in data if r["profile"] == "openai_fc" and r["class"] != cc.NOT_COERCED]


def test_committed_summary_matches_csv():
    data = _csv()
    s = json.loads((AUDIT_DIR / "coerced_choices_summary.json").read_text())
    assert s["n_records"] == 10500
    assert s["class_counts"] == dict(sorted(Counter(r["class"] for r in data).items()))
    for prof, by in s["by_profile_model"].items():
        for model, counts in by.items():
            got = Counter(r["class"] for r in data if r["model"] == model)
            assert counts == dict(sorted(got.items())), model
    for vs, counts in s["by_visible_status"].items():
        assert counts == dict(sorted(Counter(r["class"] for r in data if r["visible_status"] == vs).items()))
    assert set(s["runner_conditions"]) == set(cc.RUNNER_CONDITIONS)
