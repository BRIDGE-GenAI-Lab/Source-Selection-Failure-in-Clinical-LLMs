"""A7: the blank line and the page span in the original fabricated_citation inputs; the notes phrases and the
outdated_version years in all original inputs."""
import json

import pytest

from r2.audit import format_cues as F
from r2.paths import AUDIT_DIR

PROFILES = {"openai_fc": 4, "gemini_inline7": 1, "deepseek_chat_inline7": 1, "deepseek_reasoner_inline7": 1,
            "together_inline6": 14}


@pytest.fixture(scope="module")
def committed():
    return json.loads((AUDIT_DIR / "format_cues.json").read_text(encoding="utf-8"))


def test_every_original_fabricated_citation_input_carries_the_blank_line(committed):
    b = committed["blank_line_cue"]
    assert committed["n_rows"] == 10500 and committed["n_fabricated_citation_rows"] == 840
    assert b["modified_payload_rows"] == 840 and b["authentic_payload_rows"] == 0
    assert b["n_cases"] == 40 and b["n_models"] == 21
    assert b["by_profile"] == {p: 40 * n for p, n in PROFILES.items()}
    # the openai_fc payloads are function outputs: only the 17 inline models carry it in the message
    assert b["modified_serialized_message_rows"] == 40 * 17


def test_page_spans_are_implausible(committed):
    p = committed["page_span"]
    assert p["rows_with_a_citation"] == 840 and p["rows_without_a_parsable_citation"] == 0
    assert (p["min"], p["max"]) == (135, 1643) and p["cases_with_span_above_14"] == 40
    assert p["rows_within_plausible_range"] == 0 and len(p["span_by_case"]) == 40
    assert sum(sum(v.values()) for v in p["rows_by_profile_and_span_range"].values()) == 840


def test_notes_phrases_reached_no_model(committed):
    phrases = committed["notes_phrases"]["phrases"]
    assert set(phrases) == set(F.NOTES_PHRASES)
    assert {p: v["model_visible_rows"] for p, v in phrases.items()} == {p: 0 for p in F.NOTES_PHRASES}
    # the phrases are in the as-run case file, so the zero is a measurement, not a missing source
    assert phrases["Supported by recent mega-trial data"]["as_run_cases"] == 40
    assert phrases["Historical guideline version for reference only"]["as_run_cases"] == 50


def test_two_outdated_version_years_were_not_older(committed):
    o = committed["outdated_year_not_older"]
    assert o["case_ids"] == ["CASE_387", "CASE_488"] and o["n_cases"] == 2 and o["n_models"] == 21
    assert o["n_inputs"] == 2 * 21
    assert o["years_by_case"] == {"CASE_387": {"authentic": 2010, "modified": 2012},
                                  "CASE_488": {"authentic": 2014, "modified": 2014}}


def test_model_visible_text_covers_payloads_and_prompt():
    row = {"system": "sys", "user": "usr", "tools": [{"name": "t"}],
           "payloads": {"A": {"excerpt_text": "Supported by recent mega-trial data"}, "B": {}}}
    text = F._model_visible_text(row)
    assert all(part in text for part in ("sys", "usr", '"name": "t"', "Supported by recent mega-trial data"))
    assert F._year({"published_date": "2019 Update"}) == 2019 and F._year({}) is None


def test_audit_json_is_current(committed):
    assert {"blank_line_cue", "page_span", "notes_phrases", "outdated_year_not_older"} <= committed.keys()
    assert F.compute() == committed


def test_serialized_cue_detector():
    text = "a\n\nEvidence: x"
    assert F._serialized_has_cue(json.dumps({"excerpt_text": text}), text)
    assert F._serialized_has_cue("raw " + text, text)
    assert not F._serialized_has_cue("nothing here", text)
    assert not F._serialized_has_cue(json.dumps("a\nEvidence"), "a\nEvidence")


def test_readme_has_the_section():
    text = (AUDIT_DIR / "README.md").read_text(encoding="utf-8")
    assert "## format_cues.json" in text and "840 of 840" in text and "1,643" in text
    assert "mega-trial" in text and "CASE_387" in text and "42 inputs" in text
