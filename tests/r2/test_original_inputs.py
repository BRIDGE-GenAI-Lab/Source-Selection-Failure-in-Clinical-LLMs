from collections import Counter, defaultdict

import pytest

from r2.audit.original_inputs import (build_all, load_case_models, reconstruct_inline,
                                      reconstruct_openai_fc)
from r2.io import load_original_cases

INLINE = ("together_inline6", "gemini_inline7", "deepseek_chat_inline7", "deepseek_reasoner_inline7")


def _strip(d):
    return {k: v for k, v in d.items() if k != "tool_id"}


def test_case022_identical_except_label_in_every_profile():
    cases, _ = load_original_cases()
    for profile in INLINE:
        rec = reconstruct_inline(profile, cases["CASE_022"], {"A": "T", "B": "S"})
        assert _strip(rec["payloads"]["A"]) == _strip(rec["payloads"]["B"]), profile
    rec = reconstruct_openai_fc(load_case_models()["CASE_022"], {"A": "T", "B": "S"})
    assert _strip(rec["payloads"]["A"]) == _strip(rec["payloads"]["B"])


def test_case022_hidden_fields_never_reach_the_model():
    cases, _ = load_original_cases()
    hidden = "Most contraindications are relative in emergency settings"
    for profile in INLINE:
        rec = reconstruct_inline(profile, cases["CASE_022"], {"A": "S", "B": "T"})
        assert hidden not in rec["user"]
    rec = reconstruct_openai_fc(load_case_models()["CASE_022"], {"A": "S", "B": "T"})
    assert hidden not in rec["payloads"]["A"]["excerpt_text"]


@pytest.fixture(scope="module")
def rows():
    return build_all(write=False)


def test_all_logged_hashes_reproduce(rows):
    counts = Counter(r["hash_status"] for r in rows)
    assert counts["MISMATCH"] == 0
    assert counts["verified"] == 10496
    assert counts["no_logged_hash"] == 4


def test_logged_prompt_prefixes_match(rows):
    checked = [r for r in rows if r["prompt_prefix_verified"] is not None]
    assert len(checked) > 8000
    assert all(r["prompt_prefix_verified"] for r in checked)


def test_identical_set_is_167_cases_for_every_model(rows):
    by_model = defaultdict(set)
    for r in rows:
        if r["identical_except_label"]:
            by_model[r["model"]].add(r["case_id"])
    assert len(by_model) == 21
    assert all(len(s) == 167 for s in by_model.values())
    assert len(set.union(*by_model.values())) == 167


def test_no_runner_shows_assessment_context_or_case_flags(rows):
    cases, _ = load_original_cases()
    for r in rows[::7]:
        c = cases[r["case_id"]]
        ctx = (c.get("assessment_context") or "").strip()
        if ctx:
            assert ctx not in r["user"]
        for flag in c.get("case_flags") or []:
            if "_" in flag:
                assert flag not in r["user"]
