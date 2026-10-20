from collections import Counter, defaultdict

import pytest

from r2.audit.original_inputs import (build_all, load_case_models, reconstruct_inline,
                                      reconstruct_openai_fc)
from r2.io import load_original_cases
from r2.paths import LEGACY_RESULTS

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


@pytest.mark.skipif(LEGACY_RESULTS is None, reason="the logged prompts are in the raw run directories (R2_LEGACY_ROOT)")
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
    for r in rows:
        c = cases[r["case_id"]]
        ctx = (c.get("assessment_context") or "").strip()
        if ctx:
            assert ctx not in r["user"]
        for flag in c.get("case_flags") or []:
            if "_" in flag:
                assert flag not in r["user"]


def test_verification_scope_present(rows):
    from r2.audit.original_inputs import SCOPE_OPENAI_FC
    assert all(r["verification_scope"] for r in rows)
    for r in rows:
        assert (r["verification_scope"] == SCOPE_OPENAI_FC) == (r["profile"] == "openai_fc")
    assert SCOPE_OPENAI_FC.startswith("excerpt_text sha256 (logged by src/runner.py)")


# ---- Task DM item 2: guards, scoped environment, warnings, _logged_prefix ----
def test_mapping_mismatch_raises_value_error_not_assert(monkeypatch):
    import types

    import r2.audit.original_inputs as oi
    cases, _ = load_original_cases()
    fake = types.SimpleNamespace(sanitize_for_case=lambda case, rng: ({}, {}, {"A": "S", "B": "T"}))
    monkeypatch.setattr(oi, "load_runner", lambda rel: fake)
    with pytest.raises(ValueError, match="sanitizer assigned"):
        oi.reconstruct_inline("together_inline6", cases["CASE_001"], {"A": "T", "B": "S"})


def test_openai_mapping_mismatch_raises_value_error():
    case = load_case_models()["CASE_001"]
    # the FixedDraw for mapping A=T always yields A=T, so ask for the opposite mapping of what it assigns
    import r2.audit.original_inputs as oi

    class Flip(oi.FixedDraw):
        def __init__(self, mapping):
            super().__init__({"A": "S" if mapping["A"] == "T" else "T", "B": mapping["B"]})
    orig = oi.FixedDraw
    oi.FixedDraw = Flip
    try:
        with pytest.raises(ValueError, match="sanitizer assigned"):
            oi.reconstruct_openai_fc(case, {"A": "T", "B": "S"})
    finally:
        oi.FixedDraw = orig


def _stub_runner(tmp_path, monkeypatch, body):
    import r2.audit.original_inputs as oi
    (tmp_path / "stub_runner.py").write_text(body, encoding="utf-8")
    monkeypatch.setattr(oi, "REPO", tmp_path)
    monkeypatch.setattr(oi, "_MODULES", {})
    return oi


def test_load_runner_restores_environ_and_sys_path(tmp_path, monkeypatch):
    import os
    import sys
    oi = _stub_runner(tmp_path, monkeypatch,
                      "import os, sys\nSEEN = os.environ.get('TOGETHER_API_KEY')\nsys.path.insert(0, '/nonexistent-dm')\n"
                      "os.environ['LEAK_FROM_RUNNER'] = '1'\n")
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    before_env, before_path = dict(os.environ), list(sys.path)
    mod = oi.load_runner("stub_runner.py")
    assert mod.SEEN == "unused-for-reconstruction"  # the runner still sees the placeholder at import
    assert dict(os.environ) == before_env and sys.path == before_path


def test_load_case_models_restores_sys_path(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(__import__("r2.paths").paths.REPO)])
    before = list(sys.path)
    load_case_models()
    assert sys.path == before


def test_load_runner_silences_only_the_google_future_warning(tmp_path, monkeypatch):
    import warnings
    oi = _stub_runner(tmp_path, monkeypatch,
                      "import warnings\n"
                      "warnings.warn('\\nAll support for the `google.generativeai` package has ended.', FutureWarning)\n"
                      "warnings.warn('unrelated future change', FutureWarning)\n")
    filters_before = list(warnings.filters)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        oi.load_runner("stub_runner.py")
    assert [str(w.message) for w in rec] == ["unrelated future change"]
    assert list(warnings.filters) == filters_before


def test_logged_prefix_distinguishes_missing_file_from_missing_key(tmp_path, monkeypatch):
    import json

    import r2.audit.original_inputs as oi
    monkeypatch.setattr(oi, "LEGACY_RESULTS", tmp_path)
    raw = tmp_path / "run_x" / "raw"
    raw.mkdir(parents=True)
    assert oi._logged_prefix("run_x", "CASE_001") is None  # no file
    (raw / "CASE_002_response.json").write_text(json.dumps({"response": "x"}))
    with pytest.raises(KeyError, match="no 'prompt' key"):
        oi._logged_prefix("run_x", "CASE_002")
    (raw / "CASE_003_response.json").write_text(json.dumps({"prompt": "hello"}))
    assert oi._logged_prefix("run_x", "CASE_003") == "hello"
