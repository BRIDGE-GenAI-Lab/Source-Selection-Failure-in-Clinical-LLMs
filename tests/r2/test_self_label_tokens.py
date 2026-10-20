"""Final fix F1: the self-labelling token "fake" in the original fabricated-citation edits. The audit must disclose
the token in the original model inputs."""
import json

import pytest

from r2.io import load_original_cases
from r2.paths import AUDIT_DIR

FC = "fabricated_citation"
PROFILES = {"openai_fc": 4, "gemini_inline7": 1, "deepseek_chat_inline7": 1, "deepseek_reasoner_inline7": 1,
            "together_inline6": 14}


def _fc_ids():
    cases, _ = load_original_cases()
    return {cid for cid, c in cases.items() if c["eval"]["sham_trap_type"] == FC}


@pytest.fixture(scope="module")
def committed():
    return json.loads((AUDIT_DIR / "self_label_tokens.json").read_text(encoding="utf-8"))


def test_audit_counts_840_fake_rows_all_fabricated_citation(committed):
    fake = committed["modified_tool_payload"]["fake"]
    assert fake["rows"] == 840 and fake["n_cases"] == 40 and fake["n_models"] == 21
    assert fake["by_sham_type"] == {FC: 840}
    assert fake["by_profile"] == {p: 40 * n for p, n in PROFILES.items()}
    assert sorted(fake["case_ids"]) == sorted(_fc_ids())
    assert committed["n_rows"] == 10500 and committed["n_models"] == 21
    for tok in committed["tokens"]:
        assert committed["authentic_tool_payload"][tok]["rows"] == 0, tok
        if tok != "fake":
            assert committed["modified_tool_payload"][tok]["rows"] == 0, tok
            assert committed["prompt_text"][tok]["rows"] == 0, tok


def test_audit_discloses_the_openai_instruction_mention(committed):
    p = committed["prompt_text"]["fake"]
    assert p["rows"] == 2000 and p["by_profile"] == {"openai_fc": 2000}
    assert sum(p["by_sham_type"].values()) == 2000
    assert any("fake citations" in c for c in p["contexts"])


def test_audit_json_is_current(committed):
    from r2.audit.self_label_tokens import compute
    assert compute() == committed


def test_audit_readme_has_the_section():
    text = (AUDIT_DIR / "README.md").read_text(encoding="utf-8")
    assert "## self_label_tokens.json" in text
    assert "840" in text and "cue-confounded" in text and "10.9999/fake." in text and "10.9999/cg." in text
