import pytest

from r2.stimuli.transforms import TransformError, apply_manual_edit, propose_inversion, typosquat


def test_typosquat_changes_one_word_deterministically():
    org = "Infectious Diseases Society of America/American Thoracic Society"
    a, b = typosquat(org), typosquat(org)
    assert a == b and a != org and len(a) == len(org)
    assert sum(x != y for x, y in zip(a, org)) == 2


def test_typosquat_needs_a_long_word():
    with pytest.raises(TransformError):
        typosquat("ACC AHA")


def test_inversion_respects_word_boundaries_and_existing_negation():
    assert propose_inversion("Give 1 g IV now") is None
    new, old, rep = propose_inversion("Shoulder imaging should be obtained")
    assert new == "Shoulder imaging should not be obtained"
    new, _, _ = propose_inversion("Thrombolysis is not recommended. Imaging is recommended.")
    assert new == "Thrombolysis is not recommended. Imaging is not recommended."


def test_manual_edit_requires_exactly_one_occurrence():
    tool = {"excerpt_text": "A. B. A.", "metadata": {"organization": "X"}}
    with pytest.raises(TransformError):
        apply_manual_edit(tool, {"field": "excerpt_text", "find": "A.", "replace": "C."})
    out = apply_manual_edit(tool, {"field": "excerpt_text", "find": "B.", "replace": "C."})
    assert out["excerpt_text"] == "A. C. A." and tool["excerpt_text"] == "A. B. A."


def test_inversion_preserves_initial_capital():
    new, old, rep = propose_inversion("Recommended for all adults. Monitor renal function.")
    assert new == "Not recommended for all adults. Monitor renal function."
    assert (old, rep) == ("Recommended", "Not recommended")
    new, _, _ = propose_inversion("• Consider early imaging")
    assert new == "• Avoid early imaging"


def test_first_line_becomes_second_line():
    new, _, rep = propose_inversion("Metformin is the first-line agent.")
    assert new == "Metformin is the second-line agent." and rep == "second-line"
    new, _, _ = propose_inversion("First-line therapy is metformin.")
    assert new == "Second-line therapy is metformin."


# ---- Final fix F1: no self-labelling token in a modified excerpt ----
import re  # noqa: E402

from r2.stimuli.transforms import (  # noqa: E402
    SELF_LABEL_TOKENS, neutral_doi_suffix, neutralize_citation_doi, self_label_tokens)

ORIGINAL_EVIDENCE = "• Thrombolysis for massive PE\n\nEvidence: 'NEJM 2025;392:649–1127' (DOI:10.9999/fake.7201)"


def test_neutral_doi_suffix_is_journal_style_and_pinned():
    assert re.fullmatch(r"cg\.2025\.\d{4}", neutral_doi_suffix("CASE_060", "2025"))
    # sha256(case_id) mod 10000, zero-padded: pinned so a change to the rule is a visible test failure
    assert neutral_doi_suffix("CASE_060", "2025") == "cg.2025.9873"
    assert neutral_doi_suffix("CASE_085", "2025") == "cg.2025.8068"
    assert neutral_doi_suffix("CASE_101", "2025") == "cg.2025.5026"
    assert neutral_doi_suffix("CASE_060", 2024) == "cg.2024.9873"


def test_neutral_doi_suffix_is_deterministic():
    assert all(neutral_doi_suffix("CASE_060", "2025") == neutral_doi_suffix("CASE_060", "2025") for _ in range(3))
    assert neutral_doi_suffix("CASE_060", "2025") != neutral_doi_suffix("CASE_085", "2025")


def test_neutralize_replaces_only_the_doi_suffix():
    new, before, after = neutralize_citation_doi(ORIGINAL_EVIDENCE, "CASE_060")
    assert (before, after) == ("10.9999/fake.7201", "10.9999/cg.2025.9873")
    assert new == ORIGINAL_EVIDENCE.replace("10.9999/fake.7201", "10.9999/cg.2025.9873")
    assert self_label_tokens(new) == [] and self_label_tokens(ORIGINAL_EVIDENCE) == ["fake"]


@pytest.mark.parametrize("text", [
    "• No citation here",
    ORIGINAL_EVIDENCE + " (DOI:10.9999/fake.1111)",  # two self-labelling DOIs
    "• X (DOI:10.9999/fake.7201)",  # no citation year to build the suffix from
])
def test_neutralize_refuses_without_exactly_one_doi_and_year(text):
    with pytest.raises(TransformError):
        neutralize_citation_doi(text, "CASE_060")


def test_self_label_tokens_whole_word_case_insensitive():
    assert set(SELF_LABEL_TOKENS) == {"fake", "fabricated", "sham", "bogus", "dummy", "placeholder",
                                      "made-up", "made up", "not real"}
    assert self_label_tokens("DOI:10.9999/FAKE.1") == ["fake"]
    assert self_label_tokens("line one\nfake line") == ["fake"]
    assert self_label_tokens("Hisham, fakery, shampoo, dummyvar, placeholders, unreal") == []
    assert self_label_tokens("a Made-Up trial; data made\nup; NOT REAL; a Sham arm") == [
        "made up", "made-up", "not real", "sham"]
