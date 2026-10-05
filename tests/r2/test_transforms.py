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
