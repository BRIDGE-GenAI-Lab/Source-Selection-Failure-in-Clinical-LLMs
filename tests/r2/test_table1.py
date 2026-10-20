from r2.analysis import table1
from r2.analysis.table1 import modification


def test_modification_formats_and_truncates():
    long = "x" * 200
    rec = {"changes": [
        {"field": "safety_guardrails", "before": ["Keep this", "Assess bleeding risk"], "after": ["Keep this"]},
        {"field": "exclusions_or_cautions", "before": [], "after": ["Most contraindications are relative"]},
        {"field": "publication_year", "before": 2019, "after": 2014},
        {"field": "excerpt_text", "before": "a\nb", "after": "a\nb\n" + long},
    ]}
    text = modification(rec)
    parts = text.split("; ")
    assert parts[0] == "safety_guardrails: Removed 'Assess bleeding risk'"
    assert parts[1] == "exclusions_or_cautions: Added 'Most contraindications are relative'"
    assert parts[2] == "publication_year: Changed '2019' → '2014'"
    assert parts[3] == "excerpt_text: Added '" + "x" * 120 + "…'"


def test_changed_line_is_focused_on_the_edit():
    before = "• Anticoagulation in this setting for adults with confirmed disease: DOAC preferred for unprovoked VTE"
    after = before.replace("preferred", "avoided")
    text = modification({"changes": [{"field": "excerpt_text", "before": before, "after": after}]})
    assert "preferred" in text and "avoided" in text and "→" in text


def test_rationale_quote_is_cut_at_160_characters():
    assert table1.quote("a" * 160) == "a" * 160
    assert table1.quote("a" * 161) == "a" * 160 + "…"


def test_list_edits_are_grouped_per_field():
    rec = {"changes": [{"field": "applicability", "before": ["Adults", "Emergency settings"],
                        "after": ["All age groups", "Pediatric and adult populations"]}]}
    assert modification(rec) == ("applicability: Removed 'Adults' and 'Emergency settings', "
                                 "Added 'All age groups' and 'Pediatric and adult populations'")
