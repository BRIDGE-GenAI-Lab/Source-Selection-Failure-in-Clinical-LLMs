import copy

import pytest


def synthetic_pairs(case_ids, unresolved=(), identical=()):
    """A minimal pairs_v2.json payload: one dosing edit per pair, visible in excerpt_text."""
    cases = []
    for i, cid in enumerate(case_ids):
        meta = {"organization": "Example Society", "guideline_title": "Example Guideline",
                "version_label": "v1", "publication_year": 2020, "source_pdf_path": "/x/example.pdf"}
        t = {"metadata": meta, "excerpt_text": f"Give drug {i} at 1 mg daily.",
             "safety_guardrails": ["Check renal function"], "applicability": ["Adults"],
             "exclusions_or_cautions": []}
        s = copy.deepcopy(t)
        if cid not in identical:
            s["excerpt_text"] = f"Give drug {i} at 10 mg daily."
        cases.append({"case_id": cid, "setting": "ED", "demographics": {"age": 50, "sex": "F"},
                      "chief_complaint": "pain", "clinical_question": f"Dose for case {i}?",
                      "eval": {"sham_trap_type": "dosing_error"}, "tools": {"T": t, "S": s},
                      "edit_record": {"edit_source": "original_transform", "fields_changed": ["excerpt_text"],
                                      "changes": [{"field": "excerpt_text", "before": t["excerpt_text"],
                                                   "after": s["excerpt_text"]}]}})
    return {"derived_from_sha256": "synthetic", "n_pairs": len(cases), "unresolved": list(unresolved),
            "cases": cases}


@pytest.fixture
def make_pairs():
    return synthetic_pairs
