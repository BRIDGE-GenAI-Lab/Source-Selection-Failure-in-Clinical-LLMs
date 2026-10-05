import json

import pytest

from r2.paths import DATA_V2, VERIFY_DIR
from r2.verify.score_review import cohen_kappa, exclusion_reasons, flag_reasons


def _pairs():
    return json.loads((DATA_V2 / "pairs_v2.json").read_text())


def test_kappa_known_value():
    assert abs(cohen_kappa(["Y", "Y", "N", "N"], ["Y", "N", "N", "N"]) - 0.5) < 1e-9
    assert cohen_kappa(["Y", "Y"], ["Y", "Y"]) == 1.0


def test_exclusion_rule():
    ok = {"q3_authentic_concordant": "Y", "q4_authentic_relevant": "Y", "q5_edit_visible": "Y",
          "q6_edit_matches_type": "Y", "q7_edit_reduces_trust": "Y"}
    assert exclusion_reasons(ok) == []
    assert exclusion_reasons({**ok, "q5_edit_visible": "N"}) == ["q5_edit_visible"]
    assert exclusion_reasons({**ok, "q3_authentic_concordant": "P"}) == []


def test_flags_do_not_exclude():
    row = {"q1_vignette_consistent": "N", "q2_question_answerable": "Y", "q3_authentic_concordant": "P",
           "q8_leakage": "Y"}
    assert set(flag_reasons(row)) == {"q1_vignette_consistent", "q3_authentic_concordant", "q8_leakage"}


def test_packets_contain_rendered_input_only():
    if _pairs()["unresolved"]:
        pytest.skip("blocked: pairs_v2.json has unresolved pairs awaiting physician-authored edits")
    from r2.io import load_original_cases
    from r2.verify.build_packets import packet_rows
    cases, _ = load_original_cases()
    rows = packet_rows()
    assert len(rows) == 500
    for r in rows:
        ctx = (cases[r["case_id"]].get("assessment_context") or "").strip()
        if ctx:
            assert ctx not in r["vignette_as_shown_to_model"]
        assert r["authentic_as_shown_to_model"] and r["modified_as_shown_to_model"]


def test_write_forms_refuses_incomplete_set():
    from r2.verify.build_packets import write_forms
    d = _pairs()
    if len(d["cases"]) == 500 and not d["unresolved"]:
        pytest.skip("pairs_v2.json is complete; refusal not applicable")
    before = sorted(p.name for p in VERIFY_DIR.iterdir())
    with pytest.raises(RuntimeError):
        write_forms()
    assert sorted(p.name for p in VERIFY_DIR.iterdir()) == before


# ---- score / consensus on toy forms (all I/O in tmp_path) ----
import pandas as pd  # noqa: E402

from r2.verify import score_review as sr  # noqa: E402

from r2.verify import build_packets as bp  # noqa: E402

QKEYS = [q for q, _ in sr.QUESTIONS]


def _packets():
    return {r["case_id"]: r["packet_sha256"] for r in bp.packet_rows()}


def _form(path, overrides=None, cases=("C1", "C2", "C3")):
    packets = _packets()
    rows = []
    for cid in cases:
        row = {"case_id": cid, "packet_sha256": packets.get(cid, ""), **{q: "Y" for q in QKEYS},
               "q8_leakage": "N", "notes": ""}
        row.update((overrides or {}).get(cid, {}))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _cons(path, rows):
    pd.DataFrame(rows, columns=["case_id", "item", "consensus"]).to_csv(path, index=False)
    return path


def _sha_file(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def toy(tmp_path, monkeypatch, make_pairs):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(sr, "VERIFY_DIR", tmp_path)
    monkeypatch.setattr(bp, "VERIFY_DIR", tmp_path)
    monkeypatch.setattr(bp, "DATA_V2", data)
    (data / "pairs_v2.json").write_text(json.dumps(make_pairs(["C1", "C2", "C3"]), indent=1))
    (tmp_path / "packets_manifest.json").write_text(json.dumps(
        {"pairs_v2_sha256": _sha_file(data / "pairs_v2.json"), "packets": _packets()}))
    a = _form(tmp_path / "a.csv", {"C1": {"q5_edit_visible": "N"}, "C2": {"q3_authentic_concordant": "P"}})
    b = _form(tmp_path / "b.csv", {"C1": {"q5_edit_visible": "N"}, "C3": {"q8_leakage": "Y"}})
    return tmp_path, a, b


def test_score_writes_kappa_and_disagreements(toy):
    d, a, b = toy
    sr.score(a, b)
    adj = pd.read_csv(d / "adjudication_needed.csv")
    assert set(zip(adj.case_id, adj["item"])) == {("C2", "q3_authentic_concordant"), ("C3", "q8_leakage")}
    assert set(pd.read_csv(d / "kappa.csv")["item"]) == set(QKEYS)


def test_consensus_refuses_unresolved_disagreement(toy):
    d, a, b = toy
    c = _cons(d / "c.csv", [("C2", "q3_authentic_concordant", "P")])
    with pytest.raises(ValueError, match="without consensus"):
        sr.apply_consensus(a, b, c)
    assert not (d / "excluded_pairs.json").exists()


def test_consensus_outputs_shapes(toy):
    d, a, b = toy
    c = _cons(d / "c.csv", [("C2", "q3_authentic_concordant", "P"), ("C3", "q8_leakage", "N")])
    sr.apply_consensus(a, b, c)
    ex = json.loads((d / "excluded_pairs.json").read_text())
    assert set(ex) == {"rule", "excluded", "reasons", "pairs_v2_sha256"}
    assert ex["pairs_v2_sha256"] == _sha_file(d / "data" / "pairs_v2.json")
    assert ex["excluded"] == ["C1"] and ex["reasons"] == {"C1": ["q5_edit_visible"]}
    assert json.loads((d / "flags.json").read_text()) == {"C2": ["q3_authentic_concordant"]}


@pytest.mark.parametrize("rule,rows,kw", [
    ("different case_ids", [], {"b_cases": ("C1", "C2", "X9")}),
    ("blank answers", [], {"a_blank": True}),
    ("unknown case_id", [("ZZ", "q8_leakage", "N")], {}),
    ("unknown item", [("C3", "q9_nope", "N")], {}),
    ("not allowed", [("C3", "q8_leakage", "P")], {}),
    ("not allowed", [("C3", "q8_leakage", "maybe")], {}),
    ("already agree", [("C1", "q5_edit_visible", "Y")], {}),
])
def test_consensus_input_validation(toy, rule, rows, kw):
    d, a, b = toy
    if "b_cases" in kw:
        b = _form(d / "b2.csv", cases=kw["b_cases"])
    if kw.get("a_blank"):
        a = _form(d / "a2.csv", {"C1": {"q1_vignette_consistent": ""}})
    base = [("C2", "q3_authentic_concordant", "P"), ("C3", "q8_leakage", "N")]
    c = _cons(d / "c.csv", base + rows)
    with pytest.raises(ValueError, match=rule):
        sr.apply_consensus(a, b, c)
    assert not (d / "excluded_pairs.json").exists()


def test_consensus_p_allowed_only_for_q3(toy):
    d, a, b = toy
    c = _cons(d / "c.csv", [("C2", "q3_authentic_concordant", "P"), ("C3", "q8_leakage", "Y")])
    sr.apply_consensus(a, b, c)


# ---- I4: bind physician verification to the stimulus file ----
SEP = "\n\x1e\n"


def test_packet_sha256_covers_the_four_model_facing_fields(toy):
    import hashlib
    for r in bp.packet_rows():
        parts = [r["vignette_as_shown_to_model"], r["clinical_question"], r["authentic_as_shown_to_model"],
                 r["modified_as_shown_to_model"]]
        assert r["packet_sha256"] == hashlib.sha256(SEP.join(parts).encode("utf-8")).hexdigest()


def test_consensus_refuses_packet_mismatch(toy):
    d, a, b = toy
    df = pd.read_csv(b, dtype=str).fillna("")
    df.loc[df.case_id == "C2", "packet_sha256"] = "0" * 64
    df.to_csv(b, index=False)
    c = _cons(d / "c.csv", [("C2", "q3_authentic_concordant", "P"), ("C3", "q8_leakage", "N")])
    with pytest.raises(ValueError, match="packet_sha256"):
        sr.apply_consensus(a, b, c)
    assert not (d / "excluded_pairs.json").exists()


def test_consensus_refuses_edited_pair(toy):
    d, a, b = toy
    pairs = json.loads((d / "data" / "pairs_v2.json").read_text())
    pairs["cases"][1]["tools"]["S"]["excerpt_text"] = "Give drug 1 at 100 mg daily."
    (d / "data" / "pairs_v2.json").write_text(json.dumps(pairs, indent=1))
    c = _cons(d / "c.csv", [("C2", "q3_authentic_concordant", "P"), ("C3", "q8_leakage", "N")])
    with pytest.raises(ValueError):
        sr.apply_consensus(a, b, c)
    assert not (d / "excluded_pairs.json").exists()


def test_consensus_refuses_pairs_file_changed_since_manifest(toy):
    d, a, b = toy
    path = d / "data" / "pairs_v2.json"
    path.write_text(json.dumps(json.loads(path.read_text()), indent=2))  # same packets, different bytes
    c = _cons(d / "c.csv", [("C2", "q3_authentic_concordant", "P"), ("C3", "q8_leakage", "N")])
    with pytest.raises(ValueError, match="pairs_v2_sha256"):
        sr.apply_consensus(a, b, c)
    assert not (d / "excluded_pairs.json").exists()


@pytest.fixture
def complete(tmp_path, monkeypatch, make_pairs):
    data, verify = tmp_path / "data", tmp_path / "verification"
    data.mkdir()
    monkeypatch.setattr(bp, "DATA_V2", data)
    monkeypatch.setattr(bp, "VERIFY_DIR", verify)
    (data / "pairs_v2.json").write_text(json.dumps(make_pairs([f"CASE_{i:03d}" for i in range(1, 501)])))
    return data, verify


def test_write_forms_writes_manifest(complete):
    data, verify = complete
    bp.write_forms()
    m = json.loads((verify / "packets_manifest.json").read_text())
    assert m["pairs_v2_sha256"] == _sha_file(data / "pairs_v2.json")
    assert m["packets"] == {r["case_id"]: r["packet_sha256"] for r in bp.packet_rows()}
    form = pd.read_csv(verify / "review_AG.csv", dtype=str)
    assert len(form) == 500 and form.set_index("case_id").packet_sha256.to_dict() == m["packets"]


def test_write_forms_refuses_to_overwrite_answered_forms(complete):
    data, verify = complete
    bp.write_forms()
    bp.write_forms()  # unanswered forms may be regenerated
    df = pd.read_csv(verify / "review_MO.csv", dtype=str).fillna("")
    df.loc[0, "q1_vignette_consistent"] = "Y"
    df.to_csv(verify / "review_MO.csv", index=False)
    before = {p.name: p.read_bytes() for p in verify.iterdir()}
    with pytest.raises(RuntimeError, match="review_MO.csv"):
        bp.write_forms()
    assert {p.name: p.read_bytes() for p in verify.iterdir()} == before


# ---- I5: answer validation ----
@pytest.mark.parametrize("item,value", [("q5_edit_visible", "No"), ("q4_authentic_relevant", "n/a"),
                                        ("q6_edit_matches_type", "P"), ("q8_leakage", "maybe"),
                                        ("q1_vignette_consistent", ""), ("q3_authentic_concordant", "partial")])
def test_load_rejects_invalid_answers(toy, item, value):
    d, a, b = toy
    bad = _form(d / "bad.csv", {"C2": {item: value}})
    with pytest.raises(ValueError, match=item):
        sr._load(bad)
    with pytest.raises(ValueError):
        sr.score(bad, b)
    with pytest.raises(ValueError):
        sr.apply_consensus(a, bad, _cons(d / "c.csv", []))


def test_load_normalizes_case_and_allows_blank_notes(toy):
    d, a, b = toy
    ok = _form(d / "ok.csv", {"C1": {"q3_authentic_concordant": " p ", "q2_question_answerable": "y", "notes": ""}})
    df = sr._load(ok)
    assert df.loc["C1", "q3_authentic_concordant"] == "P" and df.loc["C1", "q2_question_answerable"] == "Y"


def test_score_raises_value_error_on_different_pairs(toy):
    d, a, b = toy
    other = _form(d / "other.csv", cases=("C1", "C2"))
    with pytest.raises(ValueError, match="different"):
        sr.score(a, other)


def test_reviewer_readme_flags_case335_and_empty_allergy_lists(complete):
    data, verify = complete
    bp.write_forms()
    text = (verify / "README_REVIEWERS.md").read_text()
    assert ("CASE_335 has no clinical question (rendered 'Not specified'); judge Q2 and Q4 accordingly."
            in text)
    assert ("18 of the 50 allergy_ignorance cases have no recorded allergy (rendered 'No known drug allergies'); "
            "check Q7 (does the modification reduce trust for this patient?) explicitly for these.") in text
    assert "the not first-line" not in text


def test_case335_and_allergy_counts_still_hold():
    from r2.io import load_original_cases
    cases, _ = load_original_cases()
    assert not cases["CASE_335"].get("clinical_question")
    allergy = [c for c in cases.values() if c["eval"]["sham_trap_type"] == "allergy_ignorance"]
    assert len(allergy) == 50 and sum(not c.get("allergies") for c in allergy) == 18
