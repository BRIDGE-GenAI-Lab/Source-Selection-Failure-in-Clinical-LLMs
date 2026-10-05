from collections import Counter

from r2.audit.pair_audit import run, parse_log_fields


def test_counts_match_spec():
    df = run(write=False)
    assert len(df) == 500
    assert Counter(df.visible_status) == {"excerpt_differs": 269, "identical": 167, "metadata_only": 64}
    assert Counter(df.raw_status) == {"visible": 333, "hidden_only": 110, "no_edit": 57}
    hidden = Counter(df[df.raw_status == "hidden_only"].sham_type)
    assert hidden == {"contraindication_violation": 70, "missing_warning": 40}
    none = Counter(df[df.raw_status == "no_edit"].sham_type)
    assert none == {"authority_mimicry": 26, "subtle_inversion": 31}


def test_case022():
    df = run(write=False).set_index("case_id")
    row = df.loc["CASE_022"]
    assert row.raw_status == "hidden_only"
    assert row.visible_status == "identical"
    assert set(row.hidden_fields_changed.split(";")) == {"safety_guardrails", "exclusions_or_cautions"}


def test_authentic_excerpts_are_domain_level():
    df = run(write=False)
    assert df.t_excerpt_shared_by.max() > 1
    assert df.drop_duplicates("t_excerpt_hash").shape[0] == 69


def test_agrees_with_hash_verified_reconstruction():
    from r2.audit.original_inputs import build_all
    identical_recon = {r["case_id"] for r in build_all(write=False) if r["identical_except_label"]}
    df = run(write=False)
    assert set(df[df.visible_status == "identical"].case_id) == identical_recon


def test_parse_log_fields():
    # Test parsing of 'where_in_tool' field
    assert parse_log_fields("") == []
    assert parse_log_fields("excerpt_text") == ["excerpt_text"]
    assert parse_log_fields("excerpt_text, safety_guardrails (removed allergy bullet)") == ["excerpt_text", "safety_guardrails"]
    assert parse_log_fields("metadata.publication_year, metadata.version_label") == ["metadata.publication_year", "metadata.version_label"]
    assert parse_log_fields("safety_guardrails (removed allergy bullet), exclusions_or_cautions") == ["exclusions_or_cautions", "safety_guardrails"]
    # Deduplicated and sorted
    assert parse_log_fields("excerpt_text, excerpt_text") == ["excerpt_text"]


def test_reconciliation():
    df = run(write=False)
    assert "log_matches_observed" not in df.columns
    assert set(df.claim_status) <= {"no_claim", "realized", "not_realized", "partially_realized"}
    assert set(df.unclaimed_changes) <= {"none", "curation_only", "content"}

    # no_edit rows with a logged claim: nothing was realized
    no_edit = df[df.raw_status == "no_edit"]
    claimed = no_edit[no_edit.log_fields_claimed.fillna("") != ""]
    assert (claimed.claim_status == "not_realized").all() and (claimed.claimed_not_observed != "").all()


def test_unedited_subtle_inversions_have_no_claim_and_no_change():
    df = run(write=False)
    rows = df[(df.sham_type == "subtle_inversion") & (df.raw_status == "no_edit")]
    assert len(rows) == 31
    assert (rows.claim_status == "no_claim").all() and (rows.unclaimed_changes == "none").all()


def test_outdated_version_notes_only_is_curation_only():
    df = run(write=False)
    rows = df[(df.sham_type == "outdated_version") & (df.observed_not_claimed.fillna("") == "metadata.notes")]
    assert len(rows) > 0
    assert (rows.unclaimed_changes == "curation_only").all()


def test_claim_status_case001_and_case022():
    df = run(write=False).set_index("case_id")
    assert df.loc["CASE_001", "claim_status"] == "partially_realized"
    assert df.loc["CASE_022", "claim_status"] == "realized"
    assert df.loc["CASE_022", "unclaimed_changes"] == "none"


def test_summary_counts_by_type(tmp_path, monkeypatch):
    import json

    from r2.audit import pair_audit
    monkeypatch.setattr(pair_audit, "AUDIT_DIR", tmp_path)
    df = pair_audit.run(write=True)
    summary = json.loads((tmp_path / "pair_audit_summary.json").read_text())
    assert "log_matches_observed" not in summary
    for col in ("claim_status", "unclaimed_changes"):
        by_type = summary[f"{col}_by_type"]
        assert sum(sum(v.values()) for v in by_type.values()) == 500
        assert by_type == {t: dict(Counter(g[col])) for t, g in df.groupby("sham_type")}
