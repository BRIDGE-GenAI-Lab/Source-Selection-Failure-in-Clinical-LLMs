from collections import Counter

from r2.audit.pair_audit import run


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
