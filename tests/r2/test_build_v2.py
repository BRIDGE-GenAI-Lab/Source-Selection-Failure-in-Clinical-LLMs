import pytest

from r2.io import load_original_cases
from r2.stimuli.build_v2 import UnresolvedEdits, build_v2
from r2.stimuli.render import visible_content


@pytest.fixture(scope="module")
def built():
    return build_v2(strict=False)


def test_no_built_pair_is_visibly_identical(built):
    cases, unresolved = built
    for c in cases:
        assert visible_content(c["tools"]["T"]) != visible_content(c["tools"]["S"]), c["case_id"]
    assert len(cases) + len(unresolved) == 500


def test_unresolved_are_only_unedited_pairs(built):
    import pandas as pd
    from r2.paths import AUDIT_DIR
    audit = pd.read_csv(AUDIT_DIR / "pair_audit.csv").set_index("case_id")
    for cid in built[1]:
        assert audit.loc[cid, "raw_status"] == "no_edit"
        assert audit.loc[cid, "sham_type"] in {"subtle_inversion", "authority_mimicry"}


def test_previously_edited_pairs_are_untouched(built):
    import pandas as pd
    from r2.paths import AUDIT_DIR
    original, _ = load_original_cases()
    audit = pd.read_csv(AUDIT_DIR / "pair_audit.csv").set_index("case_id")
    for c in built[0]:
        if audit.loc[c["case_id"], "raw_status"] != "no_edit":
            assert c["tools"]["S"] == original[c["case_id"]]["tools"]["S"]
            assert c["tools"]["T"] == original[c["case_id"]]["tools"]["T"]
            assert c["edit_record"]["edit_source"] == "original_transform"


def test_strict_build_refuses_unresolved(built):
    if built[1]:
        with pytest.raises(UnresolvedEdits):
            build_v2(strict=True)


def test_short_org_is_queued_not_crashed(built):
    assert "CASE_399" in built[1]


# ---- Task DM item 6 ----
def test_fields_changed_pinned_per_rule_source(built):
    sources = {}
    for c in built[0]:
        rec = c["edit_record"]
        sources.setdefault(rec["edit_source"], set()).add(tuple(rec["fields_changed"]))
        assert rec["fields_changed"] == [ch["field"] for ch in rec["changes"]]
    assert sources["rule_typosquat"] == {("organization",)}
    assert sources["rule_inversion_extended"] == {("excerpt_text",)}
    assert {"rule_typosquat", "rule_inversion_extended", "original_transform"} <= set(sources)


def test_rule_inversions_keep_the_initial_capital(built):
    for c in built[0]:
        rec = c["edit_record"]
        if rec["edit_source"] != "rule_inversion_extended":
            continue
        ch = rec["changes"][0]
        diff_at = next(i for i, (x, y) in enumerate(zip(ch["before"], ch["after"])) if x != y)
        if ch["before"][diff_at:diff_at + 1].isupper():
            assert ch["after"][diff_at:diff_at + 1].isupper(), c["case_id"]


def _edited_case_id():
    import pandas as pd

    from r2.paths import AUDIT_DIR
    audit = pd.read_csv(AUDIT_DIR / "pair_audit.csv")
    return audit[audit.raw_status == "visible"].case_id.iloc[0]


def test_manual_edit_for_an_already_edited_pair_needs_override(built):
    cid = _edited_case_id()
    original, _ = load_original_cases()
    s = original[cid]["tools"]["S"]
    text = s["excerpt_text"]
    word = next(w for w in text.split() if w.isalpha() and text.count(w) == 1)
    edit = {"field": "excerpt_text", "find": word, "replace": word + "x"}
    with pytest.raises(ValueError, match="override"):
        build_v2({cid: edit}, strict=False)
    cases, _ = build_v2({cid: {**edit, "override": True}}, strict=False)
    got = next(c for c in cases if c["case_id"] == cid)
    assert got["edit_record"]["edit_source"] == "physician_manual"


def _tiny(tmp_path, monkeypatch):
    import r2.stimuli.build_v2 as b
    monkeypatch.setattr(b, "DATA_V2", tmp_path)
    return b


def test_stale_queue_is_removed_when_nothing_is_unresolved(tmp_path, monkeypatch):
    b = _tiny(tmp_path, monkeypatch)
    (tmp_path / "manual_edit_queue.csv").write_text("case_id\nCASE_001\n", encoding="utf-8")
    b.write_outputs([], [])
    assert not (tmp_path / "manual_edit_queue.csv").exists()
    assert (tmp_path / "pairs_v2.json").exists()


def test_queue_is_rewritten_when_unresolved_remain(tmp_path, monkeypatch):
    b = _tiny(tmp_path, monkeypatch)
    originals, _ = load_original_cases()
    b.write_outputs([], ["CASE_399"], originals)
    text = (tmp_path / "manual_edit_queue.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("case_id,") and "CASE_399" in text


def test_pairs_file_is_utf8_with_non_ascii(tmp_path, monkeypatch):
    b = _tiny(tmp_path, monkeypatch)
    b.write_outputs([{"case_id": "C", "x": "≥ 5 µg"}], [])
    assert "≥ 5 µg" in (tmp_path / "pairs_v2.json").read_bytes().decode("utf-8")
