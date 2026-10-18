import copy

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
        if c["eval"]["sham_trap_type"] == "fabricated_citation":
            continue  # only the DOI suffix changes: test_fabricated_citation_dois_carry_the_neutral_suffix
        if audit.loc[c["case_id"], "raw_status"] != "no_edit":
            expected = copy.deepcopy(original[c["case_id"]]["tools"]["S"])
            for fix in c["edit_record"].get("cue_fixes", []):  # cue fix A2: the year only (test_cue_fixes.py)
                assert fix["rule"] == "outdated_year" and fix["before"] == expected["metadata"]["publication_year"]
                expected["metadata"]["publication_year"] = fix["after"]
            assert c["tools"]["S"] == expected
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


# ---- Final fix F1: the fabricated-citation DOI no longer labels itself "fake" ----
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402

from r2.paths import DATA_V2, REPO  # noqa: E402
from r2.stimuli.build_v2 import before_self_label_fix, self_labels_in  # noqa: E402
from r2.stimuli.render import SYSTEM_PROMPT_V2, USER_TEMPLATE_V2  # noqa: E402
from r2.stimuli.transforms import neutral_doi_suffix, self_label_tokens  # noqa: E402

FC = "fabricated_citation"
# Pinned from data/v2/pairs_v2.json at 1a9831f, the commit before F1. Case hash = sha256 of
# json.dumps(case, sort_keys=True, ensure_ascii=False); digest = sha256 of the sorted "case_id:hash" lines.
HEAD_COMMIT = "1a9831f"
HEAD_437_UNCHANGED = "1cb37a397de3c0c21dfc07c99366e928dcf969352f482983b5d7766ba8c8fefb"
# The same hash over the 432 of those 437 that the cue fixes (A1-A3) leave alone; the other five are CUE_FIVE.
# Pinned from 30c5805, the commit before the cue fixes (identical to the build now).
HEAD_432_UNCHANGED = "b1e3e0a911c9121992975bc81d7b76dacdcbc59975bdf338c3e7c944116f1c04"
CUE_FIVE = ("CASE_036", "CASE_337", "CASE_387", "CASE_475", "CASE_488")
HEAD_40_FABRICATED = "ef8bf4d1e8aba204a5ba92555c0aae5cae967aa0d22ad8bcd9466c2434a07194"


def _case_sha(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _digest(pairs):
    return hashlib.sha256("\n".join(f"{a}:{b}" for a, b in sorted(pairs)).encode("utf-8")).hexdigest()


def _assert_only_fc_changed(cases):
    # the 23 physician-authored pairs (once manual_edits.json exists) are new, not round-1 pairs
    by_id = {c["case_id"]: c for c in cases if c["edit_record"]["edit_source"] != "physician_manual"}
    unchanged = [(cid, _case_sha(c)) for cid, c in by_id.items()
                 if c["eval"]["sham_trap_type"] != FC and cid not in CUE_FIVE]
    assert len(unchanged) == 432 and _digest(unchanged) == HEAD_432_UNCHANGED
    assert sorted(cid for cid in by_id if cid in CUE_FIVE) == sorted(CUE_FIVE)
    fc = {cid: c for cid, c in by_id.items() if c["eval"]["sham_trap_type"] == FC}
    reverted = {cid: before_self_label_fix(c) for cid, c in fc.items()}
    assert len(fc) == 40 and all(_case_sha(fc[cid]) != _case_sha(reverted[cid]) for cid in fc)
    assert _digest([(cid, _case_sha(c)) for cid, c in reverted.items()]) == HEAD_40_FABRICATED


def test_fabricated_citation_dois_carry_the_neutral_suffix(built):
    """F1 changes only the DOI suffix; the cue fix A1 then rewrites the Evidence line (test_cue_fixes.py)."""
    original, _ = load_original_cases()
    fc = [c for c in built[0] if c["eval"]["sham_trap_type"] == FC]
    assert len(fc) == 40
    for c in fc:
        cid, s0 = c["case_id"], original[c["case_id"]]["tools"]["S"]
        before = re.search(r"10\.9999/fake\.\d{4}", s0["excerpt_text"]).group(0)
        after = "10.9999/" + neutral_doi_suffix(cid, "2025")
        assert re.fullmatch(r"10\.9999/cg\.2025\.\d{4}", after)
        assert c["tools"]["S"]["excerpt_text"].endswith(f"(DOI:{after})")
        assert {**c["tools"]["S"], "excerpt_text": None} == {**s0, "excerpt_text": None}, cid
        assert c["tools"]["T"] == original[cid]["tools"]["T"]
        assert c["edit_record"]["edit_source"] == "original_transform"
        assert c["edit_record"]["self_label_fix"] == {"rule": "neutral_doi_suffix", "field": "excerpt_text",
                                                      "before": before, "after": after}
        # before_self_label_fix undoes the cue fix and then F1: the original generator's output, DOI included
        assert before_self_label_fix(c)["tools"]["S"] == s0, cid
    assert not any("self_label_fix" in c["edit_record"] for c in built[0] if c["eval"]["sham_trap_type"] != FC)


def test_build_is_deterministic(built):
    again, unresolved = build_v2(strict=False)
    assert again == built[0] and unresolved == built[1]


def test_post_condition_holds_on_all_500(built):
    original, _ = load_original_cases()
    by_id = {c["case_id"]: c for c in built[0]}
    assert len(by_id) + len(built[1]) == 500 and not set(by_id) & set(built[1])
    for cid in sorted(original):
        case = by_id.get(cid, original[cid])  # an unresolved pair is checked as the original holds it
        assert self_labels_in(case["tools"]["S"]) == [], cid
        assert self_labels_in(case["tools"]["T"]) == [], cid
    assert self_label_tokens(SYSTEM_PROMPT_V2 + "\n" + USER_TEMPLATE_V2) == []


def test_self_labels_in_reads_every_visible_field():
    tool = {"metadata": {"organization": "Org", "guideline_title": "Dummy title"}, "excerpt_text": "x",
            "safety_guardrails": ["a", "not real"], "applicability": [], "exclusions_or_cautions": []}
    assert self_labels_in(tool) == ["dummy", "not real"]
    assert self_labels_in({"metadata": {"notes": "fake"}, "excerpt_text": "x"}) == []  # notes are never shown


def test_post_condition_refuses_a_self_labelling_edit():
    edit = {"field": "metadata.organization", "find": "ENT UK", "replace": "ENT UK (placeholder)"}
    with pytest.raises(AssertionError, match="self-labelling"):
        build_v2({"CASE_399": edit}, strict=False)


def test_only_the_cue_fix_pairs_change_against_head(built):
    _assert_only_fc_changed(built[0])


def committed_manual_edits(data_dir=None):
    """The physicians' edits when data/v2/manual_edits.json exists (after import-edits), else none."""
    path = (DATA_V2 if data_dir is None else data_dir) / "manual_edits.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def check_committed_pairs(committed, manual):
    """pairs_v2.json is exactly the build with the committed manual edits (none before they arrive)."""
    cases, unresolved = build_v2(manual, strict=False)
    assert committed["cases"] == cases and committed["unresolved"] == unresolved
    return cases


def test_committed_pairs_match_head_except_the_cue_fix_pairs():
    committed = json.loads((DATA_V2 / "pairs_v2.json").read_text(encoding="utf-8"))
    cases = check_committed_pairs(committed, committed_manual_edits())
    _assert_only_fc_changed(cases)


def test_changed_pairs_against_the_head_commit_by_git(built):
    out = subprocess.run(["git", "show", f"{HEAD_COMMIT}:data/v2/pairs_v2.json"], cwd=REPO, capture_output=True)
    if out.returncode != 0:
        pytest.skip(f"commit {HEAD_COMMIT} not available (no git history)")
    head = {c["case_id"]: c for c in json.loads(out.stdout)["cases"]}
    now = {c["case_id"]: c for c in built[0]}
    assert set(head) == set(now)
    changed = sorted(cid for cid in now if _case_sha(now[cid]) != _case_sha(head[cid]))
    fc = sorted(cid for cid, c in now.items() if c["eval"]["sham_trap_type"] == FC)
    assert changed == sorted(fc + list(CUE_FIVE))
    assert len(changed) == 45
    assert all(before_self_label_fix(now[cid]) == head[cid] for cid in fc)
