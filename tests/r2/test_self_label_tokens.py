"""Final fix F1: the self-labelling token "fake" in the fabricated-citation edits.

The corrected set must carry no such token (test_build_v2.py); the regenerated round-1 forms must show the
corrected text and leave every other packet unchanged; the audit must disclose the token in the original
model inputs."""
import csv
import hashlib
import json

import pytest

from r2.paths import AUDIT_DIR, DATA_V2, VERIFY_DIR
from r2.stimuli.transforms import self_label_tokens
from tests.r2.test_pipeline import sandboxed  # noqa: F401  (a fixture)

FC = "fabricated_citation"
ROUND1 = VERIFY_DIR / "round1"
# Pinned from verification/round1/packets_manifest.json at 1a9831f (before F1): digest = sha256 of the
# sorted "case_id:packet_sha256" lines.
HEAD_437_PACKETS = "01b794082bb95f37e79555aad75c41cf29028022c7d8f5032751d43f9d95fc0c"
HEAD_40_FC_PACKETS = "8fa2329f199e314818d00e3cf072e12fb975d7194c92f82d7b9efb6306f73a5c"
PROFILES = {"openai_fc": 4, "gemini_inline7": 1, "deepseek_chat_inline7": 1, "deepseek_reasoner_inline7": 1,
            "together_inline6": 14}


def _digest(pairs):
    return hashlib.sha256("\n".join(f"{a}:{b}" for a, b in sorted(pairs)).encode("utf-8")).hexdigest()


def _pairs():
    return json.loads((DATA_V2 / "pairs_v2.json").read_text(encoding="utf-8"))


def _fc_ids():
    return {c["case_id"] for c in _pairs()["cases"] if c["eval"]["sham_trap_type"] == FC}


# ---------------------------------------------------------------- round-1 forms


def check_manifest_against_pairs(m, pairs):
    """Every round-1 packet hash equals the packet recomputed from the CURRENT pairs for that case. This holds
    forever: the physicians' edits touch only the 23 unresolved cases, which are not in round 1."""
    from r2.verify.build_packets import packet_rows
    now = {r["case_id"]: r["packet_sha256"] for r in packet_rows(pairs["cases"])}
    assert len(m["packets"]) == 477 and len(m["unresolved"]) == 23
    assert set(m["packets"]) <= set(now)
    assert {cid for cid, h in m["packets"].items() if now[cid] != h} == set()
    assert set(m["unresolved"]) <= set(now) | set(pairs["unresolved"])


def check_round1_render(tmp_dir, m, pairs, raw):
    """Regenerate the round-1 forms in tmp from the round-1 case subset and compare them byte for byte."""
    import difflib

    from r2.verify import build_packets as bp
    saved = difflib.HtmlDiff._default_prefix
    difflib.HtmlDiff._default_prefix = 0  # the committed file was written by a fresh process
    subset = [c for c in pairs["cases"] if c["case_id"] in m["packets"]]
    assert len(subset) == len(m["packets"])
    try:
        bp._write_forms_to(tmp_dir, raw, subset, {"unresolved": list(m["unresolved"])})
    finally:
        difflib.HtmlDiff._default_prefix = saved
    for name in ("review_AG.csv", "review_MO.csv", "packets.html", "README_REVIEWERS.md"):
        assert (tmp_dir / name).read_bytes() == (ROUND1 / name).read_bytes(), name
    # the manifest differs only in the whole-file hash, which moves when pairs_v2.json gains the new pairs
    fresh = json.loads((tmp_dir / "packets_manifest.json").read_text(encoding="utf-8"))
    fresh.pop("pairs_v2_sha256"), m.pop("pairs_v2_sha256", None)
    assert fresh == m


def test_round1_manifest_matches_the_current_pairs():
    m = json.loads((ROUND1 / "packets_manifest.json").read_text(encoding="utf-8"))
    check_manifest_against_pairs(m, _pairs())


def test_round1_keeps_437_packets_and_changes_exactly_the_40_fabricated_citation_packets():
    from r2.verify.build_packets import packet_rows
    from r2.stimuli.build_v2 import before_self_label_fix
    packets = json.loads((ROUND1 / "packets_manifest.json").read_text(encoding="utf-8"))["packets"]
    fc = _fc_ids()
    assert len(fc) == 40
    assert _digest([(cid, h) for cid, h in packets.items() if cid not in fc]) == HEAD_437_PACKETS
    # the packets of the 40 pairs as they were before F1, recomputed from the reverted cases
    before = {r["case_id"]: r["packet_sha256"]
              for r in packet_rows([before_self_label_fix(c) for c in _pairs()["cases"] if c["case_id"] in fc])}
    assert _digest(before.items()) == HEAD_40_FC_PACKETS
    assert all(packets[cid] != before[cid] for cid in fc)


@pytest.mark.parametrize("reviewer", ["AG", "MO"])
def test_round1_csv_shows_no_self_label_token(reviewer):
    with open(ROUND1 / f"review_{reviewer}.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 477
    for r in rows:
        shown = json.loads(r["modified_as_shown_to_model"])
        text = "\n".join(str(x) for v in shown.values() for x in (v if isinstance(v, list) else [v]))
        assert self_label_tokens(text) == [], r["case_id"]
        if r["sham_type"] == FC:
            assert "10.9999/cg.2025." in r["modified_as_shown_to_model"]
            assert "10.9999/fake." not in r["what_changed"]


@pytest.mark.parametrize("reviewer", ["AG", "MO"])
def test_round1_xlsx_matches_the_current_render(reviewer):
    from openpyxl import load_workbook

    from r2.verify.build_packets import REVIEW, _cell_text, _check_review_content
    path = ROUND1 / f"review_{reviewer}.xlsx"
    wb = load_workbook(path, read_only=True, data_only=True)
    rows = [[_cell_text(c) for c in row] for row in wb[REVIEW].iter_rows(values_only=True)]
    while rows and not any(rows[-1]):
        rows.pop()
    _check_review_content(rows, path)  # raises if any content cell differs from a fresh render
    assert len(rows) == 478


def test_round1_files_equal_a_fresh_render(tmp_path):
    """The committed round-1 CSV, HTML, manifest and README are exactly what build_packets writes for the round-1
    cases (HtmlDiff wraps long lines, so the HTML is compared whole, not searched for a DOI)."""
    raw = (DATA_V2 / "pairs_v2.json").read_bytes()
    m = json.loads((ROUND1 / "packets_manifest.json").read_text(encoding="utf-8"))
    check_round1_render(tmp_path, m, json.loads(raw), raw)
    assert "fake" not in (ROUND1 / "packets.html").read_text(encoding="utf-8").lower()


def _queued_edits():
    """A valid physician edit for each of the 23 queued cases (whole-field find, so it occurs exactly once)."""
    from r2.io import load_original_cases
    from r2.stimuli.build_v2 import build_v2
    _, queued = build_v2(strict=False)
    originals, _ = load_original_cases()
    edits = {}
    for cid in queued:
        t = originals[cid]["tools"]["T"]
        if originals[cid]["eval"]["sham_trap_type"] == "authority_mimicry":
            edits[cid] = {"field": "metadata.organization", "find": t["metadata"]["organization"],
                          "replace": t["metadata"]["organization"] + " Group"}
        else:
            edits[cid] = {"field": "excerpt_text", "find": t["excerpt_text"],
                          "replace": t["excerpt_text"] + " Avoid the usual dose."}
    return edits


def test_the_three_checks_hold_once_the_physicians_edits_exist_and_still_catch_a_round1_change(sandboxed):
    """Synthetic world: a manual_edits.json for the 23 queued cases in the sandbox's data/v2 (never the real one)."""
    import copy

    from r2 import paths
    from tests.r2.test_build_v2 import check_committed_pairs, committed_manual_edits
    from r2.stimuli.build_v2 import build_v2
    edits = _queued_edits()
    assert len(edits) == 23 and paths.DATA_V2 != DATA_V2
    paths.DATA_V2.mkdir(parents=True)
    (paths.DATA_V2 / "manual_edits.json").write_text(json.dumps(edits))
    manual = committed_manual_edits(paths.DATA_V2)
    cases, unresolved = build_v2(manual, strict=True)
    pairs = {"derived_from_sha256": "synthetic", "n_pairs": 500, "unresolved": unresolved, "cases": cases}
    raw = json.dumps(pairs, indent=1).encode("utf-8")
    (paths.DATA_V2 / "pairs_v2.json").write_bytes(raw)
    assert len(cases) == 500 and unresolved == []
    m = json.loads((ROUND1 / "packets_manifest.json").read_text(encoding="utf-8"))
    assert m["pairs_v2_sha256"] != hashlib.sha256(raw).hexdigest()   # the whole-file binding does move

    # 1. test_build_v2: the committed pairs equal the build with the committed edits
    check_committed_pairs(pairs, manual)
    with pytest.raises(AssertionError):
        check_committed_pairs(pairs, {})                             # a build without the edits would differ
    # 2. the manifest packets equal the packets recomputed from the current pairs
    check_manifest_against_pairs(copy.deepcopy(m), pairs)
    # 3. the round-1 forms stay byte-reproducible from the round-1 subset
    out = sandboxed / "render"
    check_round1_render(out, copy.deepcopy(m), pairs, raw)

    # an accidental change to any round-1 pair is still caught by all three
    victim = next(c for c in cases if c["case_id"] in m["packets"] and c["eval"]["sham_trap_type"] != FC)
    altered = copy.deepcopy(pairs)
    tool = next(c for c in altered["cases"] if c["case_id"] == victim["case_id"])["tools"]["S"]
    tool["excerpt_text"] += " Altered."
    with pytest.raises(AssertionError):
        check_committed_pairs(altered, manual)
    with pytest.raises(AssertionError):
        check_manifest_against_pairs(copy.deepcopy(m), altered)
    with pytest.raises(AssertionError):
        check_round1_render(sandboxed / "render2", copy.deepcopy(m), altered,
                            json.dumps(altered, indent=1).encode("utf-8"))


# ---------------------------------------------------------------- audit disclosure


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
