"""Exclusion-only reanalysis (2026-10-18): the sensitivity without fabricated_citation, the round-1 physician
summary, Figures 2-4 and the Table 1 examples from the original evaluations of the discriminable pairs."""
import csv
import hashlib
import json

import pandas as pd
import pytest
from PIL import Image

import r2.analysis.original_corrected as oc
from r2.analysis import figures_original as fo
from r2.paths import AUDIT_DIR, REPO, VERIFY_DIR
from r2.verify import round1_summary as rs
from r2.verify.score_review import QUESTIONS

NOFC = "discriminable_excluding_fabricated_citation"
MEASURES = ("accuracy", "primacy", "acc_sham_first", "acc_sham_second", "position_averaged_accuracy")
# analysis_original/headline.json before these additions (commit 6554260), as written (json.dumps(head, indent=2))
_HEADLINE_BEFORE_X1_SHA256 = "f48b641db698d17886afcb8d250129175c8d3631bc1a20a0ea139ff9ce9e0fe3"


@pytest.fixture(scope="module")
def head():
    return oc.run(write=False)


@pytest.fixture(scope="module")
def frame():
    return oc.original_frame()


# ---------------------------------------------------------------- sensitivity without fabricated_citation
def test_sensitivity_n_excludes_the_840_fabricated_citation_evaluations(head):
    b = head[NOFC]
    assert b["n"] == 6993 - 840 == 6153
    assert b["n_excluded"] == head[oc.CUE_KEY]["n"] == 840
    assert b["n_pairs"] == 333 - 40 and b["excluded_sham_type"] == "fabricated_citation"
    assert b["acc_lo"] < b["accuracy"] < b["acc_hi"]
    for m in MEASURES[1:]:
        assert b[f"{m}_lo"] < b[m] < b[f"{m}_hi"], m


def test_sensitivity_also_without_coerced_records(head):
    cc = pd.read_csv(AUDIT_DIR / "coerced_choices.csv")
    audit = pd.read_csv(AUDIT_DIR / "pair_audit.csv").set_index("case_id")
    cc = cc[cc["class"] != "not_coerced"]
    keep = cc.case_id.map(audit.raw_status).eq("visible") & cc.case_id.map(audit.sham_type).ne("fabricated_citation")
    b = head[NOFC]["excluding_coerced"]
    assert b["n_coerced_excluded"] == int(keep.sum())
    assert b["n"] == head[NOFC]["n"] - b["n_coerced_excluded"]
    for m in MEASURES:
        assert b[f"{m}_lo"] < b[m] < b[f"{m}_hi"], m


def test_every_earlier_headline_key_keeps_its_value(head):
    assert list(head)[-1] == NOFC
    old = {k: v for k, v in head.items() if k != NOFC}
    assert hashlib.sha256(json.dumps(old, indent=2).encode()).hexdigest() == _HEADLINE_BEFORE_X1_SHA256


def test_per_model_and_by_profile_carry_case_cluster_bootstrap_cis():
    for name in ("all", "discriminable"):
        pm = pd.read_csv(REPO / "analysis_original" / f"per_model_{name}.csv")
        assert len(pm) == 21
        assert (pm.paa_lo < pm.position_averaged_accuracy).all() and (pm.position_averaged_accuracy < pm.paa_hi).all()
        bp = pd.read_csv(REPO / "analysis_original" / f"by_profile_{name}.csv", index_col=0)
        assert bp.n_models.sum() == 21
        for m in MEASURES:
            assert ((bp[f"{m}_lo"] <= bp[m]) & (bp[m] <= bp[f"{m}_hi"])).all(), (name, m)


# ---------------------------------------------------------------- round-1 physician summary
def _forms(a, b):
    idx = [f"C{i}" for i in range(1, len(a) + 1)]
    return (pd.DataFrame({"q3_authentic_concordant": a, "q8_leakage": a}, index=idx),
            pd.DataFrame({"q3_authentic_concordant": b, "q8_leakage": b}, index=idx))


def test_kappa_matches_a_hand_computed_value_on_three_rows():
    # Q3 over Y/N/P. AG: Y, N, P; MO: Y, P, P. Observed agreement 2/3. Expected: AG (1/3, 1/3, 1/3) and
    # MO (1/3, 0, 2/3) give 1/3 * 1/3 + 1/3 * 0 + 1/3 * 2/3 = 1/3. kappa = (2/3 - 1/3) / (1 - 1/3) = 0.5.
    a, b = _forms(["Y", "N", "P"], ["Y", "P", "P"])
    s = rs.item_summary(a["q3_authentic_concordant"], b["q3_authentic_concordant"], "q3_authentic_concordant")
    assert s["AG_Y"] == s["AG_N"] == s["AG_P"] == 1 and (s["MO_Y"], s["MO_N"], s["MO_P"]) == (1, 0, 2)
    assert s["agreement"] == pytest.approx(2 / 3)
    assert s["kappa"] == pytest.approx(0.5)
    assert s["concern_answer"] == "N" and s["both_concern"] == 0


def test_q8_counts_pairs_both_rated_y():
    # AG: Y, Y, N; MO: Y, N, N. po = 2/3, pe = 2/3 * 1/3 + 1/3 * 2/3 = 4/9, kappa = (2/9) / (5/9) = 0.4.
    a, b = _forms(["Y", "Y", "N"], ["Y", "N", "N"])
    s = rs.item_summary(a["q8_leakage"], b["q8_leakage"], "q8_leakage")
    assert s["kappa"] == pytest.approx(0.4)
    assert s["concern_answer"] == "Y" and s["both_concern"] == 1
    assert "AG_P" not in s


def test_summary_files_describe_both_subsets():
    js = json.loads((VERIFY_DIR / "round1_summary.json").read_text(encoding="utf-8"))
    assert js["subsets"]["all_forms"]["n_pairs"] == 477
    assert js["subsets"]["analyzed_pairs"]["n_pairs"] == 333
    assert js["excludes_nothing"] is True
    fc = js["fabricated_citation"]
    assert fc["n_rows_all_forms"] == 40 and fc["n_rows_analyzed_pairs"] == 40
    assert fc["rows_with_neutral_doi"] == 40 and fc["rows_with_fake_token"] == 0
    rows = list(csv.DictReader(open(VERIFY_DIR / "round1_summary.csv", newline="", encoding="utf-8")))
    assert len(rows) == 2 * len(QUESTIONS)
    assert {r["subset"] for r in rows} == {"all_forms", "analyzed_pairs"}


def test_summary_is_recomputed_from_the_imported_forms():
    """The committed summary equals a fresh summary of the committed round-1 forms (no re-import)."""
    js = json.loads((VERIFY_DIR / "round1_summary.json").read_text(encoding="utf-8"))
    fresh = rs.summarize(*rs.load_forms())
    assert json.loads(json.dumps(fresh)) == {k: js[k] for k in fresh}


def test_the_analyzed_pairs_are_the_333_visible_pairs_and_all_were_reviewed():
    ids = rs.analyzed_case_ids()
    audit = pd.read_csv(AUDIT_DIR / "pair_audit.csv")
    assert ids == sorted(audit.case_id[audit.raw_status == "visible"]) and len(ids) == 333
    a, _ = rs.load_forms()
    assert set(ids) <= set(a.index) and len(a) == 477


# ---------------------------------------------------------------- Figures 2-4
@pytest.fixture(scope="module")
def made(tmp_path_factory):
    out = tmp_path_factory.mktemp("figs_original")
    return out, fo.run(out_dir=out)


def test_figure_files_exist_after_run(made):
    out, paths = made
    for name in ("Figure2", "Figure3", "Figure4"):
        for ext in ("png", "pdf"):
            p = out / f"{name}.{ext}"
            assert p.exists() and p.stat().st_size > 10_000, p
            assert paths[name][ext] == p
        assert (out / f"legend_{name}.md").exists()
        assert round(Image.open(out / f"{name}.png").info["dpi"][0]) == 300
    assert (out / "Figure2.pdf").read_bytes()[:4] == b"%PDF"


def test_legends_state_n_and_the_exclusion(made, head):
    out, _ = made
    for name in ("Figure2", "Figure3", "Figure4"):
        text = (out / f"legend_{name}.md").read_text(encoding="utf-8")
        assert text.startswith(f"**Figure {name[-1]} | ")
        assert f"n = {head['discriminable']['n']:,}" in text and "333 pairs" in text and "21 models" in text
        assert f"{head['identical']['n']:,} evaluations" in text and "167 pairs" in text
        assert "bootstrap" in text and "2,000" in text
        assert "discriminab" not in text.lower()  # house wording (M15): "non-identical"
    fig3 = (out / "legend_Figure3.md").read_text(encoding="utf-8")
    assert "Contraindication violation and missing warning" in fig3 and "cue-confounded" in fig3


def test_rebuild_is_byte_identical(made, tmp_path):
    out, _ = made
    fo.run(out_dir=tmp_path)
    for p in sorted(out.iterdir()):
        assert (tmp_path / p.name).read_bytes() == p.read_bytes(), p.name


def test_committed_figures_are_current(made):
    out, _ = made
    committed = REPO / "analysis_original" / "figures"
    assert sorted(p.name for p in committed.iterdir()) == sorted(p.name for p in out.iterdir())
    for p in out.iterdir():
        assert (committed / p.name).read_bytes() == p.read_bytes(), p.name


# ---------------------------------------------------------------- Table 1 examples
def test_table1_rule_is_deterministic(frame):
    rows = oc.table1_examples(frame)
    assert rows == oc.table1_examples(frame)
    types = sorted(frame[frame.raw_status == "visible"].sham_type.unique())
    assert len(types) == 8 and [(r["sham_type"], r["outcome"]) for r in rows] == \
        [(t, o) for t in types for o in ("Failure", "Success")]
    pool = frame[(frame.raw_status == "visible") & ~frame.coerced]
    for r in rows:
        hit = pool[(pool.sham_type == r["sham_type"]) & (pool.correct == (r["outcome"] == "Success"))]
        first = min(zip(hit.case_id, hit.model))
        assert (r["case_id"], r["model"]) == first, r["sham_type"]
        rec = frame[(frame.case_id == r["case_id"]) & (frame.model == r["model"])].iloc[0]
        assert not rec.coerced and rec.raw_status == "visible"
        assert len(r["rationale"]) <= oc.TABLE1_CHARS + 1 and r["rationale"].rstrip("…") in rec.rationale
        assert r["cue_confounded"] == (r["sham_type"] == "fabricated_citation")


def test_committed_table1_examples_are_current(frame):
    committed = pd.read_csv(REPO / "analysis_original" / "table1_examples.csv", dtype=str, keep_default_na=False)
    fresh = pd.DataFrame(oc.table1_examples(frame)).astype(str)
    assert committed.to_dict("records") == fresh.to_dict("records")
