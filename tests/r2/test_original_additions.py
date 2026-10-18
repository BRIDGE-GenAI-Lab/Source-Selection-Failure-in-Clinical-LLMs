"""Exclusion-only reanalysis (2026-10-18): the sensitivity without fabricated_citation, the round-1 physician
summary, Figures 2-4 and the Table 1 examples from the original evaluations of the discriminable pairs."""
import copy
import csv
import hashlib
import io
import json

import pandas as pd
import pytest
from PIL import Image

import r2.analysis.original_corrected as oc
from r2.analysis import confidence_original as co
from r2.analysis import figures_original as fo
from r2.paths import AUDIT_DIR, REPO, VERIFY_DIR
from r2.verify import round1_summary as rs
from r2.verify.score_review import QUESTIONS

NOFC = "discriminable_excluding_fabricated_citation"
MEASURES = ("accuracy", "primacy", "acc_sham_first", "acc_sham_second", "position_averaged_accuracy")
# analysis_original/headline.json before these additions (commit 6554260), as written (json.dumps(head, indent=2)),
# without the fabricated_citation note (reworded in fix round 1: no repeated experiment was run). Every number pinned.
_HEADLINE_BEFORE_X1_SHA256 = "ec38d83434cbea5a2cfffdc6d3c91a1a61bf9880861dd4daa26adc0457e4ae02"
# analysis_original/headline.json before the confidence key (Task 1b; commit 29dd858), as written
_HEADLINE_BEFORE_X1B_SHA256 = "128c23b165b6a87d98e2dff6d36524a9fccb2e2b18231ce0e5470b8cf9d069f8"


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
    assert list(head)[-2:] == [NOFC, "confidence"]
    before_1b = {k: v for k, v in head.items() if k != "confidence"}
    assert hashlib.sha256(json.dumps(before_1b, indent=2).encode()).hexdigest() == _HEADLINE_BEFORE_X1B_SHA256
    old = copy.deepcopy({k: v for k, v in head.items() if k not in (NOFC, "confidence")})
    del old[oc.CUE_KEY]["note"]
    assert hashlib.sha256(json.dumps(old, indent=2).encode()).hexdigest() == _HEADLINE_BEFORE_X1_SHA256


def test_cue_note_names_no_repeated_experiment(head):
    note = head[oc.CUE_KEY]["note"]
    assert "repeated" not in note.lower()
    assert "10.9999/fake." in note and "blank line" in note and "physician review" in note


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


@pytest.fixture
def committed_summary():
    """The committed round-1 summary. A rehearsal clone (tests/r2/test_rehearse.py) replaces verification/ with
    simulated files that have none; the summary describes the real returned forms only."""
    path = VERIFY_DIR / "round1_summary.json"
    if not path.exists():
        pytest.skip(f"{path} absent (verification/ replaced by a rehearsal)")
    return json.loads(path.read_text(encoding="utf-8"))


def test_summary_files_describe_both_subsets(committed_summary):
    js = committed_summary
    assert js["subsets"]["all_forms"]["n_pairs"] == 477
    assert js["subsets"]["analyzed_pairs"]["n_pairs"] == 333
    assert js["excludes_nothing"] is True
    fc = js["fabricated_citation"]
    assert fc["n_rows_all_forms"] == 40 and fc["n_rows_analyzed_pairs"] == 40
    assert fc["rows_with_neutral_doi"] == 40 and fc["rows_with_fake_token"] == 0
    assert "null" in js["kappa"] and "undefined" in js["kappa"]
    q5 = [it for sub in js["subsets"].values() for it in sub["items"] if it["item"] == "q5_edit_visible"]
    assert all(it["kappa"] is None and it["kappa_note"].startswith("undefined") for it in q5)
    assert all(it["kappa_note"] == "" for sub in js["subsets"].values() for it in sub["items"]
               if it["kappa"] is not None)
    rows = list(csv.DictReader(open(VERIFY_DIR / "round1_summary.csv", newline="", encoding="utf-8")))
    assert len(rows) == 2 * len(QUESTIONS)
    assert {r["subset"] for r in rows} == {"all_forms", "analyzed_pairs"}


def test_summary_is_recomputed_from_the_imported_forms(committed_summary):
    """The committed summary equals a fresh summary of the committed round-1 forms (no re-import)."""
    js = committed_summary
    fresh = rs.summarize(*rs.load_forms())
    assert json.loads(json.dumps(fresh)) == {k: js[k] for k in fresh}


def test_the_analyzed_pairs_are_the_333_visible_pairs_and_all_were_reviewed():
    ids = rs.analyzed_case_ids()
    audit = pd.read_csv(AUDIT_DIR / "pair_audit.csv")
    assert ids == sorted(audit.case_id[audit.raw_status == "visible"]) and len(ids) == 333
    # the headline and figures select visible_status != "identical"; both definitions give the same 333 pairs
    assert set(audit.case_id[audit.raw_status == "visible"]) == set(audit.case_id[audit.visible_status != "identical"])
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
    assert rows == oc.table1_examples(frame.sample(frac=1, random_state=0))  # independent of the input order
    types = sorted(frame[frame.raw_status == "visible"].sham_type.unique())
    assert len(types) == 8 and [(r["sham_type"], r["outcome"]) for r in rows] == \
        [(t, o) for t in types for o in ("Failure", "Success")]
    pool = frame[(frame.raw_status == "visible") & ~frame.coerced]
    used = set()
    for r in rows:
        hit = pool[(pool.sham_type == r["sham_type"]) & (pool.correct == (r["outcome"] == "Success"))]
        keys = sorted(zip(hit.case_id, hit.model))
        fresh = [k for k in keys if k[1] not in used]
        assert (r["case_id"], r["model"]) == (fresh or keys)[0], (r["sham_type"], r["outcome"])
        used.add(r["model"])
        rec = frame[(frame.case_id == r["case_id"]) & (frame.model == r["model"])].iloc[0]
        assert not rec.coerced and rec.raw_status == "visible"
        assert len(r["rationale"]) <= oc.TABLE1_CHARS + 1 and r["rationale"].rstrip("…") in rec.rationale
        assert r["cue_confounded"] == (r["sham_type"] == "fabricated_citation")
    assert len({r["model"] for r in rows}) == 16  # the table spans models


def test_table1_rule_is_stated_in_the_module_docstring():
    doc = " ".join(oc.__doc__.split())
    assert "whose model did not already appear in the table" in doc and "first 300 characters" in doc
    assert "ascending order of case identifier and then model identifier" in doc and oc.TABLE1_CHARS == 300


def test_committed_table1_examples_are_current(frame):
    committed = pd.read_csv(REPO / "analysis_original" / "table1_examples.csv", dtype=str, keep_default_na=False)
    fresh = pd.DataFrame(oc.table1_examples(frame)).astype(str)
    assert committed.to_dict("records") == fresh.to_dict("records")


# ---------------------------------------------------------------- Task 1b: stated confidence (R1 definitions)

# R1 Supplementary Table S10 (all 10,500 evaluations, 4 without confidence): model -> (n, accuracy %, mean
# confidence, AUROC, ECE, Brier), as printed (rounded)
R1_S10 = {
    "Qwen/Qwen3-Next-80B-A3B-Thinking": (500, 64.0, 0.64, 0.74, 0.04, 0.18),
    "deepseek-reasoner": (500, 78.2, 0.54, 0.73, 0.25, 0.23),
    "deepseek-chat": (500, 60.6, 0.72, 0.72, 0.12, 0.21),
    "mistralai/Mistral-Small-24B-Instruct-2501": (500, 64.2, 0.68, 0.63, 0.13, 0.24),
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": (500, 69.8, 0.79, 0.63, 0.15, 0.22),
    "gpt-4.1-2025-04-14": (500, 63.8, 0.71, 0.62, 0.11, 0.24),
    "openai/gpt-oss-20b": (500, 69.2, 0.75, 0.61, 0.14, 0.22),
    "openai/gpt-oss-120b": (500, 71.0, 0.75, 0.59, 0.15, 0.21),
    "gpt-5-nano-2025-08-07": (500, 56.4, 0.66, 0.59, 0.12, 0.25),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (500, 52.6, 0.56, 0.57, 0.30, 0.35),
    "Qwen/Qwen3-VL-8B-Instruct": (500, 73.0, 0.79, 0.56, 0.29, 0.30),
    "ServiceNow-AI/Apriel-1.6-15b-Thinker": (500, 71.4, 0.84, 0.56, 0.17, 0.21),
    "gpt-4o-mini-2024-07-18": (500, 52.8, 0.67, 0.55, 0.32, 0.35),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (500, 50.8, 0.49, 0.54, 0.13, 0.28),
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": (496, 49.4, 0.70, 0.54, 0.32, 0.37),
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": (500, 47.0, 0.65, 0.54, 0.31, 0.38),
    "google/gemma-3n-E4B-it": (500, 53.6, 0.83, 0.52, 0.45, 0.46),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": (500, 54.2, 0.68, 0.51, 0.20, 0.33),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (500, 44.0, 0.88, 0.51, 0.47, 0.48),
    "gemini-2.5-flash-lite": (500, 50.6, 0.89, 0.47, 0.46, 0.46),
    "gpt-4.1-nano-2025-04-14": (500, 50.4, 0.87, 0.47, 0.37, 0.39),
}


@pytest.fixture(scope="module")
def conf_frame(frame):
    return co.attach_confidence(frame)


def test_confidence_reproduces_r1_on_all_pairs(head, conf_frame):
    """R1 Results and Table S10: mean confidence 0.72, accuracy 59.4%, +13 pp, ECE 0.22, Brier 0.30, AUROC 0.57,
    11 of 21 models after BH FDR, 40.1% of failures at confidence >= 0.90, N 10,496."""
    a = head["confidence"]["all"]
    assert (a["n"], a["n_missing_confidence"], a["n_models"]) == (10496, 4, 21)
    assert round(a["mean_confidence"], 2) == 0.72 and round(100 * a["accuracy"], 1) == 59.4
    assert round(100 * a["overconfidence"]) == 13
    assert (round(a["ece"], 2), round(a["brier"], 2), round(a["auroc"], 2)) == (0.22, 0.30, 0.57)
    assert a["n_models_higher_confidence_when_correct"] == 11
    assert round(100 * a["high_conf_error_share"], 1) == 40.1
    pm = co.per_model(conf_frame).set_index("model")
    for model, (n, acc, mc, auroc, ece, brier) in R1_S10.items():
        r = pm.loc[model]
        assert (int(r.n), round(100 * r.accuracy, 1), round(r.mean_confidence, 2)) == (n, acc, mc), model
        assert (round(r.auroc, 2), round(r.ece, 2), round(r.brier, 2)) == (auroc, ece, brier), model
    assert round(pm.auroc.min(), 2) == 0.47 and round(pm.auroc.max(), 2) == 0.74


def test_confidence_point_estimates_match_a_direct_computation(conf_frame):
    from scipy import stats as st
    import numpy as np
    sub = conf_frame[conf_frame.visible_status != "identical"]
    r = sub[sub.conf.notna()]
    p = co.pooled(sub, n_boot=10)
    c, y = r.conf.to_numpy(float), r.correct.to_numpy(float)
    b = np.clip(np.digitize(c, np.linspace(0, 1, 11)) - 1, 0, 9)
    gaps = [abs(c[b == k].mean() - y[b == k].mean()) for k in range(10) if (b == k).any()]
    ece = sum((b == k).mean() * abs(c[b == k].mean() - y[b == k].mean()) for k in range(10) if (b == k).any())
    auroc = st.mannwhitneyu(c[y == 1], c[y == 0]).statistic / ((y == 1).sum() * (y == 0).sum())
    assert p["n"] == len(r) == 6993 - 2 and p["n_missing_confidence"] == 2
    assert p["ece"] == pytest.approx(ece) and p["mce"] == pytest.approx(max(gaps))
    assert p["auroc"] == pytest.approx(auroc) and p["brier"] == pytest.approx(((c - y) ** 2).mean())
    assert p["mean_confidence"] == pytest.approx(c.mean()) and p["accuracy"] == pytest.approx(y.mean())
    assert p["high_conf_error_share"] == pytest.approx((c[y == 0] >= 0.9).mean())


def test_confidence_bootstrap_equals_resampling_the_cases(conf_frame):
    """The vectorized case-cluster bootstrap gives the same CIs as recomputing on each resample of whole cases."""
    import numpy as np
    sub = conf_frame[conf_frame.case_id.isin(sorted(conf_frame.case_id.unique())[:30])]
    got = co.pooled(sub, n_boot=40, seed=7)
    rows = sub[sub.conf.notna()]
    cases = np.unique(rows.case_id)
    draws = np.random.default_rng(7).integers(0, len(cases), size=(40, len(cases)))
    stats = {k: [] for k in co.MEASURES}
    for d in draws:
        res = pd.concat([rows[rows.case_id == cases[i]] for i in d])
        levels = np.unique(res.conf.to_numpy(float))
        pos, neg = co._counts(res, levels)
        for k, v in co._from_counts(levels, pos.sum(0), neg.sum(0)).items():
            stats[k].append(float(v))
    for k in co.MEASURES:
        s = np.array(stats[k])
        s = s[np.isfinite(s)]
        assert (got[f"{k}_lo"], got[f"{k}_hi"]) == pytest.approx((np.percentile(s, 2.5), np.percentile(s, 97.5))), k


def test_confidence_block_on_the_analysis_set(head):
    c = head["confidence"]
    d = c["discriminable"]
    assert d["n"] == head["discriminable"]["n"] - d["n_missing_confidence"] and d["n_missing_confidence"] == 2
    for name in ("all", "discriminable"):
        for k in co.MEASURES:
            assert c[name][f"{k}_lo"] <= c[name][k] <= c[name][f"{k}_hi"], (name, k)
        ex = c["excluding_coerced"][name]
        assert ex["n"] == c[name]["n"] - head["excluding_coerced"][name]["n_coerced_excluded"], name
    pm = pd.read_csv(REPO / "analysis_original" / "confidence_discriminable.csv")
    assert len(pm) == 21 and int(pm.n.sum()) == d["n"]
    assert int(pm.higher_confidence_when_correct.sum()) == d["n_models_higher_confidence_when_correct"]
    assert c["definitions"] == co.DEFINITIONS


def test_confidence_refuses_a_value_that_differs_from_the_run_records(frame, tmp_path, monkeypatch):
    csv_ = pd.read_csv(co.RESULTS_CSV, usecols=["model", "case_id", "confidence"])
    csv_.loc[0, "confidence"] = 0.123
    csv_.to_csv(tmp_path / "r.csv", index=False)
    monkeypatch.setattr(co, "RESULTS_CSV", tmp_path / "r.csv")
    with pytest.raises(ValueError, match="differs"):
        co.attach_confidence(frame)


def test_committed_confidence_tables_are_current(conf_frame):
    for name, sub in (("all", conf_frame), ("discriminable", conf_frame[conf_frame.visible_status != "identical"])):
        committed = pd.read_csv(REPO / "analysis_original" / f"confidence_{name}.csv", dtype=str, keep_default_na=False)
        fresh = pd.read_csv(io.StringIO(co.per_model(sub).to_csv(index=False)), dtype=str, keep_default_na=False)
        assert committed.equals(fresh), name


# ---------------------------------------------------------------- Task 1b: physician answers export and status
def test_analyzed_pairs_export(committed_summary):
    out = pd.read_csv(VERIFY_DIR / "round1_analyzed_pairs.csv", dtype=str)
    a, b = rs.load_forms()
    assert list(out.case_id) == rs.analyzed_case_ids() and len(out) == 333
    assert list(out.columns) == ["case_id", "sham_type"] + [f"{w}_{q}" for w in ("AG", "MO") for q, _ in QUESTIONS]
    assert not any("notes" in c for c in out.columns)
    for who, form in (("AG", a), ("MO", b)):
        for q, _ in QUESTIONS:
            assert list(out[f"{who}_{q}"]) == list(form.loc[out.case_id, q]), (who, q)
    assert list(out.sham_type) == list(a.loc[out.case_id, "sham_type"])


def test_round1_status_readme(committed_summary):
    text = (VERIFY_DIR / "round1" / "README_STATUS.md").read_text(encoding="utf-8")
    assert text == rs.status_text(committed_summary)
    assert "repeated experiment that was not run" in text and "exactly what the evaluated models receive" in text
    assert "show each of the 477 pairs" in text and "333 analyzed pairs" in text and "descriptively" in text
    assert "477 round-1 forms" not in text  # two forms (A.G., M.O.), 477 pairs each


# ---------------------------------------------------------------- Task 1b: model naming and legends
def test_one_name_per_model_in_the_figures_and_table1(made):
    out, _ = made
    from r2.analysis.figures import display_name
    legends = " ".join((out / f"legend_Figure{i}.md").read_text(encoding="utf-8") for i in (2, 3, 4))
    assert "DeepSeek-Chat" not in legends and f"{display_name('deepseek-chat')}, excerpts inline" in legends
    assert display_name("deepseek-chat") == "DeepSeek-V3.2"  # the R1 name
    t1 = pd.read_csv(REPO / "analysis_original" / "table1_examples.csv")
    assert all(t1.model_display == t1.model.map(display_name))


def test_figure_legend_caveats(made):
    out, _ = made
    fig2 = (out / "legend_Figure2.md").read_text(encoding="utf-8")
    fig4 = (out / "legend_Figure4.md").read_text(encoding="utf-8")
    assert "Figure 4 gives them" not in fig2
    assert "within a model, the two accuracies (circle and diamond) come from different pairs" in fig4
