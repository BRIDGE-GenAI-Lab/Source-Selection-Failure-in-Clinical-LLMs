import pandas as pd
import pytest

from r2.analysis import clinically_evaluable as ce
from r2.verify import evaluable as ev


def _consensus(take="MO"):
    d = ev.discordant_q3(ev.load_pairs())
    return pd.DataFrame({f"consensus_{ev.Q3}": d[f"{take}_{ev.Q3}"]}, index=d.index)


def test_tiers_with_fixture_consensus():
    p = ev.load_pairs()
    ids = ce.tier_case_ids(p, _consensus("MO"))
    sizes = {k: len(v) for k, v in ids.items()}
    assert sizes == {"q4_relevant": 85, "q2_answerable": 141, "q7_trust_reducing": 150, "evaluable": 25,
                     "evaluable_concordant": 24, "evaluable_concordant_full": 16,
                     "evaluable_excluding_fabricated_citation": 21}


def test_run_reproduces_known_subset_estimates():
    out = ce.run(write=False, consensus=_consensus("MO"))
    t = out["tiers"]["evaluable"]
    assert (t["n_pairs"], t["n"], t["estimable"]) == (25, 525, True)
    assert abs(t["position_averaged_accuracy"] - 0.664) < 0.0006
    assert abs(t["acc_sham_first"] - 0.491) < 0.0006 and abs(t["acc_sham_second"] - 0.837) < 0.0006
    q = out["q3_consensus"]
    assert (q["n_evaluable"], q["n_agreed"], q["n_discordant"]) == (25, 6, 19)
    assert out["evaluable_other_items"] == {"q1_both_Y": 25, "q6_both_Y": 25, "q8_any_Y": 0}


def test_consensus_must_cover_exactly_the_discordant_pairs():
    c = _consensus()
    with pytest.raises(ValueError):
        ce.q3_final(ev.load_pairs(), c.iloc[1:])
    extra = pd.concat([c, pd.DataFrame({f"consensus_{ev.Q3}": ["Y"]}, index=["CASE_001"])])
    with pytest.raises(ValueError):
        ce.q3_final(ev.load_pairs(), extra)


def test_single_position_tier_is_not_estimable():
    from r2.analysis import original_corrected as oc
    df = oc.original_frame()
    sub = df[(df.visible_status != "identical") & df.sham_first].head(42)
    s = ce.summarize(sub)
    assert s["estimable"] is False and "position_averaged_accuracy" not in s


def test_excluding_coerced_guarded_when_one_position_left():
    import json

    from r2.analysis import original_corrected as oc
    df = oc.original_frame()
    d = df[df.visible_status != "identical"]
    first = d[d.sham_first & ~d.coerced].head(20)
    second = d[~d.sham_first & d.coerced].head(5)
    assert len(second) > 0
    s = ce.summarize(pd.concat([first, second]))
    assert s["estimable"] is True
    assert s["excluding_coerced"]["estimable"] is False
    assert "NaN" not in json.dumps(s)


def test_summarize_empty_frame():
    from r2.analysis import original_corrected as oc
    s = ce.summarize(oc.original_frame().iloc[0:0])
    assert s["n_pairs"] == 0 and s["estimable"] is False


def test_new_tier_is_last_and_drops_only_fabricated_citation_pairs():
    assert ce.TIERS[-1][0] == "evaluable_excluding_fabricated_citation"
    p = ev.load_pairs()
    ids = ce.tier_case_ids(p, _consensus("MO"))
    drop = set(ids["evaluable"]) - set(ids["evaluable_excluding_fabricated_citation"])
    assert drop and set(p.loc[sorted(drop), "sham_type"]) == {"fabricated_citation"}
    assert not (set(p.loc[ids["evaluable_excluding_fabricated_citation"], "sham_type"]) & {"fabricated_citation"})


def test_new_tier_values_and_distinct_authentic_excerpts():
    out = ce.run(write=False, consensus=_consensus("MO"))
    t = out["tiers"]["evaluable_excluding_fabricated_citation"]
    assert (t["n_pairs"], t["estimable"]) == (21, True)
    assert out["tiers"]["evaluable"]["n_distinct_authentic_excerpts"] == 19
    for name, _ in ce.TIERS:
        assert 1 <= out["tiers"][name]["n_distinct_authentic_excerpts"] <= out["tiers"][name]["n_pairs"]


def test_distinct_excerpts_counted_from_the_audit_hash():
    from r2.analysis import original_corrected as oc
    sub = oc.original_frame().head(0)
    assert ce.summarize(sub, hashes=pd.Series(dtype=str))["n_distinct_authentic_excerpts"] == 0
    df = oc.original_frame()
    df = df[df.visible_status != "identical"]
    ids = sorted(df.case_id.unique())[:3]
    h = pd.Series({ids[0]: "x", ids[1]: "x", ids[2]: "y"})
    assert ce.summarize(df[df.case_id.isin(ids)], hashes=h)["n_distinct_authentic_excerpts"] == 2
