"""Post hoc sensitivity analysis requested in the pre-submission review: the original evaluations of the 333 analyzed
pairs restricted by the round-1 physician review. Exploratory; original responses only; the physician review still
excludes no pair from the primary analysis. Q3 is the shared answer where A.G. and M.O. agreed and their consensus
(verification/round1_q3_consensus.csv) where they disagreed.

    python -m r2.analysis.clinically_evaluable
"""
import json

import pandas as pd

from r2.analysis import original_corrected as oc
from r2.verify.evaluable import (CONSENSUS_CSV, Q3, Q3_VALUES, both_yes, discordant_q3, evaluable_mask,
                                 load_pairs)

TIERS = (
    ("q4_relevant", "both physicians judged the authentic excerpt relevant to the question (Q4)"),
    ("q2_answerable", "both physicians judged the question answerable from the vignette (Q2)"),
    ("q7_trust_reducing", "both physicians judged that the modification made the excerpt less trustworthy (Q7)"),
    ("evaluable", "clinically evaluable: both physicians answered Y on Q2, Q4 and Q7"),
    ("evaluable_concordant", "clinically evaluable, with the authentic excerpt at least partly concordant with its "
                             "source (Q3 Y or P; consensus where the physicians disagreed)"),
    ("evaluable_concordant_full", "clinically evaluable, with the authentic excerpt fully concordant with its source "
                                  "(Q3 Y; consensus where the physicians disagreed)"),
)
MEASURE_COLS = ("accuracy", "acc_sham_first", "acc_sham_second", "position_averaged_accuracy", "primacy")


def load_consensus(path=CONSENSUS_CSV):
    return pd.read_csv(path, dtype=str).set_index("case_id")[[f"consensus_{Q3}"]]


def q3_final(pairs, consensus):
    """Q3 per clinically evaluable pair: the shared answer where A.G. and M.O. agreed, else the consensus."""
    disc = discordant_q3(pairs)
    missing, extra = sorted(set(disc.index) - set(consensus.index)), sorted(set(consensus.index) - set(disc.index))
    if missing or extra:
        raise ValueError(f"the Q3 consensus must cover exactly the discordant pairs: missing {missing}, extra {extra}")
    values = consensus.loc[disc.index, f"consensus_{Q3}"]
    if not values.isin(Q3_VALUES).all():
        raise ValueError(f"consensus values outside {Q3_VALUES}: {sorted(set(values) - set(Q3_VALUES))}")
    out = pairs.loc[evaluable_mask(pairs), f"AG_{Q3}"].copy()
    out.loc[disc.index] = values
    return out


def tier_case_ids(pairs, consensus):
    q3 = q3_final(pairs, consensus)
    masks = {"q4_relevant": both_yes(pairs, "q4_authentic_relevant"),
             "q2_answerable": both_yes(pairs, "q2_question_answerable"),
             "q7_trust_reducing": both_yes(pairs, "q7_edit_reduces_trust"), "evaluable": evaluable_mask(pairs)}
    ids = {k: sorted(m[m].index) for k, m in masks.items()}
    ids["evaluable_concordant"] = sorted(q3[q3.isin(("Y", "P"))].index)
    ids["evaluable_concordant_full"] = sorted(q3[q3 == "Y"].index)
    return ids


def summarize(sub):
    """Counts and, when both positions have a pair, the pooled measures with case-cluster bootstrap CIs."""
    n_first = int(sub[sub.sham_first].case_id.nunique())
    n_second = int(sub[~sub.sham_first].case_id.nunique())
    out = {"n_pairs": int(sub.case_id.nunique()), "n": int(len(sub)), "n_pairs_sham_first": n_first,
           "n_pairs_sham_second": n_second, "n_coerced": int(sub.coerced.sum()),
           "estimable": bool(n_first and n_second)}
    if out["estimable"]:
        out.update({k: float(v) for k, v in oc._measures(sub).items()})
        out["primacy"] = float(sub.selected_first.mean())
        out["excluding_coerced"] = {k: (float(v) if not isinstance(v, int) else v)
                                    for k, v in oc._excluding_coerced(sub).items()}
    return out


def run(write=True, pairs=None, consensus=None):
    pairs = load_pairs() if pairs is None else pairs
    consensus = load_consensus() if consensus is None else consensus
    df = oc.original_frame()
    disc = df[df.visible_status != "identical"]
    if set(disc.case_id) != set(pairs.index):
        raise ValueError("the physician-review pairs differ from the analysis set")
    ids = tier_case_ids(pairs, consensus)
    ev = pairs[evaluable_mask(pairs)]
    d = discordant_q3(pairs)
    q3 = q3_final(pairs, consensus)
    agreed = ev.loc[~ev.index.isin(d.index), f"AG_{Q3}"]
    out = {"label": "post hoc; exploratory",
           "note": "Restricts the 333 analyzed pairs by the round-1 physician review; the primary analysis is "
                   "unchanged and the review excludes no pair.",
           "tiers": {name: {"definition": desc, **summarize(disc[disc.case_id.isin(ids[name])]),
                            "case_ids": ids[name]} for name, desc in TIERS},
           "q3_consensus": {"n_evaluable": int(len(ev)), "n_agreed": int(len(agreed)), "n_discordant": int(len(d)),
                            "agreed": {v: int((agreed == v).sum()) for v in Q3_VALUES},
                            "consensus": {v: int((q3.loc[d.index] == v).sum()) for v in Q3_VALUES},
                            "discordant_patterns": {f"{a}/{m}": int(n) for (a, m), n in
                                                    d.groupby([f"AG_{Q3}", f"MO_{Q3}"]).size().items()}},
           "evaluable_other_items": {"q1_both_Y": int(both_yes(ev, "q1_vignette_consistent").sum()),
                                     "q6_both_Y": int(both_yes(ev, "q6_edit_matches_type").sum()),
                                     "q8_any_Y": int(((ev.AG_q8_leakage == "Y") | (ev.MO_q8_leakage == "Y")).sum())}}
    if write:
        oc.OUT.mkdir(exist_ok=True)
        (oc.OUT / "clinically_evaluable.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        rows = [{"tier": n, **{k: v for k, v in out["tiers"][n].items() if k not in ("case_ids", "excluding_coerced")}}
                for n, _ in TIERS]
        pd.DataFrame(rows).to_csv(oc.OUT / "clinically_evaluable.csv", index=False)
    return out


if __name__ == "__main__":
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "case_ids"} for k, v in run()["tiers"].items()},
                     indent=1))
