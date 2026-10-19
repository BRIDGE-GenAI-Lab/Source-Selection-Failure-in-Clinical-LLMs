"""Clinically evaluable pairs of the round-1 physician review (post hoc; requested in the pre-submission review):
pairs that both physicians judged answerable from the vignette (Q2), relevant to the question (Q4) and made less
trustworthy by the modification (Q7). The primary analysis still includes all 333 analyzed pairs."""
import pandas as pd

from r2.paths import VERIFY_DIR

PAIRS_CSV = VERIFY_DIR / "round1_analyzed_pairs.csv"
CONSENSUS_CSV = VERIFY_DIR / "round1_q3_consensus.csv"
CRITERIA = ("q2_question_answerable", "q4_authentic_relevant", "q7_edit_reduces_trust")
Q3 = "q3_authentic_concordant"
Q3_VALUES = ("Y", "P", "N")
N_ANALYZED = 333


def load_pairs(path=PAIRS_CSV):
    p = pd.read_csv(path, dtype=str).set_index("case_id")
    if len(p) != N_ANALYZED or not p.index.is_unique:
        raise ValueError(f"{path}: expected {N_ANALYZED} unique analyzed pairs, found {len(p)}")
    return p


def both_yes(pairs, item):
    return (pairs[f"AG_{item}"] == "Y") & (pairs[f"MO_{item}"] == "Y")


def evaluable_mask(pairs):
    mask = pd.Series(True, index=pairs.index)
    for item in CRITERIA:
        mask &= both_yes(pairs, item)
    return mask


def discordant_q3(pairs):
    """The clinically evaluable pairs on which A.G. and M.O. answered Q3 differently, sorted by case_id."""
    ev = pairs[evaluable_mask(pairs)]
    return ev[ev[f"AG_{Q3}"] != ev[f"MO_{Q3}"]].sort_index()
