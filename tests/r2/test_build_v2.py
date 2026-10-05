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
