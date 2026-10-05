from r2.analysis.original_corrected import run


def test_headline_numbers():
    h = run(write=False)
    assert h["identical"]["n"] == 3507
    assert abs(h["identical"]["accuracy"] - 0.490) < 0.001
    assert abs(h["identical"]["primacy"] - 0.880) < 0.001
    assert h["discriminable"]["n"] == 6993
    assert abs(h["discriminable"]["accuracy"] - 0.6458) < 0.0005
    assert h["all"]["n"] == 10500
    assert abs(h["all"]["primacy"] - 0.7274) < 0.0005
    assert set(h["types_without_valid_pairs"]) == {"contraindication_violation", "missing_warning"}


def test_profile_interaction_is_estimable():
    import numpy as np
    import pandas as pd

    from r2.analysis.original_corrected import OUT

    run(write=True)
    for name in ("all", "discriminable"):
        g = pd.read_csv(OUT / f"gee_profile_interaction_{name}.csv", index_col=0)
        row = g.loc["sham_first_x_inline"]
        assert np.isfinite([row["OR"], row["lo"], row["hi"]]).all()
