import ast

import pytest

from r2.analysis import figures
from r2.analysis.figures_original import load_specs
from r2.paths import LEGACY_ROOT

R1_SCRIPT = (LEGACY_ROOT / "npj_R1_Revision_Submission_2026-06-14" / "05_Analysis" / "regen_all_figures.py"
             if LEGACY_ROOT else None)


def test_display_names_cover_every_repeated_model():
    for s in load_specs():
        name = figures.display_name(s["original_name"])
        assert name and "/" not in name, s["original_name"]
    assert figures.display_name("deepseek-chat") == "DeepSeek-V3.2"
    assert figures.display_name("gemini-2.5-flash-lite") == "Gemini-2.5-Flash-Lite"
    assert figures.display_name("unknown/model-x") == "model-x"


@pytest.mark.skipif(R1_SCRIPT is None or not R1_SCRIPT.exists(),
                    reason="R1 figure script not on this machine (R2_LEGACY_ROOT)")
def test_palette_and_names_are_the_r1_ones():
    tree = ast.parse(R1_SCRIPT.read_text())
    r1 = {t.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
          for t in node.targets if getattr(t, "id", None) in ("C", "DISPLAY", "ALIAS")}
    assert figures.C == r1["C"] and figures.DISPLAY == r1["DISPLAY"] and figures.ALIAS == r1["ALIAS"]


def test_figure4_order_markers_and_ci_whisker_are_on_separate_rows():
    assert abs(figures.MARKER_DY - figures.CI_DY) >= 0.25
    assert abs(figures.MARKER_DY) < 0.34 and abs(figures.CI_DY) < 0.34  # both stay inside the bar's band


def test_figure4_open_squares_only_for_models_with_runner_set_records():
    import matplotlib
    matplotlib.use("Agg")
    from r2.analysis import figures as F
    from r2.analysis import figures_original as FO
    head, per_model, per_type, by_profile, wr = FO.load()   # load() returns 5 elements after Step 5
    fig = F.draw_figure4(per_model, load_specs(), without_runner_set=wr)
    squares = [c for c in fig.axes[0].collections if getattr(c, "get_label", lambda: "")() == "without_runner_set"]
    assert len(squares) == 1 and len(squares[0].get_offsets()) == 12


def test_figure4_labels_give_valid_replies_for_runner_set_models_and_do_not_clip():
    import matplotlib
    matplotlib.use("Agg")
    from r2.analysis import figures as F
    from r2.analysis import figures_original as FO
    head, per_model, per_type, by_profile, wr = FO.load()
    fig = F.draw_figure4(per_model, load_specs(), without_runner_set=wr)
    texts = [t.get_text() for t in fig.axes[0].texts]
    labelled = [t for t in texts if "valid)" in t]
    assert len(labelled) == 12 and all(t.endswith(" valid)") for t in labelled)
    n_valid = {int(float(r["n_valid"])) for r in wr if int(float(r["n_runner_set"])) > 0}
    assert {int(t.split("(")[1].split()[0]) for t in labelled} == n_valid
    assert sum(t.endswith("%") for t in texts) == len(per_model) - 12
    fig.canvas.draw()
    ax_box = fig.axes[0].get_window_extent()
    for t in fig.axes[0].texts:
        assert t.get_window_extent().x1 <= fig.bbox.x1


def test_squares_sentence_is_computed_and_states_the_conditional_caveat():
    from r2.analysis import figures_original as FO
    head, per_model, per_type, by_profile, wr = FO.load()
    s = FO._squares_sentence(wr)
    rows = [r for r in wr if int(float(r["n_runner_set"])) > 0]
    top = max(rows, key=lambda r: int(float(r["n_runner_set"])))
    assert s.startswith("Open squares: position-averaged accuracy among only the evaluations with a valid A or B reply, "
                        f"for the {len(rows)} models whose choice the original runner set to A in some evaluations; "
                        "the label gives each one's valid replies (Supplementary Table S9, Part D). ")
    assert f"these were {int(float(top['n_valid']))} of its {int(float(top['n']))} evaluations." in s
    assert s.endswith("The squares are conditional on a valid reply, a selected subset rather than a random sample; "
                      "they are not the models' overall accuracy and do not rank the models. ")
    assert FO._squares_sentence(None) == ""
