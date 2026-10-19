import ast
import json
from pathlib import Path

import pytest
from PIL import Image

from r2.analysis import figures, repeated, synthetic
from r2.run.harness import load_specs

R1_SCRIPT = Path("/Volumes/Lexar/LLM_Tool_use/npj_R1_Revision_Submission_2026-06-14/05_Analysis/regen_all_figures.py")
NAMES = ("Figure2", "Figure3", "Figure4")


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    root = tmp_path_factory.mktemp("figs")
    world = synthetic.build_world(root)
    repeated.run(world["runs_main"], analysis_dir=world["analysis_dir"], pairs_path=world["pairs_path"],
                 verify_dir=world["verify_dir"], specs=world["specs"])
    out = root / "out"
    paths = figures.make_figures(world["analysis_dir"], out, specs=world["specs"])
    return world, out, paths


def test_png_pdf_and_legend_for_each_figure(made):
    _, out, paths = made
    for name in NAMES:
        for ext in (".png", ".pdf"):
            p = out / f"{name}{ext}"
            assert p.exists() and p.stat().st_size > 10_000, p
        legend = (out / f"{name}_legend.md").read_text()
        assert legend.startswith(synthetic.SYNTHETIC_BANNER + f"**Figure {name[-1]} | ")  # synthetic fixture
        assert set(paths[name]) == {"png", "pdf", "legend"}
    assert (out / "Figure2.pdf").read_bytes()[:4] == b"%PDF"


def test_png_is_600_dpi(made):
    _, out, _ = made
    for name in NAMES:
        dpi = Image.open(out / f"{name}.png").info["dpi"]
        assert round(dpi[0]) == 600 and round(dpi[1]) == 600


def _fmt(n):
    return f"{int(n):,}"


def test_legends_state_n_and_the_exclusion_rule(made):
    world, out, _ = made
    head = json.loads((world["analysis_dir"] / "headline.json").read_text())
    for name in NAMES:
        text = (out / f"{name}_legend.md").read_text()
        assert _fmt(head["primary"]["n"]) in text, name
        assert f"{head['n_pairs']} pairs" in text and f"{head['n_models']} models" in text, name
        assert f"{head['n_excluded_pairs']} pairs excluded" in text, name
        for q in ("Q3", "Q4", "Q5", "Q6", "Q7"):
            assert q in text, (name, q)
        assert "bootstrap" in text and "2,000" in text, name
    fig2 = (out / "Figure2_legend.md").read_text()
    assert _fmt(head["n_pair_units"]) in fig2 and _fmt(head["non_response"]) in fig2


def test_figure3_legend_and_labels_use_n_after_exclusions(made):
    import csv
    world, out, _ = made
    with open(world["analysis_dir"] / "per_type.csv", newline="") as fh:
        per_type = list(csv.DictReader(fh))
    text = (out / "Figure3_legend.md").read_text()
    assert "after exclusions" in text
    labels = figures.type_labels(per_type)
    for r in per_type:
        assert f"n = {_fmt(r['n'])}" in labels[r["sham_type"]]


def test_unknown_exclusion_rule_is_quoted_verbatim():
    assert figures.rule_text("Some other rule") == "Some other rule"
    from r2.verify.score_review import EXCLUSION_RULE
    plain = figures.rule_text(EXCLUSION_RULE)
    assert "q3_authentic_concordant" not in plain and "Q3" in plain and "Q7" in plain


def test_display_names_cover_every_repeated_model():
    for s in load_specs():
        name = figures.display_name(s["original_name"])
        assert name and "/" not in name, s["original_name"]
    assert figures.display_name("deepseek-chat") == "DeepSeek-V3.2"
    assert figures.display_name("gemini-2.5-flash-lite") == "Gemini-2.5-Flash-Lite"
    assert figures.display_name("unknown/model-x") == "model-x"


@pytest.mark.skipif(not R1_SCRIPT.exists(), reason="R1 figure script not on this machine")
def test_palette_and_names_are_the_r1_ones():
    tree = ast.parse(R1_SCRIPT.read_text())
    r1 = {t.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
          for t in node.targets if getattr(t, "id", None) in ("C", "DISPLAY", "ALIAS")}
    assert figures.C == r1["C"] and figures.DISPLAY == r1["DISPLAY"] and figures.ALIAS == r1["ALIAS"]


def test_missing_input_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="headline.json"):
        figures.make_figures(tmp_path / "nothing", tmp_path / "out")


def test_figures_do_not_write_into_the_analysis_dir(made):
    world, _, _ = made
    assert not list(Path(world["analysis_dir"]).glob("Figure*"))


def test_real_headline_gives_an_unmarked_legend():
    head = {"primary": {"n": 10}, "n_models": 1, "n_pairs": 5, "n_excluded_pairs": 0, "n_pair_units": 5,
            "non_response": 0, "n_api_errors": 0, "exclusion_rule": "x", "synthetic": False}
    assert figures.legend3(head).startswith("**Figure 3 | ")


# ---- fix round 1, M7: readability geometry ----
def test_figure2_per_model_points_sit_beside_their_bar_not_on_it():
    for k in (0, 1):
        xs = figures.strip_x(k, 21)
        assert all(abs(x - figures.BAR_X[k]) > figures.BAR_WIDTH / 2 for x in xs), k
    # the two strips sit in the gap between the bars, so the paired lines of panel b never cross a bar
    assert max(figures.strip_x(0, 21)) < min(figures.strip_x(1, 21))
    assert min(figures.strip_x(0, 21)) > figures.BAR_X[0] + figures.BAR_WIDTH / 2
    assert max(figures.strip_x(1, 21)) < figures.BAR_X[1] - figures.BAR_WIDTH / 2


def test_figure4_order_markers_and_ci_whisker_are_on_separate_rows():
    assert abs(figures.MARKER_DY - figures.CI_DY) >= 0.25
    assert abs(figures.MARKER_DY) < 0.34 and abs(figures.CI_DY) < 0.34  # both stay inside the bar's band


def test_figure4_open_squares_only_for_models_with_runner_set_records():
    import matplotlib
    matplotlib.use("Agg")
    from r2.analysis import figures as F
    from r2.analysis import figures_original as FO
    head, per_model, per_type, by_profile, wr = FO.load()   # load() returns 5 elements after Step 5
    from r2.run.harness import load_specs
    fig = F.draw_figure4(per_model, load_specs(), without_runner_set=wr)
    squares = [c for c in fig.axes[0].collections if getattr(c, "get_label", lambda: "")() == "without_runner_set"]
    assert len(squares) == 1 and len(squares[0].get_offsets()) == 12
