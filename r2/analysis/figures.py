"""Figures 2-4 of the R2 manuscript, from the repeated-experiment outputs (analysis_repeated/).

Numbering follows first citation in the manuscript (ruling 2026-10-05): Figure 2 = position effects,
Figure 3 = failure rate by modification type, Figure 4 = per-model position-averaged accuracy.
Each figure is written as a 600-dpi PNG and a PDF, with its legend (n and the exclusion rule) as
FigureN_legend.md, the convention of npj_R2_WORKING/05_Figures (Figure1_legend.md).
Palette `C` and model names `DISPLAY` / `ALIAS` are copied from the R1 figure script
(npj_R1_Revision_Submission_2026-06-14/05_Analysis/regen_all_figures.py); a test pins them.
CLI: python -m r2.analysis.figures [analysis_dir] [out_dir]
"""
import csv
import inspect
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from r2.analysis.stats import cluster_bootstrap_mean  # noqa: E402

OUT_DIR = Path("/Volumes/Lexar/LLM_Tool_use/npj_R2_WORKING/05_Figures")
DPI = 600
BOOT_N = inspect.signature(cluster_bootstrap_mean).parameters["n_boot"].default

# ---- verbatim from the R1 figure script ----
C = {'primary': '#4A6FA5', 'safety': '#B85C5C', 'semantic': '#4A6FA5',
     'injection': '#C4A35A', 'metadata': '#8B9BAE', 'navy': '#2D3748',
     'gray': '#8B9BAE', 'neutral': '#8B9BAE', 'text': '#2D3748',
     'reasoning': '#8B5CF6', 'closed': '#EF4444'}
DISPLAY = {
    'deepseek-reasoner': 'DeepSeek-Reasoner', 'DeepSeek Reasoner': 'DeepSeek-Reasoner',
    'Qwen/Qwen3-VL-8B-Instruct': 'Qwen3-VL-8B',
    'ServiceNow-AI/Apriel-1.6-15b-Thinker': 'Apriel-1.6-15B-Thinker',
    'openai/gpt-oss-120b': 'GPT-OSS-120B',
    'Qwen/Qwen3-235B-A22B-Instruct-2507-tput': 'Qwen3-235B-A22B',
    'openai/gpt-oss-20b': 'GPT-OSS-20B',
    'mistralai/Mistral-Small-24B-Instruct-2501': 'Mistral-Small-24B',
    'Qwen/Qwen3-Next-80B-A3B-Thinking': 'Qwen3-Next-80B',
    'GPT-4.1': 'GPT-4.1', 'DeepSeek-V3.2': 'DeepSeek-V3.2', 'GPT-5-Nano': 'GPT-5-Nano',
    'meta-llama/Llama-4-Scout-17B-16E-Instruct': 'Llama-4-Scout-17B',
    'google/gemma-3n-E4B-it': 'Gemma-3n-E4B', 'GPT-4o-Mini': 'GPT-4o-Mini',
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8': 'Llama-4-Maverick-17B',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo': 'Llama-3.3-70B',
    'Gemini-2.5-Flash-Lite': 'Gemini-2.5-Flash-Lite', 'Gemini-2.5-Flash': 'Gemini-2.5-Flash-Lite',
    'GPT-4.1-Nano': 'GPT-4.1-Nano',
    'meta-llama/Llama-3.2-3B-Instruct-Turbo': 'Llama-3.2-3B',
    'nvidia/NVIDIA-Nemotron-Nano-9B-v2': 'Nemotron-Nano-9B',
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 'Mixtral-8x7B',
}
ALIAS = {'GPT-4.1': 'gpt-4.1-2025-04-14', 'DeepSeek-V3.2': 'deepseek-chat',
         'GPT-5-Nano': 'gpt-5-nano-2025-08-07', 'GPT-4o-Mini': 'gpt-4o-mini-2024-07-18',
         'Gemini-2.5-Flash-Lite': 'gemini-2.5-flash-lite', 'GPT-4.1-Nano': 'gpt-4.1-nano-2025-04-14'}
# ---- end of the R1 copy ----

CATEGORY_LABEL = {"clinical_safety": "Clinical safety", "semantic": "Semantic", "injection": "Injection",
                  "metadata": "Metadata"}
CATCOLOR = {"clinical_safety": C["safety"], "semantic": C["semantic"], "injection": C["injection"],
            "metadata": C["metadata"]}
OUTCOME_COLOR = {"invariant_correct": C["primary"], "position_driven": C["gray"], "wrong_both": C["safety"]}
OUTCOME_LABEL = {"invariant_correct": "Correct in both orders", "position_driven": "Position-driven (same position in both orders)",
                 "wrong_both": "Wrong in both orders"}
INK, MUTED, REF = "#1A1A1A", "#6B7280", "#9CA3AF"
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": REF, "axes.linewidth": 0.8,
    "axes.labelcolor": C["text"], "xtick.color": C["text"], "ytick.color": C["text"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
}

RULE_PREFIX = "Excluded from the primary analysis if physician consensus answered N to any of: "
ITEM_PHRASE = {
    "q1_vignette_consistent": "the vignette is internally consistent (Q1)",
    "q2_question_answerable": "the question is answerable from the vignette (Q2)",
    "q3_authentic_concordant": "the authentic excerpt is concordant with the cited guideline (Q3)",
    "q4_authentic_relevant": "the authentic excerpt is relevant to the clinical question (Q4)",
    "q5_edit_visible": "the modification is visible in the excerpt as shown to the model (Q5)",
    "q6_edit_matches_type": "the modification matches its declared type, with no other change (Q6)",
    "q7_edit_reduces_trust": "the modification makes the excerpt less trustworthy for the case (Q7)",
}


def display_name(original_name):
    if original_name in DISPLAY:
        return DISPLAY[original_name]
    for short, full in ALIAS.items():
        if full == original_name:
            return DISPLAY.get(short, short)
    return original_name.split("/")[-1]


def rule_text(rule):
    """The exclusion rule of verification/excluded_pairs.json in words; any other rule is quoted verbatim."""
    if not rule or not rule.startswith(RULE_PREFIX):
        return rule or "not recorded"
    items = [x.strip() for x in rule[len(RULE_PREFIX):].split(",")]
    if not all(i in ITEM_PHRASE for i in items):
        return rule
    return ("pairs were excluded when the physicians' consensus answered No to any of these questions: "
            + "; ".join(ITEM_PHRASE[i] for i in items))


def type_labels(per_type):
    return {r["sham_type"]: f"{r['sham_type'].replace('_', ' ').title()}\nn = {int(float(r['n'])):,}"
            for r in per_type}


def _fmt(n):
    return f"{int(n):,}"


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _load(analysis_dir):
    analysis_dir = Path(analysis_dir)
    for name in ("headline.json", "per_model.csv", "per_type.csv"):
        if not (analysis_dir / name).exists():
            raise FileNotFoundError(f"{analysis_dir / name} is missing; run r2.analysis.repeated first")
    head = json.loads((analysis_dir / "headline.json").read_text())
    return head, _read_csv(analysis_dir / "per_model.csv"), _read_csv(analysis_dir / "per_type.csv")


def _common_tail(head):
    rule = rule_text(head.get("exclusion_rule"))
    return (f"95% CIs: case-cluster bootstrap over vignettes ({BOOT_N:,} resamples). Analysis set: "
            f"n = {_fmt(head['primary']['n'])} valid responses from {head['n_models']} models on "
            f"{head['n_pairs']} pairs, each pair shown in both orders ({head['n_excluded_pairs']} pairs excluded). "
            f"Exclusion rule: {rule[:1].upper() + rule[1:]}.")


def legend2(head):
    return ("**Figure 2 | Position effects in the repeated experiment.** a, Share of selections of the excerpt in the "
            "first position (A) and in the second position (B), pooled across models (bars; whiskers, 95% CI) and "
            "for each model (points). b, Accuracy when the modified excerpt was shown first and when it was shown "
            "second, pooled (bars; whiskers, 95% CI) and for each model (points joined by a line). The dashed lines "
            "mark 50%, the value expected without a position preference (a) and without discrimination (b). "
            "c, Outcome of each model-pair unit with a valid response in both orders: correct in both orders, "
            "position-driven (the same position chosen in both orders) or wrong in both orders; models are ordered "
            f"by the share correct in both orders (n = {_fmt(head['n_pair_units'])} model-pair units). "
            f"{_fmt(head['non_response'])} non-responses (replies without a valid choice, "
            f"{_fmt(head['n_api_errors'])} of them after an API error) are not included. " + _common_tail(head))


def legend3(head):
    return ("**Figure 3 | Failure rate by modification type in the repeated experiment.** A failure is a selection of "
            f"the modified excerpt. Bars: failure rate pooled across {head['n_models']} models and both orders, "
            "colored by modification category; whiskers: 95% CI; n: evaluations of each type after exclusions. "
            "The dashed line marks 50%, the rate expected from a choice made by chance. Differences between types "
            "are associations: modification type is confounded with vignette, clinical task and guideline. "
            + _common_tail(head))


def legend4(head):
    return ("**Figure 4 | Position-averaged accuracy by model in the repeated experiment.** Bars: position-averaged "
            "accuracy, the mean of the accuracy with the modified excerpt shown first and with it shown second; "
            "whiskers: 95% CI. Circles: accuracy with the modified excerpt first; diamonds: accuracy with the "
            "modified excerpt second. Bar color: reasoning model, closed commercial model or open-weight model. "
            "Models are ordered by position-averaged accuracy; the dashed line marks 50% (chance). "
            + _common_tail(head))


def _letter(fig, ax, letter, dx=0.0):
    pos = ax.get_position()
    fig.text(pos.x0 + dx, pos.y1 + 0.012, letter, fontsize=12, fontweight="bold", color=INK, va="bottom")


def _ref50(ax, vertical=False, label=None):
    if vertical:
        ax.axvline(50, color=REF, ls="--", lw=0.9, zorder=1)
    else:
        ax.axhline(50, color=REF, ls="--", lw=0.9, zorder=1)
    if label:
        ax.text(50, 1.005, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.5, color=MUTED)


def _grid(ax, axis):
    ax.grid(axis=axis, color="#EEF0F3", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _jitter(n, width=0.16):
    return np.linspace(-width, width, n) if n > 1 else np.zeros(n)


def _pct(v):
    return f"{100 * v:.1f}%"


def _save(fig, out_dir, name, legend, synthetic=False):
    """PNG (600 dpi), PDF and the legend as markdown (05_Figures convention: FigureN_legend.md). A synthetic
    figure gets a watermark and its legend opens with the synthetic banner."""
    if synthetic:
        from r2.analysis.synthetic import SYNTHETIC_BANNER
        fig.text(0.5, 0.5, "SYNTHETIC DATA", fontsize=54, color="#9CA3AF", alpha=0.25, rotation=30,
                 ha="center", va="center", zorder=100)
        legend = SYNTHETIC_BANNER + legend
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"png": out_dir / f"{name}.png", "pdf": out_dir / f"{name}.pdf", "legend": out_dir / f"{name}_legend.md"}
    fig.savefig(paths["png"], dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    paths["legend"].write_text(legend + "\n", encoding="utf-8")
    return paths


# Figure 2a/2b layout: two bars far enough apart that each bar's per-model points sit in the gap beside it
# (never over the bar or its narrow pooled whisker), and panel b's paired lines run between the two strips.
BAR_X = (0.0, 1.8)
BAR_WIDTH = 0.6
STRIP_OFFSET = 0.55   # bar centre to strip centre, towards the other bar
STRIP_HALF = 0.1      # half-width of a strip's jitter


def strip_x(k, n):
    """x positions of the n per-model points beside bar k (0: right of the first bar, 1: left of the second)."""
    centre = BAR_X[k] + (STRIP_OFFSET if k == 0 else -STRIP_OFFSET)
    return centre + _jitter(n, STRIP_HALF)


def _bar_panel(ax, values, cis, colors, ticklabels, per_model, ylabel, paired=False):
    x = np.array(BAR_X)
    heights = [100 * v for v in values]
    ax.bar(x, heights, width=BAR_WIDTH, color=colors, edgecolor="white", zorder=2)
    lo = [100 * (v - c[0]) for v, c in zip(values, cis)]
    hi = [100 * (c[1] - v) for v, c in zip(values, cis)]
    ax.errorbar(x, heights, yerr=np.nan_to_num([lo, hi]), fmt="none", ecolor=INK, elinewidth=1.2, capsize=5,
                zorder=5)
    n = len(per_model[0])
    xs = [strip_x(k, n) for k in range(len(x))]
    if paired:
        for i in range(n):
            ax.plot([xs[0][i], xs[1][i]], [100 * per_model[0][i], 100 * per_model[1][i]], color=MUTED, lw=0.5,
                    alpha=0.6, zorder=3)
    for k in range(len(x)):
        ax.scatter(xs[k], 100 * np.asarray(per_model[k]), s=15, color="white", edgecolor=INK, linewidth=0.6,
                   zorder=4)
    for k, v in enumerate(values):
        top = 100 * cis[k][1] if np.isfinite(cis[k][1]) else 100 * v
        ax.text(x[k], max(top, 100 * v) + 3, _pct(v), ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=C["text"])
    ax.set_xticks(x, ticklabels)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel(ylabel)
    _ref50(ax)
    _grid(ax, "y")


def figure2(head, per_model, out_dir):
    prim = head["primary"]
    pm = sorted(per_model, key=lambda r: (-_f(r["invariant_correct"]), display_name(r["model"])))
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.2, 9.0))
        top = fig.add_gridspec(1, 2, left=0.09, right=0.98, top=0.95, bottom=0.66, wspace=0.32)
        bottom = fig.add_gridspec(1, 1, left=0.25, right=0.98, top=0.56, bottom=0.07)
        ax_a, ax_b, ax_c = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1]), fig.add_subplot(bottom[0, 0])

        p, plo, phi = prim["primacy"], prim["primacy_lo"], prim["primacy_hi"]
        first = [_f(r["primacy"]) for r in pm]
        _bar_panel(ax_a, [p, 1 - p], [(plo, phi), (1 - phi, 1 - plo)], [C["navy"], C["gray"]],
                   ["First position (A)", "Second position (B)"], [first, [1 - v for v in first]],
                   "Selections (%)")
        _bar_panel(ax_b, [prim["acc_sham_first"], prim["acc_sham_second"]],
                   [(prim["acc_sham_first_lo"], prim["acc_sham_first_hi"]),
                    (prim["acc_sham_second_lo"], prim["acc_sham_second_hi"])], [C["gray"], C["navy"]],
                   ["Modified excerpt\nfirst", "Modified excerpt\nsecond"],
                   [[_f(r["acc_sham_first"]) for r in pm], [_f(r["acc_sham_second"]) for r in pm]],
                   "Accuracy (%)", paired=True)

        y = np.arange(len(pm))
        left = np.zeros(len(pm))
        for key in ("invariant_correct", "position_driven", "wrong_both"):
            w = np.array([100 * _f(r[key]) for r in pm])
            w = np.nan_to_num(w)
            ax_c.barh(y, w, left=left, height=0.72, color=OUTCOME_COLOR[key], edgecolor="white", linewidth=0.6,
                      zorder=2, label=OUTCOME_LABEL[key])
            for yy, ll, ww in zip(y, left, w):
                if ww >= 7:
                    ax_c.text(ll + ww / 2, yy, f"{ww:.0f}", ha="center", va="center", fontsize=7,
                              color="white" if key != "position_driven" else INK, zorder=3)
            left += w
        ax_c.set_yticks(y, [display_name(r["model"]) for r in pm])
        ax_c.set_ylim(len(pm) - 0.45, -0.55)
        ax_c.set_xlim(0, 100)
        ax_c.set_xticks(range(0, 101, 20))
        ax_c.set_xlabel("Model-pair units with a valid response in both orders (%)")
        ax_c.spines["left"].set_visible(False)
        ax_c.tick_params(axis="y", length=0)
        ax_c.legend(loc="lower center", bbox_to_anchor=(0.38, 1.0), ncol=3, frameon=False,
                    handlelength=1.2, columnspacing=1.4, fontsize=8)
        for ax, letter in ((ax_a, "a"), (ax_b, "b")):
            _letter(fig, ax, letter, dx=-0.075)
        _letter(fig, ax_c, "c", dx=-0.235)
        return _save(fig, out_dir, "Figure2", legend2(head), head.get("synthetic") is True)


def figure3(head, per_type, out_dir):
    rows = sorted(per_type, key=lambda r: -_f(r["failure_rate"]))
    labels = type_labels(rows)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        y = np.arange(len(rows))
        rate = np.array([100 * _f(r["failure_rate"]) for r in rows])
        lo = np.array([100 * _f(r["lo"]) for r in rows])
        hi = np.array([100 * _f(r["hi"]) for r in rows])
        ax.barh(y, rate, height=0.68, color=[CATCOLOR[r["category"]] for r in rows], edgecolor="white", zorder=2)
        ax.errorbar(rate, y, xerr=np.nan_to_num([rate - lo, hi - rate]), fmt="none", ecolor=INK, elinewidth=1.0,
                    capsize=3, zorder=3)
        for yy, r in zip(y, rate):
            ax.text(103, yy, f"{r:.1f}%", va="center", ha="left", fontsize=8.5, color=C["text"])
        ax.set_yticks(y, [labels[r["sham_type"]] for r in rows])
        ax.set_ylim(len(rows) - 0.45, -0.75)
        ax.set_xlim(0, 112)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel("Failure rate: modified excerpt selected (%)")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        _ref50(ax, vertical=True, label="50%")
        _grid(ax, "x")
        present = [k for k in CATEGORY_LABEL if any(r["category"] == k for r in rows)]
        ax.legend(handles=[mpatches.Patch(color=CATCOLOR[k], label=CATEGORY_LABEL[k]) for k in present],
                  title="Modification category", loc="upper center", bbox_to_anchor=(0.45, -0.12),
                  ncol=len(present), frameon=False, title_fontsize=8.5, handlelength=1.2)
        return _save(fig, out_dir, "Figure3", legend3(head), head.get("synthetic") is True)


MARKER_DY, CI_DY = -0.13, 0.17  # Figure 4 rows: order markers above, CI whisker below (y axis is inverted)


def _model_class(spec):
    if spec and spec.get("reasoning"):
        return "reasoning"
    if spec and spec.get("provider") in ("openai", "gemini"):
        return "closed"
    return "primary"


def figure4(head, per_model, out_dir, specs):
    by_name = {s["original_name"]: s for s in specs}
    pm = sorted(per_model, key=lambda r: (-_f(r["position_averaged_accuracy"]), display_name(r["model"])))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(pm) + 1.9))
        y = np.arange(len(pm))
        paa = np.array([100 * _f(r["position_averaged_accuracy"]) for r in pm])
        lo = np.array([100 * _f(r["paa_lo"]) for r in pm])
        hi = np.array([100 * _f(r["paa_hi"]) for r in pm])
        colors = [C[_model_class(by_name.get(r["model"]))] for r in pm]
        ax.barh(y, paa, height=0.68, color=colors, edgecolor="white", alpha=0.9, zorder=2)
        # the CI whisker runs along the lower part of the bar, the order markers along the upper part, so a
        # marker inside the CI never sits on the whisker
        ax.errorbar(paa, y + CI_DY, xerr=np.nan_to_num([paa - lo, hi - paa]), fmt="none", ecolor=INK,
                    elinewidth=1.1, capsize=2.2, zorder=3)
        a = np.array([100 * _f(r["acc_sham_first"]) for r in pm])
        b = np.array([100 * _f(r["acc_sham_second"]) for r in pm])
        ax.scatter(a, y + MARKER_DY, marker="o", s=30, color=INK, edgecolor="white", linewidth=0.7, zorder=4)
        ax.scatter(b, y + MARKER_DY, marker="D", s=26, color=C["injection"], edgecolor="white", linewidth=0.7,
                   zorder=4)
        for yy, v in zip(y, paa):
            ax.text(104.5, yy, f"{v:.1f}%", va="center", ha="left", fontsize=8.5, color=C["text"])
        ax.set_yticks(y, [display_name(r["model"]) for r in pm])
        ax.set_ylim(len(pm) - 0.45, -0.75)
        ax.set_xlim(0, 117)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel("Accuracy (%); bars: position-averaged accuracy")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        _ref50(ax, vertical=True, label="Chance (50%)")
        _grid(ax, "x")
        handles = [mpatches.Patch(color=C["reasoning"], label="Reasoning model"),
                   mpatches.Patch(color=C["closed"], label="Closed commercial model"),
                   mpatches.Patch(color=C["primary"], label="Open-weight model"),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor=INK, markersize=7,
                          label="Accuracy, modified excerpt first"),
                   Line2D([0], [0], marker="D", color="w", markerfacecolor=C["injection"], markersize=6.5,
                          label="Accuracy, modified excerpt second"),
                   Line2D([0], [0], color=INK, lw=1, marker="|", markersize=7, label="95% CI")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.6 / (0.27 * len(pm) + 1.9)),
                  ncol=3, frameon=False, handlelength=1.4, columnspacing=1.2, fontsize=8)
        return _save(fig, out_dir, "Figure4", legend4(head), head.get("synthetic") is True)


def make_figures(analysis_dir=None, out_dir=None, specs=None):
    from r2.analysis.repeated import OUT
    from r2.run.harness import load_specs
    head, per_model, per_type = _load(OUT if analysis_dir is None else analysis_dir)
    out_dir = Path(OUT_DIR if out_dir is None else out_dir)
    if head.get("synthetic") is True:  # synthetic figures never land in a real output location
        from r2.analysis.synthetic import assert_sandbox
        assert_sandbox(out_dir)
    specs = load_specs() if specs is None else specs
    return {"Figure2": figure2(head, per_model, out_dir), "Figure3": figure3(head, per_type, out_dir),
            "Figure4": figure4(head, per_model, out_dir, specs)}


def main():
    analysis_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    for name, paths in make_figures(analysis_dir, out_dir).items():
        print(name, paths["png"])


if __name__ == "__main__":
    main()
