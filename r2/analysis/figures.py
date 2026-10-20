"""Style, palette, model names and the Figure 3 and Figure 4 drawings of the article's figures
(r2.analysis.figures_original builds Figures 2-4 with them).

Palette `C` and model names `DISPLAY` / `ALIAS` are copied from the R1 figure script
(npj_R1_Revision_Submission_2026-06-14/05_Analysis/regen_all_figures.py); a test pins them.
"""
import inspect

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from r2.analysis.stats import cluster_bootstrap_mean  # noqa: E402

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
INK, MUTED, REF = "#1A1A1A", "#6B7280", "#9CA3AF"
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": REF, "axes.linewidth": 0.8,
    "axes.labelcolor": C["text"], "xtick.color": C["text"], "ytick.color": C["text"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
}


def display_name(original_name):
    if original_name in DISPLAY:
        return DISPLAY[original_name]
    for short, full in ALIAS.items():
        if full == original_name:
            return DISPLAY.get(short, short)
    return original_name.split("/")[-1]


def type_labels(per_type):
    return {r["sham_type"]: f"{r['sham_type'].replace('_', ' ').title()}\nn = {int(float(r['n'])):,}"
            for r in per_type}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


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


def draw_figure3(per_type, marked=(), mark="†"):
    """The Figure 3 bars (call inside plt.rc_context(STYLE)). Types in `marked` are hatched and their label
    ends with `mark`."""
    rows = sorted(per_type, key=lambda r: -_f(r["failure_rate"]))
    labels = type_labels(rows)
    for r in rows:
        if r["sham_type"] in marked:
            first, rest = labels[r["sham_type"]].split("\n", 1)
            labels[r["sham_type"]] = f"{first}{mark}\n{rest}"
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    y = np.arange(len(rows))
    rate = np.array([100 * _f(r["failure_rate"]) for r in rows])
    lo = np.array([100 * _f(r["lo"]) for r in rows])
    hi = np.array([100 * _f(r["hi"]) for r in rows])
    ax.barh(y, rate, height=0.68, color=[CATCOLOR[r["category"]] for r in rows], edgecolor="white", zorder=2,
            hatch=[("////" if r["sham_type"] in marked else None) for r in rows] if marked else None)
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
    return fig


MARKER_DY, CI_DY = -0.13, 0.17  # Figure 4 rows: order markers above, CI whisker below (y axis is inverted)


def _model_class(spec):
    if spec and spec.get("reasoning"):
        return "reasoning"
    if spec and spec.get("provider") in ("openai", "gemini"):
        return "closed"
    return "primary"


XLIM_FIG4 = 130  # room for the right-hand label "100.0% (333 valid)"


def draw_figure4(per_model, specs, closed_label="Closed commercial model", without_runner_set=None):
    """The Figure 4 bars (call inside plt.rc_context(STYLE)); per_model rows need paa_lo and paa_hi. A closed
    commercial reasoning model is drawn as a reasoning model; `closed_label` names the closed-commercial key."""
    by_name = {s["original_name"]: s for s in specs}
    pm = sorted(per_model, key=lambda r: (-_f(r["position_averaged_accuracy"]), display_name(r["model"])))
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
    if without_runner_set is not None:
        wr = {r["model"]: r for r in without_runner_set}
        pts = [(100 * _f(wr[r["model"]]["position_averaged_accuracy_valid"]), yy) for yy, r in zip(y, pm)
               if int(float(wr[r["model"]]["n_runner_set"])) > 0]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, marker="s", s=30, facecolor="white", edgecolor=INK, linewidth=1.0, zorder=5,
                       label="without_runner_set")
    n_valid = ({r["model"]: int(float(r["n_valid"])) for r in without_runner_set
                if int(float(r["n_runner_set"])) > 0} if without_runner_set is not None else {})
    for yy, v, r in zip(y, paa, pm):
        label = f"{v:.1f}%" + (f" ({n_valid[r['model']]} valid)" if r["model"] in n_valid else "")
        ax.text(104.5, yy, label, va="center", ha="left", fontsize=8.5, color=C["text"])
    ax.set_yticks(y, [display_name(r["model"]) for r in pm])
    ax.set_ylim(len(pm) - 0.45, -0.75)
    ax.set_xlim(0, XLIM_FIG4 if without_runner_set is not None else 117)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Accuracy (%); bars: position-averaged accuracy")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    _ref50(ax, vertical=True, label="Chance (50%)")
    _grid(ax, "x")
    handles = [mpatches.Patch(color=C["reasoning"], label="Reasoning model"),
               mpatches.Patch(color=C["closed"], label=closed_label),
               mpatches.Patch(color=C["primary"], label="Open-weight model"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=INK, markersize=7,
                      label="Accuracy, modified excerpt first"),
               Line2D([0], [0], marker="D", color="w", markerfacecolor=C["injection"], markersize=6.5,
                      label="Accuracy, modified excerpt second"),
               Line2D([0], [0], color=INK, lw=1, marker="|", markersize=7, label="95% CI")]
    if without_runner_set is not None:
        handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor="white", markeredgecolor=INK,
                              markersize=6.5, label="Without runner-set choices"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.6 / (0.27 * len(pm) + 1.9)),
              ncol=3, frameon=False, handlelength=1.4, columnspacing=1.2, fontsize=8)
    return fig
