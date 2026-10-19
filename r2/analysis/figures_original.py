"""Figures 2-4 of the exclusion-only reanalysis: the original evaluations of the discriminable pairs, from
analysis_original/*_discriminable.csv and headline.json (python -m r2.analysis.original_corrected).

Figure 2 = position effects (primacy, and accuracy with the modified excerpt first and second, pooled and by
runner configuration), Figure 3 = failure rate by modification type, Figure 4 = per-model position-averaged
accuracy. Style, palette, model names and the Figure 3 and 4 drawings are those of r2.analysis.figures. Each
figure is written to analysis_original/figures/ as a 300-dpi PNG and a PDF without a creation date (a rebuild is
byte-identical), with its legend (n and the exclusion) as legend_FigureN.md.
CLI: python -m r2.analysis.figures_original [out_dir]
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from r2.analysis import figures as F  # noqa: E402
from r2.analysis.original_corrected import CUE_KEY, CUE_TYPE, OUT  # noqa: E402
from r2.paths import AUDIT_DIR  # noqa: E402

DPI = 300
# runner profile -> reader-facing name, numbered as in the supplement (npj_R2_WORKING/edits/build_supplement.py
# CONFIG); a single-model configuration is named by its model's display name (the R1 name)
CONFIG = {
    "openai_fc": ("Configuration 1", "OpenAI, function calling"),
    "gemini_inline7": ("Configuration 2", "Gemini, excerpts inline"),
    "deepseek_chat_inline7": ("Configuration 3", f"{F.display_name('deepseek-chat')}, excerpts inline"),
    "deepseek_reasoner_inline7": ("Configuration 4", f"{F.display_name('deepseek-reasoner')}, excerpts inline"),
    "together_inline6": ("Configuration 5", "Together AI, excerpts inline"),
}
GROUP_X = np.array([0.0, 1.35, 2.35, 3.35, 4.35, 5.35])  # pooled, then the five configurations


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load(analysis_dir=None):
    d = Path(OUT if analysis_dir is None else analysis_dir)
    head = json.loads((d / "headline.json").read_text(encoding="utf-8"))
    by_profile = {r[""]: r for r in _read_csv(d / "by_profile_discriminable.csv")}
    if set(by_profile) != set(CONFIG):
        raise ValueError(f"configurations {sorted(by_profile)} differ from {sorted(CONFIG)}")
    return (head, _read_csv(d / "per_model_discriminable.csv"), _read_csv(d / "per_type_discriminable.csv"),
            by_profile)


def _sample(head, per_model):
    """n of the analysis set and of the excluded identical pairs, from the headline and the per-model rows."""
    n_models = len(per_model)
    per = {int(float(r["n"])) for r in per_model}
    if len(per) != 1:
        raise ValueError(f"models have different numbers of evaluations: {sorted(per)}")
    n_pairs = per.pop()
    n, n_ident = head["discriminable"]["n"], head["identical"]["n"]
    if n != n_pairs * n_models or n_ident % n_models:
        raise ValueError("headline n does not match the per-model rows")
    return {"n": n, "n_pairs": n_pairs, "n_models": n_models, "n_ident": n_ident,
            "n_ident_pairs": n_ident // n_models,
            "n_coerced": head["excluding_coerced"]["discriminable"]["n_coerced_excluded"]}


def _fmt(n):
    return f"{int(n):,}"


def _tail(s):
    return (f"Analysis set: the original evaluations of the {s['n_pairs']} pairs whose authentic and modified "
            f"excerpts differ in content visible to the models (n = {_fmt(s['n'])} evaluations, {s['n_models']} "
            "models, one evaluation per model and pair). Excluded: the "
            f"{s['n_ident_pairs']} pairs whose two excerpts were identical as presented ({_fmt(s['n_ident'])} "
            f"evaluations). The {_fmt(s['n_coerced'])} evaluations of this set whose choice the original runner "
            "set to A are included (a sensitivity analysis excludes them). 95% CIs: case-cluster bootstrap over vignettes "
            f"({F.BOOT_N:,} resamples).")


def _models(n):
    return f"{n} model" + ("" if n == 1 else "s")


def legend2(head, per_model, by_profile):
    s = _sample(head, per_model)
    configs = "; ".join(f"{label.split()[-1]}, {name} ({_models(int(float(by_profile[p]['n_models'])))})"
                        for p, (label, name) in CONFIG.items())
    return ("**Figure 2 | Position effects in the original evaluations of the non-identical pairs.** a, Share of "
            f"selections of the excerpt in the first position (A), pooled across the {s['n_models']} models and for "
            "each runner configuration (bars; whiskers, 95% CI); points, individual models. b, Accuracy with the "
            "modified excerpt shown first (A) and second (B), pooled and for each configuration (bars; whiskers, "
            "95% CI). Configurations: " + configs + ". The dashed "
            "lines mark 50%, the value expected without a position preference (a) and without discrimination (b). "
            "Each model saw each pair once, in one order; within a model, the two accuracies in b therefore come "
            "from different pairs. " + _tail(s))


def _cue_text(head):
    cue = head[CUE_KEY]
    blank = json.loads((AUDIT_DIR / "format_cues.json").read_text(encoding="utf-8"))["blank_line_cue"]
    if blank["authentic_payload_rows"] != 0:
        raise ValueError("audit/format_cues.json: authentic excerpts with a blank line; the legend says none")
    def share(k):
        return f"all {_fmt(k)}" if k == cue["n"] else f"{_fmt(k)} of the {_fmt(cue['n'])}"

    return (f"†Fabricated citation (hatched) is cue-confounded: in {share(cue['rows_with_token'])} of its "
            "evaluations the modified excerpt carried the DOI 10.9999/fake.NNNN, so the word "
            f"\"{cue['token']}\" was visible, and in {share(blank['modified_payload_rows'])} it had a blank line "
            "before the citation that no authentic excerpt had.")


def _absent(head):
    names = [t.replace("_", " ") for t in head["types_without_valid_pairs"]]
    if not names:
        return ""
    text = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return (f"{text[:1].upper() + text[1:]} {'is' if len(names) == 1 else 'are'} not shown: none of "
            f"{'its' if len(names) == 1 else 'their'} pairs differed in content visible to the models. ")


def legend3(head, per_model):
    s = _sample(head, per_model)
    return ("**Figure 3 | Failure rate by modification type in the original evaluations of the non-identical "
            "pairs.** A failure is a selection of the modified excerpt. Bars: failure rate pooled across the "
            f"{s['n_models']} models, colored by modification category; whiskers: 95% CI; n: evaluations of each "
            "type. The dashed line marks 50%, the rate expected from a choice made by chance. " + _cue_text(head)
            + " " + _absent(head) + "Differences between types are associations: modification type is confounded "
            "with vignette, clinical task and guideline. " + _tail(s))


def closed_reasoning(per_model, specs):
    """Display names of the closed commercial models that Figure 4 draws as reasoning models (r2.analysis.figures
    _model_class puts reasoning first)."""
    by_name = {s["original_name"]: s for s in specs}
    return sorted(F.display_name(r["model"]) for r in per_model
                  if F._model_class(by_name.get(r["model"])) == "reasoning"
                  and F._model_class({**by_name[r["model"]], "reasoning": False}) == "closed")  # reasoning: spec exists


def _color_key(per_model, specs):
    names = closed_reasoning(per_model, specs)
    if not names:
        return "reasoning model, closed commercial model or open-weight model"
    inc = (f"{names[0]}, a closed commercial reasoning model" if len(names) == 1
           else f"{', '.join(names[:-1])} and {names[-1]}, closed commercial reasoning models")
    return f"reasoning model (including {inc}), other closed commercial model or open-weight model"


def legend4(head, per_model, specs):
    s = _sample(head, per_model)
    return ("**Figure 4 | Position-averaged accuracy by model in the original evaluations of the non-identical "
            "pairs.** Bars: position-averaged accuracy, the mean of the accuracy with the modified excerpt shown "
            "first and with it shown second; whiskers: 95% CI. Circles: accuracy with the modified excerpt first; "
            "diamonds: accuracy with the modified excerpt second. Each model saw each pair once, in one order; within a "
            "model, the two accuracies (circle and diamond) come from different pairs. Bar color: "
            + _color_key(per_model, specs) + ". Models are ordered by position-averaged accuracy; the dashed line marks "
            f"50% (chance). Each model: {s['n_pairs']} evaluations, one per pair. " + _tail(s))


def _save(fig, out_dir, name, legend):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"png": out_dir / f"{name}.png", "pdf": out_dir / f"{name}.pdf",
             "legend": out_dir / f"legend_{name}.md"}
    fig.savefig(paths["png"], dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.08, metadata={"CreationDate": None})
    plt.close(fig)
    paths["legend"].write_text(legend + "\n", encoding="utf-8")
    return paths


def _groups(head, per_model, by_profile):
    """(tick label, measures with CIs, per-model rows) for the pooled set and each configuration."""
    pooled = dict(head["discriminable"])
    out = [(f"Pooled\n{_models(len(per_model))}\nn = {_fmt(pooled['n'])}", pooled, per_model)]
    for p, (label, _) in CONFIG.items():
        r = {k: F._f(v) for k, v in by_profile[p].items() if k}
        out.append((f"{label}\n{_models(int(r['n_models']))}\nn = {_fmt(r['n'])}", r,
                    [m for m in per_model if m["profile"] == p]))
    return out


def _bars(ax, x, values, cis, color, width, per_model_values=None, label_size=8.5):
    """Bars with CI whiskers, the value printed inside the foot of each bar, and the per-model points of the
    groups with more than one model over their bar."""
    ax.bar(x, [100 * v for v in values], width=width, color=color, edgecolor="white", zorder=2)
    lo = [100 * (v - c[0]) for v, c in zip(values, cis)]
    hi = [100 * (c[1] - v) for v, c in zip(values, cis)]
    for xx, pts in zip(x, per_model_values or []):
        if len(pts) > 1:
            ax.scatter(xx + F._jitter(len(pts), width * 0.3), 100 * np.asarray(pts), s=11, color="white",
                       edgecolor=F.INK, linewidth=0.5, zorder=3)
    ax.errorbar(x, [100 * v for v in values], yerr=np.nan_to_num([lo, hi]), fmt="none", ecolor=F.INK,
                elinewidth=1.1, capsize=3.5, zorder=5)
    for xx, v in zip(x, values):
        ax.text(xx, 3, F._pct(v), ha="center", va="bottom", fontsize=label_size, fontweight="bold",
                color="white", zorder=6)


def figure2(head, per_model, by_profile, out_dir):
    groups = _groups(head, per_model, by_profile)

    def ci(g, m):
        return (g[f"{m}_lo"], g[f"{m}_hi"])

    with plt.rc_context(F.STYLE):
        fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.2, 7.6), gridspec_kw={"hspace": 0.5})
        _bars(ax_a, GROUP_X, [g["primacy"] for _, g, _ in groups], [ci(g, "primacy") for _, g, _ in groups],
              F.C["navy"], 0.55, [[F._f(m["primacy"]) for m in pm] for _, _, pm in groups])
        ax_a.set_ylabel("Selections of the first excerpt, A (%)")

        # per-model accuracies with the modified excerpt first and second are in Figure 4
        w, dx = 0.36, 0.2
        _bars(ax_b, GROUP_X - dx, [g["acc_sham_first"] for _, g, _ in groups],
              [ci(g, "acc_sham_first") for _, g, _ in groups], F.C["gray"], w, label_size=7)
        _bars(ax_b, GROUP_X + dx, [g["acc_sham_second"] for _, g, _ in groups],
              [ci(g, "acc_sham_second") for _, g, _ in groups], F.C["navy"], w, label_size=7)
        ax_b.set_ylabel("Accuracy (%)")
        ax_b.legend(handles=[mpatches.Patch(color=F.C["gray"], label="Modified excerpt first (A)"),
                             mpatches.Patch(color=F.C["navy"], label="Modified excerpt second (B)")],
                    loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.2)
        for ax in (ax_a, ax_b):
            ax.set_xticks(GROUP_X, [label for label, _, _ in groups], fontsize=7.5)
            ax.set_xlim(GROUP_X[0] - 0.55, GROUP_X[-1] + 0.55)
            ax.set_ylim(0, 115)
            ax.set_yticks(range(0, 101, 25))
            ax.axvline((GROUP_X[0] + GROUP_X[1]) / 2, color=F.REF, lw=0.6, zorder=1)
            F._ref50(ax)
            F._grid(ax, "y")
        F._letter(fig, ax_a, "a", dx=-0.075)
        F._letter(fig, ax_b, "b", dx=-0.075)
        return _save(fig, out_dir, "Figure2", legend2(head, per_model, by_profile))


def figure3(head, per_model, per_type, out_dir):
    with plt.rc_context(F.STYLE):
        fig = F.draw_figure3(per_type, marked={CUE_TYPE})
        return _save(fig, out_dir, "Figure3", legend3(head, per_model))


def figure4(head, per_model, out_dir, specs):
    with plt.rc_context(F.STYLE):
        closed_label = "Other closed commercial model" if closed_reasoning(per_model, specs) else \
            "Closed commercial model"
        fig = F.draw_figure4(per_model, specs, closed_label=closed_label)
        return _save(fig, out_dir, "Figure4", legend4(head, per_model, specs))


def run(analysis_dir=None, out_dir=None, specs=None):
    from r2.run.harness import load_specs
    head, per_model, per_type, by_profile = load(analysis_dir)
    out_dir = Path(OUT if analysis_dir is None else analysis_dir) / "figures" if out_dir is None else Path(out_dir)
    specs = load_specs() if specs is None else specs
    return {"Figure2": figure2(head, per_model, by_profile, out_dir),
            "Figure3": figure3(head, per_model, per_type, out_dir),
            "Figure4": figure4(head, per_model, out_dir, specs)}


if __name__ == "__main__":
    for name, paths in run(out_dir=Path(sys.argv[1]) if len(sys.argv) > 1 else None).items():
        print(name, paths["png"])
