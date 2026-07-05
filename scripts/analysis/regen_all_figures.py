#!/usr/bin/env python3
"""Regenerate Figures 2, 3, 4 for the npj R1 preflight pass.
Fig2: failure-by-type, recolored by modification CATEGORY with a legend (no significance stars).
Fig3: position bias (two panels) with harmonized headline numbers (72.7/27.3, 36.7/82.4).
Fig4: per-model accuracy with position-stratification overlay, sorted by BALANCED accuracy,
      clean model labels, Gemini-2.5-Flash-Lite and Qwen3-VL=73.0 (from corrected data).
Outputs PNG (600 dpi) + PDF into 04_Figures. Fig2/Fig3 use a fixed figsize aspect that matches
their embedded slot so the docx extents stay valid on re-embed."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

BASE = Path("/Volumes/Lexar/LLM_Tool_use/safe-guideline-tooling-eval")
RV = BASE / "scripts/revision_R1/outputs"
OUTDIR = Path("/Volumes/Lexar/LLM_Tool_use/npj_R1_Revision_Submission_2026-06-14/04_Figures")

C = {'primary': '#4A6FA5', 'safety': '#B85C5C', 'semantic': '#4A6FA5',
     'injection': '#C4A35A', 'metadata': '#8B9BAE', 'navy': '#2D3748',
     'gray': '#8B9BAE', 'neutral': '#8B9BAE', 'text': '#2D3748',
     'reasoning': '#8B5CF6', 'closed': '#EF4444'}
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

CATEGORY = {
    'missing_warning': 'Clinical safety', 'dosing_error': 'Clinical safety',
    'contraindication_violation': 'Clinical safety', 'allergy_ignorance': 'Clinical safety',
    'wrong_population': 'Semantic', 'subtle_inversion': 'Semantic', 'authority_mimicry': 'Semantic',
    'prompt_injection': 'Injection',
    'fabricated_citation': 'Metadata', 'outdated_version': 'Metadata',
}
CATCOLOR = {'Clinical safety': C['safety'], 'Semantic': C['semantic'],
            'Injection': C['injection'], 'Metadata': C['metadata']}


def save(fig, name, w_in, h_in, tight=False):
    fig.set_size_inches(w_in, h_in)
    bbox = 'tight' if tight else None
    for ext, dpi in (("png", 600), ("pdf", None)):
        fig.savefig(OUTDIR / f"{name}.{ext}", dpi=dpi, facecolor='white', bbox_inches=bbox)
    print("wrote", OUTDIR / f"{name}.png", f"({w_in:.2f}x{h_in:.2f} in, tight={tight})")


def figure2():
    ms = json.load(open(RV / "master_stats.json"))
    items = sorted(ms['per_type'].items(), key=lambda kv: kv[1]['fail_pct'], reverse=True)
    names = [k.replace('_', ' ').title() for k, _ in items]
    rates = [v['fail_pct'] for _, v in items]
    totals = [v['total'] for _, v in items]
    cats = [CATEGORY[k] for k, _ in items]
    colors = [CATCOLOR[c] for c in cats]

    fig, ax = plt.subplots()
    y = np.arange(len(names))
    bars = ax.barh(y, rates, color=colors, edgecolor='white', height=0.72)
    for bar, r, t in zip(bars, rates, totals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{r:.1f}%  (n={t:,})', va='center', fontsize=9, color=C['text'])
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel('LLM failure rate (%)')
    ax.set_xlim(0, 78)
    ax.invert_yaxis()
    # add headroom above the top bar so the chance-line label never crowds its value label
    ax.set_ylim(len(names) - 0.5, -1.25)
    ax.axvline(x=50, color=C['neutral'], linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(50, -0.95, 'Chance (50%)', fontsize=8, color=C['neutral'],
            ha='center', va='center')
    ax.set_title('LLM failure rate by adversarial modification type\n(n = 10,500 evaluations across 21 models)',
                 fontweight='bold', pad=12)
    order = ['Clinical safety', 'Semantic', 'Injection', 'Metadata']
    handles = [mpatches.Patch(color=CATCOLOR[c], label=c) for c in order]
    ax.legend(handles=handles, title='Modification category', loc='lower right',
              frameon=True, framealpha=0.95, edgecolor=C['neutral'], fontsize=9, title_fontsize=9)
    fig.tight_layout()
    save(fig, "Figure2_failure_by_type", 7.55, 5.00)   # aspect 1.510 == slot
    plt.close(fig)


def figure3():
    head = json.load(open(RV / "canonical_headline.json"))
    toolA = head['primacy_pct']           # 72.74
    toolB = 100 - toolA
    accA = head['acc_sham_A']             # 36.7
    accB = head['acc_sham_B']             # 82.4
    nA = round(10500 * toolA / 100)       # 7638
    nB = 10500 - nA                       # 2862

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Panel A
    vals = [nA, nB]
    cols = [C['navy'], C['gray']]
    b = ax1.bar([0, 1], vals, color=cols, edgecolor='white', width=0.62)
    ax1.axhline(5250, color=C['gray'], linestyle='--', alpha=0.7, linewidth=1.5)
    for bar, pct in zip(b, [toolA, toolB]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 120,
                 f'{pct:.1f}%', ha='center', fontweight='bold', fontsize=13, color=C['text'])
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(['First position\n(Tool A)', 'Second position\n(Tool B)'])
    ax1.set_ylabel('Number of selections'); ax1.set_ylim(0, 8800)
    ax1.set_title('A. Tool selection distribution', fontweight='bold')
    # Panel B
    valsB = [accA, accB]
    colsB = [C['gray'], C['navy']]
    b2 = ax2.bar([0, 1], valsB, color=colsB, edgecolor='white', width=0.62)
    ax2.axhline(50, color=C['gray'], linestyle='--', alpha=0.7, linewidth=1.5)
    for bar, pct in zip(b2, valsB):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f'{pct:.1f}%', ha='center', fontweight='bold', fontsize=13, color=C['text'])
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(['Sham in first\nposition', 'Sham in second\nposition'])
    ax2.set_ylabel('Detection accuracy (%)'); ax2.set_ylim(0, 100)
    ax2.set_title('B. Impact on detection accuracy', fontweight='bold')
    fig.tight_layout()
    save(fig, "Figure3_position_bias", 11.27, 5.00)    # aspect 2.253 == slot
    plt.close(fig)


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


def figure4():
    models = json.load(open(BASE / "all_model_results.json"))['models']
    ps = {r['model']: r for r in json.load(open(RV / "position_stratified.json"))}
    norm = lambda s: s.lower().replace('-', '').replace('_', '').replace(' ', '').replace('/', '').replace('.', '')
    psn = {norm(k): v for k, v in ps.items()}

    def strat(m):
        nm = m['model']
        if nm in ALIAS and ALIAS[nm] in ps:
            return ps[ALIAS[nm]]
        return psn.get(norm(nm))

    def balanced(m):
        s = strat(m)
        return s['balanced_acc'] if s else m['accuracy']

    sm = sorted(models, key=balanced, reverse=True)   # sort by BALANCED accuracy (matches Suppl. Table S4)
    fig, ax = plt.subplots()
    y = np.arange(len(sm))
    bar_colors = [C['reasoning'] if m.get('reasoning') else C['closed'] if m.get('type') == 'Closed'
                  else C['primary'] for m in sm]
    # Bar = position-BALANCED accuracy (position-independent); overall accuracy shown as a small grey marker.
    bars = ax.barh(y, [balanced(m) for m in sm], color=bar_colors, edgecolor='white',
                   height=0.7, zorder=2, alpha=0.92)
    for yy, m in zip(y, sm):
        s = strat(m)
        if not s:
            continue
        a, b = s['acc_sham_A'], s['acc_sham_B']
        ax.plot([a, b], [yy, yy], color=C['text'], lw=1.1, alpha=0.55, zorder=3)
        ax.scatter([a], [yy], marker='o', s=46, color='#1A1A1A', edgecolor='white', linewidth=0.8, zorder=4)
        ax.scatter([b], [yy], marker='D', s=42, color=C['injection'], edgecolor='white', linewidth=0.8, zorder=4)
        ax.scatter([m['accuracy']], [yy], marker='|', s=90, color='#6B7280', linewidth=1.6, zorder=5)

    def lbl(m):
        name = DISPLAY.get(m['model'], m['model'])
        badges = []
        if m.get('reasoning'): badges.append('[R]')
        if m.get('architecture') == 'MoE': badges.append('[MoE]')
        if m.get('type') == 'Closed': badges.append('[C]')
        return f"{name} {' '.join(badges)}".strip()

    ax.set_yticks(y); ax.set_yticklabels([lbl(m) for m in sm], fontsize=9)
    ax.set_xlabel('Accuracy (%)  —  bars: position-balanced accuracy', fontsize=11)
    # widen the x-range to open a dedicated label column past 100% so the values never
    # collide with the position-B (orange diamond) markers near the bar ends
    ax.set_xlim(0, 118); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.invert_yaxis()
    LABX = 103
    for bar, m in zip(bars, sm):
        ax.text(LABX, bar.get_y() + bar.get_height() / 2,
                f'{balanced(m):.1f}%', va='center', ha='left', fontsize=9, color=C['text'])
    ax.axvline(x=50, color=C['neutral'], linestyle='--', alpha=0.7, linewidth=1.5, zorder=1)
    ax.text(50, -0.75, 'Chance level', fontsize=9, color=C['neutral'], ha='center')
    legend_elements = [
        mpatches.Patch(color=C['reasoning'], label='Reasoning model'),
        mpatches.Patch(color=C['closed'], label='Closed / commercial'),
        mpatches.Patch(color=C['primary'], label='Open source'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1A1A1A', markeredgecolor='white',
               markersize=8, label='Accuracy, sham in position A'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=C['injection'], markeredgecolor='white',
               markersize=8, label='Accuracy, sham in position B'),
        Line2D([0], [0], marker='|', color='#6B7280', markersize=10, markeredgewidth=1.6,
               linestyle='None', label='Overall accuracy'),
        Line2D([0], [0], marker='', color='w', label='[R] Reasoning  [MoE] Mixture-of-experts  [C] Closed'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.07),
              ncol=2, frameon=True, framealpha=0.95, edgecolor=C['neutral'], fontsize=9)
    fig.tight_layout()
    save(fig, "Figure4_position_stratified", 9.5, 8.6, tight=True)
    plt.close(fig)


if __name__ == "__main__":
    figure2()
    figure3()
    figure4()
    print("done")
