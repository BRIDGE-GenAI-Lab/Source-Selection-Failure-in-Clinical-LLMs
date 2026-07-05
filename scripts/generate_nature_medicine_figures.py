#!/usr/bin/env python3
"""
Nature Medicine Quality Figures
Multi-panel, sophisticated scientific visualizations
600 DPI, Arial font, grayish-bluish palette
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path('publication_figures')
OUTPUT_DIR.mkdir(exist_ok=True)

# Nature Medicine grayish-bluish palette
COLORS = {
    'primary': '#2C3E50',      # Dark blue-gray
    'secondary': '#34495E',    # Steel blue
    'tertiary': '#5D6D7E',     # Slate
    'light': '#85929E',        # Light gray-blue
    'pale': '#AEB6BF',         # Pale
    'very_pale': '#D5DBDB',    # Very pale
    'accent': '#1A5276',       # Deep blue accent
    'bg': '#FDFEFE',           # White
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.titleweight': 'bold',
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor': COLORS['bg'],
    'savefig.dpi': 600,
    'savefig.facecolor': COLORS['bg'],
    'savefig.bbox': 'tight',
    'legend.fontsize': 7,
    'legend.frameon': False,
})

EXPERIMENTS = {
    "GPT-4.1": "results/run_20260107_215921",
    "DeepSeek-V3.2": "results/run_20260107_183414",
    "GPT-5-Nano": "results/run_20260107_085822",
    "GPT-4o-Mini": "results/run_20260107_142734",
    "Gemini-2.5-Flash": "results/run_20260107_181625",
    "GPT-4.1-Nano": "results/run_20260107_142701",
}

SHAM_SHORT = {
    'missing_warning': 'Missing\nWarning',
    'allergy_ignorance': 'Allergy\nIgnorance',
    'dosing_error': 'Dosing\nError',
    'wrong_population': 'Wrong\nPopulation',
    'contraindication_violation': 'Contra-\nindication',
    'authority_mimicry': 'Authority\nMimicry',
    'subtle_inversion': 'Subtle\nInversion',
    'prompt_injection': 'Prompt\nInjection',
    'fabricated_citation': 'Fabricated\nCitation',
    'outdated_version': 'Outdated\nVersion',
}


def load_all_results():
    results = []
    for name, path in EXPERIMENTS.items():
        with open(Path(path) / "results.jsonl") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line.strip())
                    r['model'] = name
                    if r.get('sham_trap_type', 'unknown') != 'unknown':
                        results.append(r)
    return results


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return 0, 0
    p = successes / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    return max(0, center - margin), min(1, center + margin)


# ============================================================================
# FIGURE 1: Multi-panel Main Results Figure
# ============================================================================

def create_figure1(results):
    """
    4-panel figure:
    A) Overall vulnerability pie/bar
    B) Trap effectiveness ranked
    C) Position bias
    D) By sham category
    """
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    
    # Stats
    total = len(results)
    fooled = sum(1 for r in results if not r.get('selected_tool_correct'))
    
    # Panel A: Overall vulnerability with CI
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes, fontsize=12, fontweight='bold')
    
    # Bar with error
    ci_low, ci_high = wilson_ci(fooled, total)
    bars = ax_a.bar(['Fooled', 'Correct'], 
                    [fooled/total*100, (total-fooled)/total*100],
                    color=[COLORS['primary'], COLORS['pale']], 
                    edgecolor='white', width=0.5)
    ax_a.errorbar([0], [fooled/total*100], 
                  yerr=[[fooled/total*100 - ci_low*100], [ci_high*100 - fooled/total*100]],
                  fmt='none', color=COLORS['secondary'], capsize=4, linewidth=1.5)
    
    ax_a.set_ylabel('Proportion (%)')
    ax_a.set_ylim(0, 65)
    ax_a.axhline(y=50, color=COLORS['light'], linestyle='--', linewidth=0.8)
    
    # Add text
    for i, bar in enumerate(bars):
        val = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                 ha='center', fontsize=8, fontweight='bold', color=COLORS['primary'])
    
    ax_a.set_title('Overall Vulnerability Rate')
    
    # Panel B: Trap effectiveness
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes, fontsize=12, fontweight='bold')
    
    trap_stats = defaultdict(lambda: {'fail': 0, 'total': 0})
    for r in results:
        sham = r.get('sham_trap_type')
        trap_stats[sham]['total'] += 1
        if not r.get('selected_tool_correct'):
            trap_stats[sham]['fail'] += 1
    
    sorted_traps = sorted(trap_stats.items(), key=lambda x: x[1]['fail']/x[1]['total'], reverse=True)
    
    y_pos = np.arange(len(sorted_traps))
    rates = [t[1]['fail']/t[1]['total']*100 for t in sorted_traps]
    names = [t[0].replace('_', ' ').title()[:15] for t in sorted_traps]
    
    # Colors by rate
    colors = [COLORS['primary'] if r > 50 else COLORS['tertiary'] if r > 40 else COLORS['light'] 
              for r in rates]
    
    bars_b = ax_b.barh(y_pos, rates, color=colors, height=0.6, edgecolor='white', linewidth=0.3)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(names, fontsize=7)
    ax_b.set_xlabel('Failure Rate (%)')
    ax_b.set_xlim(0, 70)
    ax_b.axvline(x=50, color=COLORS['light'], linestyle='--', linewidth=0.8)
    ax_b.invert_yaxis()
    ax_b.set_title('Sham Trap Effectiveness')
    
    # Panel C: Position bias
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes, fontsize=12, fontweight='bold')
    
    valid = [r for r in results if r.get('model_decision')]
    picks_a = sum(1 for r in valid if r['model_decision'].get('selected_tool') == 'A')
    picks_b = len(valid) - picks_a
    
    # Stacked horizontal bar
    ax_c.barh([0], [picks_a/(picks_a+picks_b)*100], color=COLORS['primary'], height=0.4, label='First (A)')
    ax_c.barh([0], [picks_b/(picks_a+picks_b)*100], left=[picks_a/(picks_a+picks_b)*100], 
             color=COLORS['pale'], height=0.4, label='Second (B)')
    
    ax_c.text(picks_a/(picks_a+picks_b)*50, 0, f'{picks_a/(picks_a+picks_b)*100:.0f}%', 
             ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax_c.text(picks_a/(picks_a+picks_b)*100 + picks_b/(picks_a+picks_b)*50, 0, 
             f'{picks_b/(picks_a+picks_b)*100:.0f}%',
             ha='center', va='center', fontsize=10, fontweight='bold', color=COLORS['primary'])
    
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(-0.5, 0.5)
    ax_c.set_xlabel('Selection Rate (%)')
    ax_c.set_yticks([])
    ax_c.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    ax_c.axvline(x=50, color=COLORS['secondary'], linestyle='--', linewidth=1)
    ax_c.set_title('Position Bias: First-Option Preference')
    
    # Panel D: Impact by position
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.text(-0.15, 1.05, 'd', transform=ax_d.transAxes, fontsize=12, fontweight='bold')
    
    a_is_sham = [r for r in results if r.get('mapping', {}).get('A') == 'S']
    b_is_sham = [r for r in results if r.get('mapping', {}).get('B') == 'S']
    
    acc_sham_first = sum(1 for r in a_is_sham if r.get('selected_tool_correct'))/len(a_is_sham)*100
    acc_sham_second = sum(1 for r in b_is_sham if r.get('selected_tool_correct'))/len(b_is_sham)*100
    
    x = np.arange(2)
    bars_d = ax_d.bar(x, [acc_sham_first, acc_sham_second], 
                      color=[COLORS['tertiary'], COLORS['primary']], width=0.5, edgecolor='white')
    
    # Error bars
    for i, (data, pos) in enumerate([(a_is_sham, 0), (b_is_sham, 1)]):
        correct = sum(1 for r in data if r.get('selected_tool_correct'))
        ci_l, ci_h = wilson_ci(correct, len(data))
        acc = correct/len(data)*100
        ax_d.errorbar([pos], [acc], yerr=[[acc - ci_l*100], [ci_h*100 - acc]],
                      fmt='none', color=COLORS['secondary'], capsize=4, linewidth=1.5)
    
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(['Sham in First\nPosition', 'Sham in Second\nPosition'])
    ax_d.set_ylabel('Detection Accuracy (%)')
    ax_d.set_ylim(0, 80)
    ax_d.axhline(y=50, color=COLORS['light'], linestyle='--', linewidth=0.8)
    ax_d.set_title('Accuracy by Sham Position')
    
    for bar in bars_d:
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 f'{bar.get_height():.1f}%', ha='center', fontsize=8, fontweight='bold',
                 color=COLORS['primary'])
    
    plt.savefig(OUTPUT_DIR / 'Figure1_main.png')
    plt.savefig(OUTPUT_DIR / 'Figure1_main.pdf')
    plt.close()
    print("✓ Figure 1: Main Results (4-panel)")


# ============================================================================
# FIGURE 2: Heatmap with Marginals
# ============================================================================

def create_figure2(results):
    """
    Publication-quality heatmap with row/column marginals
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Create gridspec for heatmap + marginals
    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], 
                          hspace=0.05, wspace=0.05)
    
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_heat)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_heat)
    
    models = sorted(set(r['model'] for r in results),
                   key=lambda x: sum(1 for r in results if r['model'] == x and r.get('selected_tool_correct'))/
                                 sum(1 for r in results if r['model'] == x), reverse=True)
    shams = sorted(set(r['sham_trap_type'] for r in results),
                  key=lambda x: sum(1 for r in results if r['sham_trap_type'] == x and r.get('selected_tool_correct'))/
                                sum(1 for r in results if r['sham_trap_type'] == x), reverse=True)
    
    matrix = []
    for model in models:
        row = []
        for sham in shams:
            sham_model = [r for r in results if r['sham_trap_type'] == sham and r['model'] == model]
            if sham_model:
                acc = sum(1 for r in sham_model if r.get('selected_tool_correct'))/len(sham_model)*100
                row.append(acc)
            else:
                row.append(50)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Blue gradient colormap
    cmap = LinearSegmentedColormap.from_list('blues', 
        [COLORS['very_pale'], COLORS['light'], COLORS['tertiary'], COLORS['primary']], N=100)
    
    im = ax_heat.imshow(matrix, cmap=cmap, aspect='auto', vmin=10, vmax=100)
    
    # Annotations
    for i in range(len(models)):
        for j in range(len(shams)):
            val = matrix[i, j]
            color = 'white' if val > 65 else COLORS['primary']
            ax_heat.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7, 
                        color=color, fontweight='bold')
    
    sham_short = {s: s.replace('_', '\n')[:12] for s in shams}
    ax_heat.set_xticks(np.arange(len(shams)))
    ax_heat.set_xticklabels([sham_short[s] for s in shams], rotation=45, ha='right', fontsize=6)
    ax_heat.set_yticks(np.arange(len(models)))
    ax_heat.set_yticklabels(models, fontsize=7)
    
    # Top marginal (sham averages)
    sham_avgs = matrix.mean(axis=0)
    ax_top.bar(np.arange(len(shams)), sham_avgs, color=COLORS['tertiary'], width=0.7)
    ax_top.set_ylim(30, 70)
    ax_top.axhline(y=50, color=COLORS['light'], linestyle='--', linewidth=0.5)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_ylabel('Avg %', fontsize=7)
    ax_top.tick_params(axis='y', labelsize=6)
    
    # Right marginal (model averages)
    model_avgs = matrix.mean(axis=1)
    ax_right.barh(np.arange(len(models)), model_avgs, color=COLORS['tertiary'], height=0.7)
    ax_right.set_xlim(45, 70)
    ax_right.axvline(x=50, color=COLORS['light'], linestyle='--', linewidth=0.5)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.set_xlabel('Avg %', fontsize=7)
    ax_right.tick_params(axis='x', labelsize=6)
    
    # Colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.5])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Detection Accuracy (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    plt.savefig(OUTPUT_DIR / 'Figure2_heatmap.png')
    plt.savefig(OUTPUT_DIR / 'Figure2_heatmap.pdf')
    plt.close()
    print("✓ Figure 2: Heatmap with Marginals")


# ============================================================================
# FIGURE 3: Failure Mechanism Analysis
# ============================================================================

def create_figure3(results):
    """
    3-panel figure analyzing WHY LLMs fail
    """
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, wspace=0.4)
    
    failures = [r for r in results if not r.get('selected_tool_correct') and r.get('model_decision')]
    
    # Panel A: Semantic blindness pie
    ax_a = fig.add_subplot(gs[0])
    ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes, fontsize=12, fontweight='bold')
    
    identical_pattern = r'identical|same|equivalent|no differ'
    identical = sum(1 for f in failures 
                   if re.search(identical_pattern, f['model_decision'].get('trust_rationale', '').lower()))
    other = len(failures) - identical
    
    wedges, _, autotexts = ax_a.pie([identical, other], 
                                     colors=[COLORS['primary'], COLORS['pale']],
                                     autopct='%1.0f%%', startangle=90,
                                     pctdistance=0.75,
                                     wedgeprops=dict(width=0.4, edgecolor='white'))
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    ax_a.set_title('Semantic Blindness:\n"Identical Content" Claims', fontsize=9)
    ax_a.legend(['Claimed Identical', 'Other Reason'], loc='lower center', 
                bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=7)
    
    # Panel B: Confidence comparison boxplot
    ax_b = fig.add_subplot(gs[1])
    ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes, fontsize=12, fontweight='bold')
    
    correct_conf = [r['model_decision']['confidence'] for r in results 
                   if r.get('selected_tool_correct') and r.get('model_decision') 
                   and r['model_decision'].get('confidence') is not None]
    incorrect_conf = [r['model_decision']['confidence'] for r in results 
                     if not r.get('selected_tool_correct') and r.get('model_decision')
                     and r['model_decision'].get('confidence') is not None]
    
    bp = ax_b.boxplot([correct_conf, incorrect_conf], positions=[0, 1], widths=0.4,
                      patch_artist=True, showfliers=False)
    
    bp['boxes'][0].set_facecolor(COLORS['pale'])
    bp['boxes'][1].set_facecolor(COLORS['primary'])
    for box in bp['boxes']:
        box.set_edgecolor(COLORS['secondary'])
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(['Correct', 'Incorrect'])
    ax_b.set_ylabel('Model Confidence')
    ax_b.set_ylim(0.3, 1.0)
    
    # Stats
    t_stat, p_val = stats.ttest_ind(correct_conf, incorrect_conf)
    ax_b.text(0.5, 0.95, f'p = {p_val:.4f}', transform=ax_b.transAxes, 
             ha='center', fontsize=7, style='italic', color=COLORS['tertiary'])
    ax_b.set_title('Overconfidence:\nConfidence by Outcome', fontsize=9)
    
    # Panel C: Position effect by model
    ax_c = fig.add_subplot(gs[2])
    ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes, fontsize=12, fontweight='bold')
    
    models = sorted(set(r['model'] for r in results))
    model_bias = []
    for model in models:
        model_valid = [r for r in results if r['model'] == model and r.get('model_decision')]
        a = sum(1 for r in model_valid if r['model_decision'].get('selected_tool') == 'A')
        model_bias.append((model, a/len(model_valid)*100 if model_valid else 50))
    model_bias.sort(key=lambda x: -x[1])
    
    y_pos = np.arange(len(model_bias))
    rates = [m[1] for m in model_bias]
    
    bars = ax_c.barh(y_pos, rates, color=COLORS['tertiary'], height=0.6, edgecolor='white')
    ax_c.axvline(x=50, color=COLORS['light'], linestyle='--', linewidth=0.8)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels([m[0] for m in model_bias], fontsize=7)
    ax_c.set_xlabel('First-Option Selection (%)')
    ax_c.set_xlim(50, 100)
    ax_c.invert_yaxis()
    ax_c.set_title('Position Bias by Model', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Figure3_mechanisms.png')
    plt.savefig(OUTPUT_DIR / 'Figure3_mechanisms.pdf')
    plt.close()
    print("✓ Figure 3: Failure Mechanisms (3-panel)")


# ============================================================================
# FIGURE 4: Safety-Critical Analysis
# ============================================================================

def create_figure4(results):
    """
    2-panel focusing on patient safety implications
    """
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)
    
    # Safety-critical traps
    safety_traps = ['allergy_ignorance', 'dosing_error', 'contraindication_violation', 'missing_warning']
    
    # Panel A: Safety vs non-safety comparison
    ax_a = fig.add_subplot(gs[0])
    ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes, fontsize=12, fontweight='bold')
    
    safety_results = [r for r in results if r.get('sham_trap_type') in safety_traps]
    other_results = [r for r in results if r.get('sham_trap_type') not in safety_traps]
    
    safety_fails = sum(1 for r in safety_results if not r.get('selected_tool_correct'))
    other_fails = sum(1 for r in other_results if not r.get('selected_tool_correct'))
    
    x = np.arange(2)
    bars = ax_a.bar(x, [safety_fails/len(safety_results)*100, other_fails/len(other_results)*100],
                   color=[COLORS['primary'], COLORS['pale']], width=0.5, edgecolor='white')
    
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(['Safety-Critical\nTraps', 'Other\nTraps'])
    ax_a.set_ylabel('Failure Rate (%)')
    ax_a.set_ylim(0, 60)
    ax_a.axhline(y=50, color=COLORS['light'], linestyle='--', linewidth=0.8)
    
    for bar in bars:
        ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=8, fontweight='bold',
                 color=COLORS['primary'])
    
    ax_a.set_title('Safety-Critical vs Other Traps')
    
    # Panel B: Individual safety trap breakdown
    ax_b = fig.add_subplot(gs[1])
    ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes, fontsize=12, fontweight='bold')
    
    safety_stats = []
    labels = {'allergy_ignorance': 'Allergy', 'dosing_error': 'Dosing',
              'contraindication_violation': 'Contra.', 'missing_warning': 'Warning'}
    for trap in safety_traps:
        trap_results = [r for r in results if r.get('sham_trap_type') == trap]
        if trap_results:
            fails = sum(1 for r in trap_results if not r.get('selected_tool_correct'))
            safety_stats.append((labels[trap], fails/len(trap_results)*100, len(trap_results)))
    
    safety_stats.sort(key=lambda x: -x[1])
    
    y_pos = np.arange(len(safety_stats))
    rates = [s[1] for s in safety_stats]
    
    bars_b = ax_b.barh(y_pos, rates, color=COLORS['primary'], height=0.5, edgecolor='white')
    ax_b.axvline(x=50, color=COLORS['light'], linestyle='--', linewidth=0.8)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([s[0] for s in safety_stats])
    ax_b.set_xlabel('LLM Selected Unsafe Option (%)')
    ax_b.set_xlim(0, 65)
    ax_b.invert_yaxis()
    
    for i, (bar, n) in enumerate(zip(bars_b, [s[2] for s in safety_stats])):
        ax_b.text(bar.get_width() + 1, i, f'n={n}', va='center', fontsize=6, color=COLORS['tertiary'])
    
    ax_b.set_title('Safety-Critical Trap Breakdown')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Figure4_safety.png')
    plt.savefig(OUTPUT_DIR / 'Figure4_safety.pdf')
    plt.close()
    print("✓ Figure 4: Safety Impact (2-panel)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Nature Medicine Quality Figures (600 DPI)")
    print("=" * 60)
    
    results = load_all_results()
    print(f"Loaded {len(results):,} evaluations\n")
    
    create_figure1(results)
    create_figure2(results)
    create_figure3(results)
    create_figure4(results)
    
    print(f"\n{'=' * 60}")
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
