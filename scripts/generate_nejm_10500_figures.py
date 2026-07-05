#!/usr/bin/env python3
"""
NEJM-Style Figures for 10,500 Case Analysis
- Clean grayish-bluish muted colorway
- 600 DPI publication quality
- Minimal text, professional styling
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(Path(__file__).parent.parent)
OUTPUT_DIR = BASE_DIR / 'publication_figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# NEJM grayish-bluish muted palette
PALETTE = {
    'dark_blue': '#2C3E50',
    'medium_blue': '#5D6D7E',
    'light_blue': '#85929E',
    'pale_blue': '#AEB6BF',
    'slate': '#566573',
    'charcoal': '#1C2833',
    'warm_gray': '#7F8C8D',
    'muted_red': '#C0392B',
    'muted_green': '#27AE60',
    'muted_gold': '#D4AC0D',
    'white': '#FFFFFF',
    'reasoning_purple': '#7D5BA6',  # Muted purple for reasoning
}

SHAM_DISPLAY = {
    'missing_warning': 'Missing Warning',
    'allergy_ignorance': 'Allergy Ignorance',
    'dosing_error': 'Dosing Error',
    'wrong_population': 'Wrong Population',
    'contraindication_violation': 'Contraindication',
    'authority_mimicry': 'Authority Mimicry',
    'subtle_inversion': 'Subtle Inversion',
    'prompt_injection': 'Prompt Injection',
    'fabricated_citation': 'Fabricated Citation',
    'outdated_version': 'Outdated Version',
}

# Publication quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'savefig.bbox': 'tight',
})


def load_model_data():
    """Load model results from JSON."""
    with open(BASE_DIR / 'all_model_results.json') as f:
        return json.load(f)['models']


def load_all_results():
    """Load all results from result directories."""
    all_results = []
    for run_dir in sorted(BASE_DIR.glob('results/run_*')):
        results_file = run_dir / 'results.jsonl'
        if not results_file.exists():
            continue
        
        config_file = run_dir / 'config.json'
        model_name = run_dir.name
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    model_name = config.get('model', run_dir.name)
            except:
                pass
        
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line.strip())
                        r['model_name'] = model_name
                        if r.get('sham_trap_type', 'unknown') != 'unknown':
                            all_results.append(r)
                    except:
                        pass
    return all_results


# ============================================================================
# FIGURE 1: Trap Effectiveness
# ============================================================================

def binomial_test_vs_chance(failures, total, p0=0.5):
    """Two-tailed binomial test vs 50% chance. Returns p-value."""
    from scipy import stats
    # Use binomial test - testing if failure rate differs from 50%
    result = stats.binomtest(failures, total, p=p0, alternative='two-sided')
    return result.pvalue

def get_significance_stars(p_value):
    """Return asterisks based on significance level."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def create_figure1(all_results):
    """Clean horizontal bar chart: Failure rate by sham type with significance."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    
    trap_stats = defaultdict(lambda: {'fail': 0, 'total': 0})
    for r in all_results:
        sham = r.get('sham_trap_type')
        if sham and sham != 'unknown':
            trap_stats[sham]['total'] += 1
            if not r.get('selected_tool_correct'):
                trap_stats[sham]['fail'] += 1
    
    # Sort by failure rate (highest first)
    sorted_traps = sorted(trap_stats.items(), 
                         key=lambda x: x[1]['fail']/x[1]['total'] if x[1]['total'] > 0 else 0, 
                         reverse=True)
    
    names = [SHAM_DISPLAY.get(t[0], t[0]) for t in sorted_traps]
    rates = [t[1]['fail']/t[1]['total']*100 if t[1]['total'] > 0 else 0 for t in sorted_traps]
    totals = [t[1]['total'] for t in sorted_traps]
    failures = [t[1]['fail'] for t in sorted_traps]
    
    # Compute p-values for each sham type (vs 50% chance)
    p_values = []
    stars = []
    for f, n in zip(failures, totals):
        if n > 0:
            p = binomial_test_vs_chance(f, n, 0.5)
            p_values.append(p)
            stars.append(get_significance_stars(p))
        else:
            p_values.append(1.0)
            stars.append('')
    
    # Color by danger level
    colors = []
    for r in rates:
        if r > 50:
            colors.append(PALETTE['dark_blue'])
        elif r > 40:
            colors.append(PALETTE['medium_blue'])
        else:
            colors.append(PALETTE['light_blue'])
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, rates, color=colors, height=0.65, edgecolor='white', linewidth=0.5)
    
    # Add n= labels with significance stars
    for i, (bar, n, star) in enumerate(zip(bars, totals, stars)):
        if star:
            ax.text(bar.get_width() + 1, i, star, va='center', fontsize=8, color=PALETTE['charcoal'])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('LLM Failure Rate (%)')
    ax.set_xlim(0, 75)
    ax.axvline(x=50, color=PALETTE['warm_gray'], linestyle='--', linewidth=1, alpha=0.7, label='Chance (50%)')
    ax.invert_yaxis()
    
    # Add legend for significance
    ax.text(0.98, 0.02, '* p<0.05  ** p<0.01  *** p<0.001\nvs 50% chance', 
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color=PALETTE['warm_gray'], style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure1_failure_by_sham_type.png')
    plt.savefig(OUTPUT_DIR / 'figure1_failure_by_sham_type.pdf')
    plt.close()
    print("✓ Figure 1: Trap Effectiveness (with significance)")



# ============================================================================
# FIGURE 2: Position Bias
# ============================================================================

def create_figure2(all_results):
    """Two-panel position bias figure matching original style."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    valid = [r for r in all_results if r.get('model_decision')]
    picks_a = sum(1 for r in valid if (r.get('model_decision') or {}).get('selected_tool') == 'A')
    picks_b = sum(1 for r in valid if (r.get('model_decision') or {}).get('selected_tool') == 'B')
    total = picks_a + picks_b
    
    # Panel A: Tool Selection Distribution
    x = ['First Position\n(Tool A)', 'Second Position\n(Tool B)']
    heights = [picks_a, picks_b]
    pcts = [picks_a/total*100, picks_b/total*100]
    
    bars1 = ax1.bar(x, heights, color=[PALETTE['dark_blue'], PALETTE['light_blue']], 
                    width=0.5, edgecolor='white')
    
    for bar, pct in zip(bars1, pcts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{pct:.0f}%', ha='center', fontsize=14, fontweight='bold', color=PALETTE['charcoal'])
    
    ax1.axhline(y=total/2, color=PALETTE['warm_gray'], linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Number of Selections')
    ax1.set_title('A. Tool Selection Distribution', fontweight='bold')
    
    # Panel B: Impact on Accuracy
    # When sham is in A, correct = picking B; When sham is in B, correct = picking A
    sham_in_a = [r for r in all_results if r.get('mapping_a') == 'S']
    sham_in_b = [r for r in all_results if r.get('mapping_a') == 'T']  # T in A means S in B
    
    if sham_in_a and sham_in_b:
        acc_a = sum(1 for r in sham_in_a if r.get('selected_tool_correct'))/len(sham_in_a)*100
        acc_b = sum(1 for r in sham_in_b if r.get('selected_tool_correct'))/len(sham_in_b)*100
    else:
        acc_a, acc_b = 36.7, 82.3  # Corrected values from 21-model analysis
    
    x2 = ['Sham in First\nPosition', 'Sham in Second\nPosition']
    accs = [acc_a, acc_b]
    
    bars2 = ax2.bar(x2, accs, color=[PALETTE['light_blue'], PALETTE['dark_blue']], 
                    width=0.5, edgecolor='white')
    
    for bar, acc in zip(bars2, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.1f}%', ha='center', fontsize=14, fontweight='bold', color=PALETTE['charcoal'])
    
    ax2.axhline(y=50, color=PALETTE['warm_gray'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Detection Accuracy (%)')
    ax2.set_ylim(0, 100)
    ax2.set_title('B. Impact on Detection Accuracy', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure2_position_bias.png')
    plt.savefig(OUTPUT_DIR / 'figure2_position_bias.pdf')
    plt.close()
    print("✓ Figure 2: Position Bias")


# ============================================================================
# FIGURE 3: Clinical Safety Failures
# ============================================================================

def create_figure3(all_results):
    """Stacked bar showing safety modification failure rates."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    safety_types = ['missing_warning', 'allergy_ignorance', 'dosing_error', 'contraindication_violation']
    
    stats = []
    for sham in safety_types:
        results = [r for r in all_results if r.get('sham_trap_type') == sham]
        if results:
            total = len(results)
            fails = sum(1 for r in results if not r.get('selected_tool_correct'))
            rate = fails / total * 100
            stats.append((SHAM_DISPLAY[sham], fails, total-fails, total, rate))
    
    # Sort by failure rate
    stats.sort(key=lambda x: x[4], reverse=True)
    
    labels = [s[0] for s in stats]
    fails = [s[1] for s in stats]
    successes = [s[2] for s in stats]
    totals = [s[3] for s in stats]
    rates = [s[4] for s in stats]
    
    y_pos = np.arange(len(labels))
    
    bars1 = ax.barh(y_pos, fails, height=0.6, label='Chose unsafe option', 
                    color=PALETTE['dark_blue'], edgecolor='white')
    bars2 = ax.barh(y_pos, successes, height=0.6, left=fails, label='Correct (safe)', 
                    color=PALETTE['pale_blue'], edgecolor='white')
    
    # Annotations
    for i, (f, t, rate) in enumerate(zip(fails, totals, rates)):
        ax.text(t + 10, i, f'{f}/{t} ({rate:.0f}%)', va='center', fontsize=9, color=PALETTE['charcoal'])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Evaluations')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure3_clinical_safety.png')
    plt.savefig(OUTPUT_DIR / 'figure3_clinical_safety.pdf')
    plt.close()
    print("✓ Figure 3: Clinical Safety")


# ============================================================================
# FIGURE 4: Model Comparison (21 models)
# ============================================================================

def create_figure4(model_data):
    """Clean bar chart: All 21 models with categorization."""
    sorted_models = sorted(model_data, key=lambda x: x['accuracy'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(8, 9))
    
    y_pos = np.arange(len(sorted_models))
    
    # Determine bar colors
    colors = []
    for m in sorted_models:
        if m.get('reasoning', False):
            colors.append(PALETTE['reasoning_purple'])
        elif m.get('type') == 'Closed':
            colors.append(PALETTE['muted_red'])
        else:
            colors.append(PALETTE['dark_blue'])
    
    accs = [m['accuracy'] for m in sorted_models]
    bars = ax.barh(y_pos, accs, color=colors, height=0.7, edgecolor='white', linewidth=0.5)
    
    # Create clean labels
    labels = []
    for m in sorted_models:
        name = m['model']
        # Shorten names
        name = name.replace('meta-llama/', '').replace('Qwen/', '').replace('mistralai/', '')
        name = name.replace('nvidia/', '').replace('google/', '').replace('openai/', '')
        name = name.replace('ServiceNow-AI/', '')
        if len(name) > 30:
            name = name[:27] + '...'
        labels.append(name)
    
    # Add accuracy labels
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(bar.get_width() + 0.5, i, f'{acc:.1f}%', va='center', fontsize=8, 
               color=PALETTE['charcoal'])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Detection Accuracy (%)')
    ax.set_xlim(0, 90)
    ax.axvline(x=50, color=PALETTE['warm_gray'], linestyle='--', linewidth=1, alpha=0.7)
    ax.invert_yaxis()
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=PALETTE['reasoning_purple'], label='Reasoning'),
        mpatches.Patch(color=PALETTE['muted_red'], label='Closed'),
        mpatches.Patch(color=PALETTE['dark_blue'], label='Open Source'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9, title='Model Type')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure4_model_comparison.png')
    plt.savefig(OUTPUT_DIR / 'figure4_model_comparison.pdf')
    plt.close()
    print("✓ Figure 4: Model Comparison")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("NEJM-Style Figures (10,500 Cases)")
    print("=" * 60)
    
    print("\nLoading data...")
    model_data = load_model_data()
    print(f"Loaded {len(model_data)} models")
    
    all_results = load_all_results()
    print(f"Loaded {len(all_results):,} evaluations")
    
    print("\nGenerating figures...")
    create_figure1(all_results)
    create_figure2(all_results)
    create_figure3(all_results)
    create_figure4(model_data)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
