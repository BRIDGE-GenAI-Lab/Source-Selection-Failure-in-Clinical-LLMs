#!/usr/bin/env python3
"""
Nature Medicine Publication-Ready Figures
Clean, professional styling with grayish-bluish muted palette
Comprehensive appendix tables

3,000 evaluations across 6 LLMs
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path('publication_figures')
OUTPUT_DIR.mkdir(exist_ok=True)

# Nature Medicine palette - grayish-bluish muted tones
PALETTE = {
    'dark_blue': '#3C4F76',
    'medium_blue': '#5B7398', 
    'light_blue': '#8CA0B8',
    'pale_blue': '#B8C6D6',
    'slate': '#6B7D8F',
    'charcoal': '#4A5568',
    'warm_gray': '#8B8C89',
    'muted_red': '#C16A6A',
    'muted_green': '#6B9080',
    'muted_gold': '#C4A962',
}

# Clean figure style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
})

EXPERIMENTS = {
    "GPT-4.1": "results/run_20260107_215921",
    "DeepSeek-V3.2": "results/run_20260107_183414",
    "GPT-5-Nano": "results/run_20260107_085822",
    "GPT-4o-Mini": "results/run_20260107_142734",
    "Gemini-2.5-Flash": "results/run_20260107_181625",
    "GPT-4.1-Nano": "results/run_20260107_142701",
}

SHAM_DISPLAY = {
    'missing_warning': 'Missing Warning',
    'allergy_ignorance': 'Allergy Ignorance',
    'dosing_error': 'Dosing Error',
    'wrong_population': 'Wrong Population',
    'contraindication_violation': 'Contraindication Violation',
    'authority_mimicry': 'Authority Mimicry',
    'subtle_inversion': 'Subtle Inversion',
    'prompt_injection': 'Prompt Injection',
    'fabricated_citation': 'Fabricated Citation',
    'outdated_version': 'Outdated Version',
}


def load_all_results():
    """Load and pool all results, excluding unknown sham types."""
    results = []
    for name, path in EXPERIMENTS.items():
        with open(Path(path) / "results.jsonl") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line.strip())
                    r['model'] = name
                    # Exclude unknown
                    if r.get('sham_trap_type', 'unknown') != 'unknown':
                        results.append(r)
    return results


def wilson_ci(successes, n, z=1.96):
    """Wilson score confidence interval."""
    if n == 0:
        return 0, 0
    p = successes / n
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n)) / denominator
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
    return max(0, center - margin), min(1, center + margin)


# ============================================================================
# FIGURE 1: Trap Effectiveness (Clean bar chart)
# ============================================================================

def create_figure1(results):
    """Which sham modifications are most effective at fooling LLMs?"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate failure rates by trap type
    trap_stats = defaultdict(lambda: {'fail': 0, 'total': 0})
    for r in results:
        sham = r.get('sham_trap_type')
        trap_stats[sham]['total'] += 1
        if not r.get('selected_tool_correct'):
            trap_stats[sham]['fail'] += 1
    
    # Sort by failure rate
    sorted_traps = sorted(trap_stats.items(), 
                         key=lambda x: x[1]['fail']/x[1]['total'], reverse=True)
    
    names = [SHAM_DISPLAY.get(t[0], t[0]) for t in sorted_traps]
    rates = [t[1]['fail']/t[1]['total']*100 for t in sorted_traps]
    totals = [t[1]['total'] for t in sorted_traps]
    
    # Color gradient from red to blue based on danger
    colors = [PALETTE['muted_red'] if r > 50 else PALETTE['slate'] if r > 40 else PALETTE['muted_green'] 
              for r in rates]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, rates, color=colors, height=0.7, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('LLM Failure Rate (%)')
    ax.set_xlim(0, 68)
    ax.axvline(x=50, color=PALETTE['warm_gray'], linestyle='--', linewidth=1, alpha=0.7)
    ax.invert_yaxis()
    
    # Add n labels
    for i, (bar, n) in enumerate(zip(bars, totals)):
        ax.text(bar.get_width() + 1, i, f'n={n}', va='center', fontsize=8, color=PALETTE['charcoal'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure1_trap_effectiveness.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure1_trap_effectiveness.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Trap Effectiveness")


# ============================================================================
# FIGURE 2: Position Bias (Clean 2-panel)
# ============================================================================

def create_figure2(results):
    """Position bias analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Filter for valid decisions
    valid = [r for r in results if r.get('model_decision')]
    
    # Panel A: Overall selection distribution
    picks_a = sum(1 for r in valid if r['model_decision'].get('selected_tool') == 'A')
    picks_b = sum(1 for r in valid if r['model_decision'].get('selected_tool') == 'B')
    
    ax1.bar(['First Tool (A)', 'Second Tool (B)'], [picks_a, picks_b], 
           color=[PALETTE['muted_red'], PALETTE['medium_blue']], width=0.5, edgecolor='white')
    ax1.axhline(y=(picks_a+picks_b)/2, color=PALETTE['warm_gray'], linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Number of Selections')
    ax1.set_ylim(0, max(picks_a, picks_b) * 1.15)
    
    # Add percentage
    for i, (val, pct) in enumerate([(picks_a, picks_a/(picks_a+picks_b)*100), 
                                     (picks_b, picks_b/(picks_a+picks_b)*100)]):
        ax1.text(i, val + 50, f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold', color=PALETTE['charcoal'])
    
    # Panel B: Accuracy by sham position
    a_is_sham = [r for r in results if r.get('mapping', {}).get('A') == 'S']
    b_is_sham = [r for r in results if r.get('mapping', {}).get('B') == 'S']
    
    acc_a = sum(1 for r in a_is_sham if r.get('selected_tool_correct'))/len(a_is_sham)*100
    acc_b = sum(1 for r in b_is_sham if r.get('selected_tool_correct'))/len(b_is_sham)*100
    
    bars = ax2.bar(['Sham First (A)', 'Sham Second (B)'], [acc_a, acc_b],
                   color=[PALETTE['light_blue'], PALETTE['dark_blue']], width=0.5, edgecolor='white')
    ax2.axhline(y=50, color=PALETTE['warm_gray'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Detection Accuracy (%)')
    ax2.set_ylim(0, 75)
    
    for i, bar in enumerate(bars):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{[acc_a, acc_b][i]:.1f}%', ha='center', fontsize=10, fontweight='bold', color=PALETTE['charcoal'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure2_position_bias.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure2_position_bias.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Position Bias")


# ============================================================================
# FIGURE 3: Failure Reasoning Patterns
# ============================================================================

def create_figure3(results):
    """What reasoning leads to failures?"""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    failures = [r for r in results if not r.get('selected_tool_correct') and r.get('model_decision')]
    
    patterns = {
        'Perceived as Identical': r'identical|same content|equivalent|both .* same',
        'More Complete/Detailed': r'more complete|additional|extra|includes more',
        'Trusted Source': r'trusted|authoritative|credible|reliable',
        'Better Formatting': r'organized|structured|clear|formatted',
        'Default/Tie-break': r'tie.?break|default|arbitrary',
    }
    
    pattern_counts = []
    for name, regex in patterns.items():
        count = sum(1 for r in failures if re.search(regex, r['model_decision'].get('trust_rationale', '').lower()))
        pattern_counts.append((name, count, count/len(failures)*100))
    
    pattern_counts.sort(key=lambda x: -x[2])
    
    names = [p[0] for p in pattern_counts]
    pcts = [p[2] for p in pattern_counts]
    
    colors = [PALETTE['dark_blue'] if i == 0 else PALETTE['medium_blue'] if i < 3 else PALETTE['light_blue'] 
              for i in range(len(names))]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, pcts, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('% of Failures')
    ax.set_xlim(0, 90)
    ax.invert_yaxis()
    
    for i, (bar, pct) in enumerate(zip(bars, pcts)):
        ax.text(bar.get_width() + 1, i, f'{pct:.0f}%', va='center', fontsize=9, color=PALETTE['charcoal'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure3_failure_patterns.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure3_failure_patterns.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Failure Patterns")


# ============================================================================
# FIGURE 4: Confidence Calibration
# ============================================================================

def create_figure4(results):
    """Are LLMs overconfident?"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    correct_conf = [r['model_decision']['confidence'] for r in results 
                   if r.get('selected_tool_correct') and r.get('model_decision') 
                   and r['model_decision'].get('confidence') is not None]
    incorrect_conf = [r['model_decision']['confidence'] for r in results 
                     if not r.get('selected_tool_correct') and r.get('model_decision')
                     and r['model_decision'].get('confidence') is not None]
    
    # Box plots
    bp = ax.boxplot([correct_conf, incorrect_conf], positions=[1, 2], widths=0.5,
                    patch_artist=True, showfliers=False)
    
    bp['boxes'][0].set_facecolor(PALETTE['muted_green'])
    bp['boxes'][1].set_facecolor(PALETTE['muted_red'])
    for box in bp['boxes']:
        box.set_edgecolor(PALETTE['charcoal'])
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Correct\nDecisions', 'Incorrect\nDecisions'])
    ax.set_ylabel('Model Confidence')
    ax.set_ylim(0.3, 1.0)
    
    # Add means
    mean_c = np.mean(correct_conf)
    mean_i = np.mean(incorrect_conf)
    ax.text(1, mean_c + 0.02, f'μ={mean_c:.2f}', ha='center', fontsize=9, color=PALETTE['charcoal'])
    ax.text(2, mean_i + 0.02, f'μ={mean_i:.2f}', ha='center', fontsize=9, color=PALETTE['charcoal'])
    
    # T-test
    t_stat, p_val = stats.ttest_ind(correct_conf, incorrect_conf)
    ax.text(0.95, 0.95, f'p = {p_val:.3f}', transform=ax.transAxes, ha='right', va='top', 
           fontsize=9, style='italic', color=PALETTE['charcoal'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure4_confidence.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure4_confidence.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: Confidence")


# ============================================================================
# APPENDIX TABLES
# ============================================================================

def create_appendix_tables(results):
    """Generate comprehensive appendix tables."""
    
    # Table A1: Complete sham type breakdown
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE A1. Detection Accuracy by Sham Trap Type (Pooled Across 6 Models)")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'Sham Type':<30} {'N':>8} {'Detected':>10} {'Missed':>10} {'Accuracy':>12} {'95% CI':>18}")
    lines.append("-" * 100)
    
    trap_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        sham = r.get('sham_trap_type')
        trap_stats[sham]['total'] += 1
        if r.get('selected_tool_correct'):
            trap_stats[sham]['correct'] += 1
    
    for sham in sorted(trap_stats.keys(), key=lambda x: trap_stats[x]['correct']/trap_stats[x]['total'], reverse=True):
        data = trap_stats[sham]
        acc = data['correct']/data['total']
        ci_low, ci_high = wilson_ci(data['correct'], data['total'])
        display = SHAM_DISPLAY.get(sham, sham)
        lines.append(f"{display:<30} {data['total']:>8} {data['correct']:>10} {data['total']-data['correct']:>10} "
                    f"{acc*100:>11.1f}% [{ci_low*100:>5.1f}%, {ci_high*100:>5.1f}%]")
    
    lines.append("-" * 100)
    total = len(results)
    correct = sum(1 for r in results if r.get('selected_tool_correct'))
    acc = correct/total
    ci_low, ci_high = wilson_ci(correct, total)
    lines.append(f"{'TOTAL':<30} {total:>8} {correct:>10} {total-correct:>10} "
                f"{acc*100:>11.1f}% [{ci_low*100:>5.1f}%, {ci_high*100:>5.1f}%]")
    lines.append("=" * 100)
    
    with open(OUTPUT_DIR / 'table_a1_sham_types.txt', 'w') as f:
        f.write("\n".join(lines))
    print("✓ Table A1: Sham Types")
    
    # Table A2: Model-by-model breakdown
    lines = []
    lines.append("=" * 90)
    lines.append("TABLE A2. Detection Accuracy by Model")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"{'Model':<25} {'N':>8} {'Correct':>10} {'Accuracy':>12} {'95% CI':>18} {'Pos. Bias':>12}")
    lines.append("-" * 90)
    
    model_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'a': 0, 'b': 0})
    for r in results:
        model = r.get('model')
        model_stats[model]['total'] += 1
        if r.get('selected_tool_correct'):
            model_stats[model]['correct'] += 1
        if r.get('model_decision'):
            if r['model_decision'].get('selected_tool') == 'A':
                model_stats[model]['a'] += 1
            else:
                model_stats[model]['b'] += 1
    
    for model in sorted(model_stats.keys(), key=lambda x: model_stats[x]['correct']/model_stats[x]['total'], reverse=True):
        data = model_stats[model]
        acc = data['correct']/data['total']
        ci_low, ci_high = wilson_ci(data['correct'], data['total'])
        pos_bias = data['a']/(data['a']+data['b'])*100 if (data['a']+data['b']) > 0 else 50
        lines.append(f"{model:<25} {data['total']:>8} {data['correct']:>10} "
                    f"{acc*100:>11.1f}% [{ci_low*100:>5.1f}%, {ci_high*100:>5.1f}%] {pos_bias:>10.0f}% A")
    
    lines.append("=" * 90)
    
    with open(OUTPUT_DIR / 'table_a2_models.txt', 'w') as f:
        f.write("\n".join(lines))
    print("✓ Table A2: Models")
    
    # Table A3: Cross-tabulation Model x Sham Type
    lines = []
    lines.append("=" * 140)
    lines.append("TABLE A3. Detection Accuracy (%) by Model and Sham Type")
    lines.append("=" * 140)
    lines.append("")
    
    models = sorted(set(r['model'] for r in results))
    shams = sorted(set(r['sham_trap_type'] for r in results))
    
    header = f"{'Sham Type':<25}"
    for m in models:
        header += f" {m[:10]:>10}"
    header += f" {'Mean':>8}"
    lines.append(header)
    lines.append("-" * 140)
    
    for sham in shams:
        row = f"{SHAM_DISPLAY.get(sham, sham)[:24]:<25}"
        accs = []
        for model in models:
            sham_model = [r for r in results if r['sham_trap_type'] == sham and r['model'] == model]
            if sham_model:
                acc = sum(1 for r in sham_model if r.get('selected_tool_correct'))/len(sham_model)*100
                accs.append(acc)
                row += f" {acc:>9.0f}%"
            else:
                row += f" {'N/A':>10}"
        if accs:
            row += f" {np.mean(accs):>7.1f}%"
        lines.append(row)
    
    lines.append("=" * 140)
    
    with open(OUTPUT_DIR / 'table_a3_cross_tab.txt', 'w') as f:
        f.write("\n".join(lines))
    print("✓ Table A3: Cross-tabulation")
    
    # Table A4: Position bias detailed
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE A4. Position Bias Analysis by Model")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Model':<25} {'Tool A':>12} {'Tool B':>12} {'Bias':>12} {'p-value':>15}")
    lines.append("-" * 80)
    
    for model in models:
        model_results = [r for r in results if r['model'] == model and r.get('model_decision')]
        a = sum(1 for r in model_results if r['model_decision'].get('selected_tool') == 'A')
        b = sum(1 for r in model_results if r['model_decision'].get('selected_tool') == 'B')
        total = a + b
        if total > 0:
            from scipy.stats import binomtest
            result = binomtest(a, total, p=0.5)
            p_val = result.pvalue
            lines.append(f"{model:<25} {a:>11} ({a/total*100:.0f}%) {b:>6} ({b/total*100:.0f}%) {(a/total-0.5)*200:>+10.1f}% {p_val:>14.2e}")
    
    lines.append("=" * 80)
    
    with open(OUTPUT_DIR / 'table_a4_position_bias.txt', 'w') as f:
        f.write("\n".join(lines))
    print("✓ Table A4: Position Bias")
    
    # Table A5: Failure mode examples
    lines = []
    lines.append("=" * 120)
    lines.append("TABLE A5. Representative Failure Cases by Sham Type")
    lines.append("=" * 120)
    
    failures = [r for r in results if not r.get('selected_tool_correct') and r.get('model_decision')]
    
    for sham in ['missing_warning', 'allergy_ignorance', 'dosing_error', 'wrong_population']:
        sham_failures = [r for r in failures if r['sham_trap_type'] == sham][:2]
        lines.append("")
        lines.append(f"--- {SHAM_DISPLAY.get(sham, sham).upper()} ---")
        for r in sham_failures:
            lines.append(f"  Case: {r['case_id']} | Model: {r['model']} | Confidence: {r['model_decision'].get('confidence', 'N/A')}")
            rationale = r['model_decision'].get('trust_rationale', '')[:150]
            lines.append(f"  Rationale: \"{rationale}...\"")
    
    lines.append("")
    lines.append("=" * 120)
    
    with open(OUTPUT_DIR / 'table_a5_failure_examples.txt', 'w') as f:
        f.write("\n".join(lines))
    print("✓ Table A5: Failure Examples")
    
    # Table A6: Summary statistics
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE A6. Summary Statistics")
    lines.append("=" * 80)
    lines.append("")
    lines.append("STUDY OVERVIEW")
    lines.append(f"  Total evaluations:              {len(results):,}")
    lines.append(f"  Number of models:               6")
    lines.append(f"  Cases per model:                500")
    lines.append(f"  Sham trap types:                10")
    lines.append("")
    lines.append("OVERALL PERFORMANCE")
    correct = sum(1 for r in results if r.get('selected_tool_correct'))
    lines.append(f"  Correct detections:             {correct:,} ({correct/len(results)*100:.1f}%)")
    lines.append(f"  Failed detections:              {len(results)-correct:,} ({(len(results)-correct)/len(results)*100:.1f}%)")
    lines.append("")
    lines.append("POSITION BIAS")
    valid = [r for r in results if r.get('model_decision')]
    a = sum(1 for r in valid if r['model_decision'].get('selected_tool') == 'A')
    b = len(valid) - a
    lines.append(f"  Tool A (first) selections:      {a:,} ({a/len(valid)*100:.1f}%)")
    lines.append(f"  Tool B (second) selections:     {b:,} ({b/len(valid)*100:.1f}%)")
    lines.append(f"  Excess bias:                    {(a/len(valid)-0.5)*200:.1f}%")
    lines.append("")
    lines.append("CONFIDENCE")
    correct_conf = [r['model_decision']['confidence'] for r in results 
                   if r.get('selected_tool_correct') and r.get('model_decision') 
                   and r['model_decision'].get('confidence') is not None]
    incorrect_conf = [r['model_decision']['confidence'] for r in results 
                     if not r.get('selected_tool_correct') and r.get('model_decision')
                     and r['model_decision'].get('confidence') is not None]
    if correct_conf and incorrect_conf:
        t_stat, p_val = stats.ttest_ind(correct_conf, incorrect_conf)
        lines.append(f"  Mean confidence (correct):      {np.mean(correct_conf):.3f} ± {np.std(correct_conf):.3f}")
        lines.append(f"  Mean confidence (incorrect):    {np.mean(incorrect_conf):.3f} ± {np.std(incorrect_conf):.3f}")
        lines.append(f"  t-test p-value:                 {p_val:.4f}")
    lines.append("")
    lines.append("=" * 80)
    
    with open(OUTPUT_DIR / 'table_a6_summary.txt', 'w') as f:
        f.write("\n".join(lines))
    print("✓ Table A6: Summary")


# ============================================================================
# SUPPLEMENTARY FIGURES
# ============================================================================

def create_supplementary_figures(results):
    """Additional supplementary figures."""
    
    # Figure S1: Heatmap Model x Sham
    fig, ax = plt.subplots(figsize=(12, 5))
    
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
    
    # Custom colormap (bluish)
    from matplotlib.colors import LinearSegmentedColormap
    colors_map = [PALETTE['muted_red'], PALETTE['warm_gray'], PALETTE['muted_green']]
    cmap = LinearSegmentedColormap.from_list('custom', colors_map)
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=20, vmax=100)
    
    for i in range(len(models)):
        for j in range(len(shams)):
            val = matrix[i, j]
            color = 'white' if val < 35 or val > 80 else PALETTE['charcoal']
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8, color=color)
    
    ax.set_xticks(np.arange(len(shams)))
    ax.set_xticklabels([SHAM_DISPLAY.get(s, s)[:12] for s in shams], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_s1_heatmap.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure_s1_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure S1: Heatmap")
    
    # Figure S2: Model comparison bar
    fig, ax = plt.subplots(figsize=(8, 4))
    
    model_accs = []
    for model in models:
        model_results = [r for r in results if r['model'] == model]
        acc = sum(1 for r in model_results if r.get('selected_tool_correct'))/len(model_results)*100
        ci_low, ci_high = wilson_ci(sum(1 for r in model_results if r.get('selected_tool_correct')), len(model_results))
        model_accs.append((model, acc, ci_low*100, ci_high*100))
    
    model_accs.sort(key=lambda x: -x[1])
    
    x_pos = np.arange(len(model_accs))
    accs = [m[1] for m in model_accs]
    yerr_low = [m[1] - m[2] for m in model_accs]
    yerr_high = [m[3] - m[1] for m in model_accs]
    
    bars = ax.bar(x_pos, accs, color=PALETTE['medium_blue'], edgecolor='white', width=0.6)
    ax.errorbar(x_pos, accs, yerr=[yerr_low, yerr_high], fmt='none', color=PALETTE['charcoal'], capsize=4)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m[0] for m in model_accs], rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Detection Accuracy (%)')
    ax.set_ylim(40, 75)
    ax.axhline(y=50, color=PALETTE['warm_gray'], linestyle='--', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_s2_model_comparison.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure_s2_model_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure S2: Model Comparison")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Nature Medicine Publication Figures")
    print("=" * 60)
    print()
    
    print("Loading results (excluding unknown)...")
    results = load_all_results()
    print(f"Loaded {len(results):,} evaluations\n")
    
    print("Generating main figures...")
    create_figure1(results)
    create_figure2(results)
    create_figure3(results)
    create_figure4(results)
    
    print("\nGenerating appendix tables...")
    create_appendix_tables(results)
    
    print("\nGenerating supplementary figures...")
    create_supplementary_figures(results)
    
    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
