#!/usr/bin/env python3
"""
Comprehensive publication-quality analysis for Nature Medicine.
Generates all figures, tables, and statistical analyses for the
Safe Guideline Tooling Evaluation study.

6 Models × 500 Cases = 3,000 Total Evaluations
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
from scipy.stats import chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path('publication_figures')
OUTPUT_DIR.mkdir(exist_ok=True)

# NEJM/Nature Medicine color palette
COLORS = {
    'primary': '#4A6FA5',
    'secondary': '#6B8EB8',
    'accent': '#2E4A62',
    'success': '#5D8A66',
    'danger': '#B85C5C',
    'warning': '#C4A35A',
    'neutral': '#8B9BAE',
}

MODEL_COLORS = ['#2E4A62', '#4A6FA5', '#6B8EB8', '#5D8A66', '#C4A35A', '#B85C5C']

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

# All experiments
EXPERIMENTS = {
    "GPT-4.1": "results/run_20260107_215921",
    "DeepSeek-V3.2": "results/run_20260107_183414",
    "GPT-5-Nano": "results/run_20260107_085822",
    "GPT-4o-Mini": "results/run_20260107_142734",
    "Gemini-2.5-Flash": "results/run_20260107_181625",
    "GPT-4.1-Nano": "results/run_20260107_142701",
}

SHAM_TYPES = ['outdated_version', 'fabricated_citation', 'prompt_injection', 
              'subtle_inversion', 'allergy_ignorance', 'contraindication_violation',
              'authority_mimicry', 'wrong_population', 'dosing_error', 'missing_warning']


def load_all_results():
    """Load results from all experiments."""
    all_results = {}
    for name, path in EXPERIMENTS.items():
        results = []
        results_file = Path(path) / "results.jsonl"
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))
        all_results[name] = results
    return all_results


def calculate_accuracy(results):
    """Calculate accuracy from results."""
    correct = sum(1 for r in results if r.get('selected_tool_correct') == True)
    return correct / len(results) * 100 if results else 0


def wilson_ci(p, n, z=1.96):
    """Calculate Wilson score confidence interval."""
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n)) / denominator
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
    return max(0, center - margin), min(1, center + margin)


# ============================================================================
# FIGURE 1: Overall Model Performance
# ============================================================================

def create_figure1(all_results):
    """Main figure: Model comparison with confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate accuracies and CIs
    data = []
    for name, results in all_results.items():
        n = len(results)
        correct = sum(1 for r in results if r.get('selected_tool_correct') == True)
        acc = correct / n
        ci_low, ci_high = wilson_ci(acc, n)
        data.append((name, acc * 100, ci_low * 100, ci_high * 100, n))
    
    # Sort by accuracy
    data.sort(key=lambda x: -x[1])
    
    names = [d[0] for d in data]
    accs = [d[1] for d in data]
    ci_lows = [d[2] for d in data]
    ci_highs = [d[3] for d in data]
    
    yerr_low = [a - l for a, l in zip(accs, ci_lows)]
    yerr_high = [h - a for a, h in zip(accs, ci_highs)]
    
    bars = ax.bar(names, accs, color=MODEL_COLORS, edgecolor='white', width=0.7)
    ax.errorbar(names, accs, yerr=[yerr_low, yerr_high], fmt='none', 
                color='black', capsize=5, capthick=1.5, linewidth=1.5)
    
    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Chance level')
    ax.set_ylabel('Detection Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 80)
    ax.set_title('Figure 1. LLM Trustworthy Tool Detection Accuracy\n(n=500 cases per model, 95% CI)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10)
    
    # Statistical annotation
    total_n = sum(d[4] for d in data)
    ax.text(0.02, 0.02, f'Total N = {total_n:,} evaluations across 6 models',
            transform=ax.transAxes, fontsize=9, style='italic')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure1_model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'figure1_model_comparison.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 1: Model Comparison")


# ============================================================================
# FIGURE 2: Accuracy by Sham Type Heatmap
# ============================================================================

def create_figure2(all_results):
    """Heatmap of accuracy by model and sham type."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Build matrix
    models = list(all_results.keys())
    models.sort(key=lambda x: -calculate_accuracy(all_results[x]))
    
    matrix = []
    for model in models:
        row = []
        results = all_results[model]
        for sham in SHAM_TYPES:
            sham_results = [r for r in results if r.get('sham_trap_type') == sham]
            if sham_results:
                correct = sum(1 for r in sham_results if r.get('selected_tool_correct') == True)
                row.append(correct / len(sham_results) * 100)
            else:
                row.append(0)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=15, vmax=100)
    
    # Annotate cells
    for i in range(len(models)):
        for j in range(len(SHAM_TYPES)):
            val = matrix[i, j]
            color = 'white' if val < 35 or val > 85 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', 
                   color=color, fontsize=9, fontweight='bold')
    
    ax.set_xticks(np.arange(len(SHAM_TYPES)))
    ax.set_xticklabels([s.replace('_', '\n') for s in SHAM_TYPES], fontsize=9)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=10)
    
    ax.set_title('Figure 2. Detection Accuracy by Model and Sham Trap Type\n(Values show % correct identification)',
                 fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure2_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'figure2_heatmap.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 2: Heatmap")


# ============================================================================
# FIGURE 3: Position Bias Analysis
# ============================================================================

def create_figure3(all_results):
    """Position bias analysis across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Tool A preference by model
    models = list(all_results.keys())
    models.sort(key=lambda x: -calculate_accuracy(all_results[x]))
    
    a_prefs = []
    for model in models:
        picks = Counter(r.get('model_decision', {}).get('selected_tool') 
                       for r in all_results[model] if r.get('model_decision'))
        total = picks.get('A', 0) + picks.get('B', 0)
        a_pref = picks.get('A', 0) / total * 100 if total else 50
        a_prefs.append(a_pref)
    
    colors = [COLORS['danger'] if p > 70 else COLORS['warning'] if p > 60 else COLORS['success'] for p in a_prefs]
    bars = ax1.barh(models, a_prefs, color=colors, edgecolor='white', height=0.6)
    
    for bar, pref in zip(bars, a_prefs):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pref:.0f}%', va='center', fontsize=10, fontweight='bold')
    
    ax1.axvline(x=50, color='gray', linestyle='--', alpha=0.7, label='No bias')
    ax1.set_xlabel('Tool A Selection Rate (%)')
    ax1.set_xlim(0, 100)
    ax1.set_title('A. Position Bias by Model', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.invert_yaxis()
    
    # Panel B: Accuracy when model says "identical"
    identical_pattern = r'identical|same content|equivalent|both .* same|no differ'
    
    identical_data = []
    for model in models:
        identical = [r for r in all_results[model] 
                    if r.get('model_decision') and 
                    re.search(identical_pattern, r['model_decision'].get('trust_rationale', '').lower())]
        if identical:
            correct = sum(1 for r in identical if r.get('selected_tool_correct') == True)
            acc = correct / len(identical) * 100
            identical_data.append((model, acc, len(identical)))
        else:
            identical_data.append((model, 50, 0))
    
    model_names = [d[0] for d in identical_data]
    accs = [d[1] for d in identical_data]
    counts = [d[2] for d in identical_data]
    
    colors = [COLORS['danger'] if a < 45 else COLORS['warning'] if a < 55 else COLORS['success'] for a in accs]
    bars = ax2.barh(model_names, accs, color=colors, edgecolor='white', height=0.6)
    
    for bar, acc, n in zip(bars, accs, counts):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.0f}% (n={n})', va='center', fontsize=9)
    
    ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.7, label='Chance')
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_xlim(0, 80)
    ax2.set_title('B. Accuracy When Model Reports\n"Identical" Content', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.invert_yaxis()
    
    plt.suptitle('Figure 3. Position Bias Analysis', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure3_position_bias.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'figure3_position_bias.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 3: Position Bias")


# ============================================================================
# FIGURE 4: Model Strengths and Weaknesses
# ============================================================================

def create_figure4(all_results):
    """Radar/spider chart showing model profiles."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    models = list(all_results.keys())
    models.sort(key=lambda x: -calculate_accuracy(all_results[x]))
    
    # Reduce to 5 major sham categories
    categories = ['Metadata\n(outdated, citation)', 'Injection\n(prompt)', 
                  'Clinical\n(allergy, dose)', 'Population', 'Safety\n(warning, contra)']
    category_map = {
        'Metadata\n(outdated, citation)': ['outdated_version', 'fabricated_citation'],
        'Injection\n(prompt)': ['prompt_injection'],
        'Clinical\n(allergy, dose)': ['allergy_ignorance', 'dosing_error'],
        'Population': ['wrong_population'],
        'Safety\n(warning, contra)': ['missing_warning', 'contraindication_violation'],
    }
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        results = all_results[model]
        
        values = []
        for cat, shams in category_map.items():
            cat_results = [r for r in results if r.get('sham_trap_type') in shams]
            if cat_results:
                correct = sum(1 for r in cat_results if r.get('selected_tool_correct') == True)
                values.append(correct / len(cat_results) * 100)
            else:
                values.append(50)
        
        # Create bar chart for each model
        colors = [plt.cm.RdYlGn(v/100) for v in values]
        bars = ax.barh(range(len(categories)), values, color=colors, edgecolor='white')
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_width() + 1, i, f'{val:.0f}%', va='center', fontsize=9)
        
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories, fontsize=9)
        ax.set_xlim(0, 110)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(model, fontweight='bold', fontsize=11)
        ax.invert_yaxis()
    
    plt.suptitle('Figure 4. Model Vulnerability Profiles by Sham Category',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure4_model_profiles.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'figure4_model_profiles.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Figure 4: Model Profiles")


# ============================================================================
# TABLE 1: Main Results Summary
# ============================================================================

def create_table1(all_results):
    """Generate Table 1: Summary statistics."""
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 1. Summary of Model Performance on Trustworthy Tool Detection Task")
    lines.append("=" * 100)
    lines.append("")
    
    # Overall stats
    total_cases = sum(len(r) for r in all_results.values())
    total_correct = sum(sum(1 for x in r if x.get('selected_tool_correct')) for r in all_results.values())
    
    lines.append(f"Total evaluations: {total_cases:,}")
    lines.append(f"Models tested: {len(all_results)}")
    lines.append(f"Cases per model: 500")
    lines.append(f"Overall accuracy: {total_correct/total_cases*100:.1f}%")
    lines.append("")
    
    # Per-model results with CIs
    lines.append(f"{'Model':<25} {'N':>6} {'Correct':>8} {'Accuracy':>10} {'95% CI':>18} {'Position Bias':>14}")
    lines.append("-" * 100)
    
    model_data = []
    for model, results in all_results.items():
        n = len(results)
        correct = sum(1 for r in results if r.get('selected_tool_correct') == True)
        acc = correct / n
        ci_low, ci_high = wilson_ci(acc, n)
        
        picks = Counter(r.get('model_decision', {}).get('selected_tool') for r in results if r.get('model_decision'))
        total_picks = picks.get('A', 0) + picks.get('B', 0)
        a_pref = picks.get('A', 0) / total_picks * 100 if total_picks else 50
        
        model_data.append((model, n, correct, acc*100, ci_low*100, ci_high*100, a_pref))
    
    model_data.sort(key=lambda x: -x[3])
    
    for data in model_data:
        model, n, correct, acc, ci_low, ci_high, a_pref = data
        lines.append(f"{model:<25} {n:>6} {correct:>8} {acc:>9.1f}% [{ci_low:>5.1f}%, {ci_high:>5.1f}%] {a_pref:>12.0f}% A")
    
    lines.append("-" * 100)
    
    # Statistical tests
    lines.append("")
    lines.append("Statistical Analysis:")
    lines.append("-" * 50)
    
    # Chi-square test for independence
    contingency = []
    for model, results in all_results.items():
        correct = sum(1 for r in results if r.get('selected_tool_correct') == True)
        incorrect = len(results) - correct
        contingency.append([correct, incorrect])
    
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    lines.append(f"Chi-square test (model × accuracy): χ²({dof}) = {chi2:.2f}, p < {p_value:.2e}")
    
    # Best vs worst comparison (Fisher exact)
    best_results = all_results[model_data[0][0]]
    worst_results = all_results[model_data[-1][0]]
    best_correct = sum(1 for r in best_results if r.get('selected_tool_correct'))
    worst_correct = sum(1 for r in worst_results if r.get('selected_tool_correct'))
    
    table = [[best_correct, 500 - best_correct], [worst_correct, 500 - worst_correct]]
    odds_ratio, fisher_p = fisher_exact(table)
    lines.append(f"Best vs Worst (Fisher exact): OR = {odds_ratio:.2f}, p < {fisher_p:.2e}")
    
    lines.append("")
    lines.append("Note: 95% CIs calculated using Wilson score interval.")
    lines.append("Position Bias indicates percentage of responses selecting Tool A (first-presented option).")
    lines.append("=" * 100)
    
    table_text = "\n".join(lines)
    with open(OUTPUT_DIR / 'table1_summary.txt', 'w') as f:
        f.write(table_text)
    
    print("✓ Table 1: Summary")
    return table_text


# ============================================================================
# TABLE 2: Accuracy by Sham Type
# ============================================================================

def create_table2(all_results):
    """Generate Table 2: Detailed accuracy by sham type."""
    lines = []
    lines.append("=" * 120)
    lines.append("TABLE 2. Detection Accuracy by Sham Trap Type")
    lines.append("=" * 120)
    lines.append("")
    
    # Header
    models = list(all_results.keys())
    models.sort(key=lambda x: -calculate_accuracy(all_results[x]))
    
    header = f"{'Sham Type':<28}"
    for m in models:
        header += f" {m[:12]:>12}"
    header += f" {'Mean':>10} {'SD':>8}"
    lines.append(header)
    lines.append("-" * 120)
    
    # Data rows
    all_accs = []
    for sham in SHAM_TYPES:
        row = f"{sham:<28}"
        sham_accs = []
        for model in models:
            results = all_results[model]
            sham_results = [r for r in results if r.get('sham_trap_type') == sham]
            if sham_results:
                correct = sum(1 for r in sham_results if r.get('selected_tool_correct'))
                acc = correct / len(sham_results) * 100
                sham_accs.append(acc)
                row += f" {acc:>11.0f}%"
            else:
                row += f" {'N/A':>12}"
        
        if sham_accs:
            mean_acc = np.mean(sham_accs)
            std_acc = np.std(sham_accs)
            row += f" {mean_acc:>9.1f}% {std_acc:>7.1f}"
            all_accs.append((sham, mean_acc, std_acc))
        
        lines.append(row)
    
    lines.append("-" * 120)
    
    # Sort by difficulty
    lines.append("")
    lines.append("Sham Types Ranked by Difficulty (Mean Accuracy):")
    all_accs.sort(key=lambda x: -x[1])
    for i, (sham, mean, std) in enumerate(all_accs, 1):
        difficulty = "Easy" if mean > 80 else "Medium" if mean > 50 else "Hard"
        lines.append(f"  {i}. {sham}: {mean:.1f}% ± {std:.1f} ({difficulty})")
    
    lines.append("=" * 120)
    
    table_text = "\n".join(lines)
    with open(OUTPUT_DIR / 'table2_sham_types.txt', 'w') as f:
        f.write(table_text)
    
    print("✓ Table 2: Sham Types")
    return table_text


# ============================================================================
# SUPPLEMENTARY FIGURE: Confidence Analysis
# ============================================================================

def create_supp_figure_confidence(all_results):
    """Supplementary: Confidence calibration."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    models = list(all_results.keys())
    models.sort(key=lambda x: -calculate_accuracy(all_results[x]))
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        results = all_results[model]
        
        correct_conf = [r['model_decision']['confidence'] for r in results 
                       if r.get('selected_tool_correct') and r.get('model_decision') 
                       and r['model_decision'].get('confidence') is not None]
        incorrect_conf = [r['model_decision']['confidence'] for r in results 
                         if not r.get('selected_tool_correct') and r.get('model_decision')
                         and r['model_decision'].get('confidence') is not None]
        
        if correct_conf and incorrect_conf:
            ax.hist([correct_conf, incorrect_conf], bins=10, label=['Correct', 'Incorrect'],
                   color=[COLORS['success'], COLORS['danger']], alpha=0.7, edgecolor='white')
            
            # T-test
            t_stat, p_val = stats.ttest_ind(correct_conf, incorrect_conf)
            ax.text(0.05, 0.95, f'p = {p_val:.3f}', transform=ax.transAxes, fontsize=9)
        
        ax.set_title(model, fontweight='bold')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    plt.suptitle('Supplementary Figure S1. Model Confidence Calibration',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'supp_figure_s1_confidence.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Supplementary Figure S1: Confidence")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Generating Publication Figures for Nature Medicine")
    print("=" * 60)
    print()
    
    print("Loading results from 6 experiments...")
    all_results = load_all_results()
    total = sum(len(r) for r in all_results.values())
    print(f"Loaded {total:,} total evaluations\n")
    
    print("Generating main figures...")
    create_figure1(all_results)
    create_figure2(all_results)
    create_figure3(all_results)
    create_figure4(all_results)
    
    print("\nGenerating tables...")
    table1 = create_table1(all_results)
    table2 = create_table2(all_results)
    
    print("\nGenerating supplementary figures...")
    create_supp_figure_confidence(all_results)
    
    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Print summary
    print("\n" + table1)


if __name__ == "__main__":
    main()
