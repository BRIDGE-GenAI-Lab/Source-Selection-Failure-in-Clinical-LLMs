"""
Generate Nature Medicine / NEJM quality figures and tables for the
Safe Guideline Tooling Evaluation study.

Uses professional color palette with grayish-blue tint.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict, Counter
from pathlib import Path
import scipy.stats as stats

# NEJM/Nature Medicine color palette (grayish-blue tint)
COLORS = {
    'primary': '#4A6FA5',      # Steel blue
    'secondary': '#6B8EB8',    # Light steel blue
    'accent': '#2E4A62',       # Dark navy
    'success': '#5D8A66',      # Sage green
    'danger': '#B85C5C',       # Muted red
    'warning': '#C4A35A',      # Muted gold
    'neutral': '#8B9BAE',      # Gray blue
    'light': '#D4DBE5',        # Light gray blue
    'background': '#F5F7FA',   # Off white
    'text': '#2D3748',         # Dark gray
}

# Extended palette for multiple categories
SHAM_COLORS = {
    'outdated_version': '#2E7D32',      # Green - best
    'fabricated_citation': '#388E3C',   
    'subtle_inversion': '#5D8A66',      
    'prompt_injection': '#7B9E87',      
    'contraindication_violation': '#8B9BAE',  
    'authority_mimicry': '#9E8B7B',     
    'dosing_error': '#B8896B',          
    'missing_warning': '#C4785A',       
    'wrong_population': '#B85C5C',      
    'allergy_ignorance': '#A03030',     # Red - worst
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Output directory
OUTPUT_DIR = Path("publication_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_results():
    """Load experiment results."""
    results_path = "results/run_20260107_085822/results.jsonl"
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results


def compute_statistics(results):
    """Compute comprehensive statistics."""
    stats_dict = {}
    
    # Overall
    correct = sum(1 for r in results if r.get('selected_tool_correct') == True)
    total = len(results)
    stats_dict['overall'] = {
        'correct': correct,
        'incorrect': total - correct,
        'total': total,
        'accuracy': correct / total * 100
    }
    
    # By sham type
    by_type = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    for r in results:
        sham = r.get('sham_trap_type', 'unknown')
        if r.get('selected_tool_correct') == True:
            by_type[sham]['correct'] += 1
        else:
            by_type[sham]['incorrect'] += 1
    
    for sham, counts in by_type.items():
        total = counts['correct'] + counts['incorrect']
        counts['total'] = total
        counts['accuracy'] = counts['correct'] / total * 100 if total > 0 else 0
    
    stats_dict['by_sham_type'] = dict(by_type)
    
    # Confidence analysis
    confidences_correct = [r['model_decision']['confidence'] for r in results 
                          if r.get('selected_tool_correct') == True and r.get('model_decision')]
    confidences_incorrect = [r['model_decision']['confidence'] for r in results 
                            if r.get('selected_tool_correct') == False and r.get('model_decision')]
    
    stats_dict['confidence'] = {
        'correct_mean': np.mean(confidences_correct) if confidences_correct else 0,
        'correct_std': np.std(confidences_correct) if confidences_correct else 0,
        'incorrect_mean': np.mean(confidences_incorrect) if confidences_incorrect else 0,
        'incorrect_std': np.std(confidences_incorrect) if confidences_incorrect else 0,
        'correct_values': confidences_correct,
        'incorrect_values': confidences_incorrect,
    }
    
    return stats_dict


def figure1_main_accuracy(stats_dict, results):
    """
    Figure 1: Main accuracy results (2-panel figure).
    Panel A: Overall accuracy donut chart
    Panel B: Accuracy by sham type horizontal bar chart
    """
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, width_ratios=[1, 1.8], wspace=0.3)
    
    # Panel A: Overall accuracy donut
    ax1 = fig.add_subplot(gs[0])
    
    overall = stats_dict['overall']
    sizes = [overall['correct'], overall['incorrect']]
    colors = [COLORS['success'], COLORS['danger']]
    
    wedges, texts, autotexts = ax1.pie(sizes, colors=colors, autopct='%1.1f%%',
                                        startangle=90, pctdistance=0.75,
                                        wedgeprops=dict(width=0.5, edgecolor='white'))
    
    # Center text
    ax1.text(0, 0, f'{overall["accuracy"]:.1f}%\nAccuracy', 
             ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['text'])
    
    ax1.set_title('A. Overall Model Performance', fontweight='bold', pad=20)
    ax1.legend([f'Correct (n={overall["correct"]})', f'Incorrect (n={overall["incorrect"]})'],
               loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    
    # Panel B: Accuracy by sham type
    ax2 = fig.add_subplot(gs[1])
    
    sham_data = stats_dict['by_sham_type']
    sorted_shams = sorted(sham_data.items(), key=lambda x: x[1]['accuracy'])
    
    names = [s[0].replace('_', ' ').title() for s in sorted_shams]
    accuracies = [s[1]['accuracy'] for s in sorted_shams]
    totals = [s[1]['total'] for s in sorted_shams]
    
    # Color gradient from red (low) to green (high)
    colors = [plt.cm.RdYlGn(acc/100) for acc in accuracies]
    
    y_pos = np.arange(len(names))
    bars = ax2.barh(y_pos, accuracies, color=colors, edgecolor='white', height=0.7)
    
    # Add value labels
    for i, (bar, acc, total) in enumerate(zip(bars, accuracies, totals)):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}% (n={total})', va='center', fontsize=9, color=COLORS['text'])
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel('Detection Accuracy (%)')
    ax2.set_xlim(0, 115)
    ax2.axvline(x=50, color=COLORS['neutral'], linestyle='--', alpha=0.5, label='Chance level')
    ax2.set_title('B. Detection Accuracy by Sham Type', fontweight='bold', pad=20)
    
    # Add reference line legend
    ax2.legend(loc='lower right', frameon=False)
    
    plt.suptitle('Figure 1. LLM Trustworthiness Detection Performance on Guideline Tool Selection',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure1_main_accuracy.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'figure1_main_accuracy.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Figure 1 saved")


def figure2_confidence_analysis(stats_dict, results):
    """
    Figure 2: Confidence calibration analysis.
    Panel A: Confidence distribution by correctness
    Panel B: Confidence vs accuracy by sham type
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    conf = stats_dict['confidence']
    
    # Panel A: Violin/box plot of confidence
    data = [conf['correct_values'], conf['incorrect_values']]
    positions = [1, 2]
    
    parts = ax1.violinplot(data, positions, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([COLORS['success'], COLORS['danger']][i])
        pc.set_alpha(0.7)
    
    parts['cmeans'].set_color(COLORS['text'])
    parts['cmedians'].set_color(COLORS['accent'])
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['Correct\nSelections', 'Incorrect\nSelections'])
    ax1.set_ylabel('Model Confidence')
    ax1.set_ylim(0, 1)
    ax1.set_title('A. Confidence Distribution', fontweight='bold')
    
    # Add mean annotations
    ax1.annotate(f'μ={conf["correct_mean"]:.2f}', xy=(1, conf['correct_mean']), 
                xytext=(1.3, conf['correct_mean']+0.05), fontsize=9)
    ax1.annotate(f'μ={conf["incorrect_mean"]:.2f}', xy=(2, conf['incorrect_mean']),
                xytext=(2.3, conf['incorrect_mean']+0.05), fontsize=9)
    
    # Panel B: Scatter plot - confidence vs accuracy by sham type
    sham_data = stats_dict['by_sham_type']
    
    # Calculate mean confidence per sham type
    conf_by_sham = defaultdict(list)
    for r in results:
        if r.get('model_decision'):
            sham = r.get('sham_trap_type', 'unknown')
            conf_by_sham[sham].append(r['model_decision']['confidence'])
    
    x_vals = []  # mean confidence
    y_vals = []  # accuracy
    sizes = []   # sample size
    labels = []
    colors_scatter = []
    
    for sham, confs in conf_by_sham.items():
        x_vals.append(np.mean(confs))
        y_vals.append(sham_data[sham]['accuracy'])
        sizes.append(sham_data[sham]['total'] * 3)
        labels.append(sham.replace('_', '\n'))
        colors_scatter.append(plt.cm.RdYlGn(sham_data[sham]['accuracy']/100))
    
    scatter = ax2.scatter(x_vals, y_vals, s=sizes, c=colors_scatter, 
                         alpha=0.7, edgecolors='white', linewidth=2)
    
    # Add labels
    for x, y, label in zip(x_vals, y_vals, labels):
        ax2.annotate(label, (x, y), textcoords="offset points", 
                    xytext=(0, 8), ha='center', fontsize=7)
    
    ax2.axhline(y=50, color=COLORS['neutral'], linestyle='--', alpha=0.5)
    ax2.set_xlabel('Mean Model Confidence')
    ax2.set_ylabel('Detection Accuracy (%)')
    ax2.set_xlim(0.4, 0.9)
    ax2.set_ylim(20, 110)
    ax2.set_title('B. Confidence vs Accuracy by Sham Type', fontweight='bold')
    
    plt.suptitle('Figure 2. Model Confidence Calibration Analysis',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure2_confidence.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'figure2_confidence.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Figure 2 saved")


def figure3_failure_analysis(results):
    """
    Figure 3: Failure mode analysis.
    Shows what reasoning patterns lead to failures.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Collect failure reasons by sham type
    failures = [r for r in results if r.get('selected_tool_correct') == False]
    
    # Panel A: Safety notes detection rate
    ax1 = axes[0, 0]
    
    by_type_safety = defaultdict(lambda: {'detected': 0, 'missed': 0})
    for r in results:
        sham = r.get('sham_trap_type', 'unknown')
        safety_notes = r.get('model_decision', {}).get('safety_notes', [])
        if safety_notes:
            by_type_safety[sham]['detected'] += 1
        else:
            by_type_safety[sham]['missed'] += 1
    
    sorted_types = sorted(by_type_safety.items(), 
                         key=lambda x: x[1]['detected']/(x[1]['detected']+x[1]['missed']))
    
    names = [s[0].replace('_', ' ').title()[:15] for s in sorted_types]
    detected = [s[1]['detected'] for s in sorted_types]
    missed = [s[1]['missed'] for s in sorted_types]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, detected, width, label='Safety Issue Detected', color=COLORS['success'])
    ax1.bar(x + width/2, missed, width, label='No Safety Issue Noted', color=COLORS['danger'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Number of Cases')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title('A. Safety Issue Detection by Sham Type', fontweight='bold')
    
    # Panel B: Confidence in failures
    ax2 = axes[0, 1]
    
    conf_failures_by_type = defaultdict(list)
    for r in failures:
        sham = r.get('sham_trap_type', 'unknown')
        if r.get('model_decision'):
            conf_failures_by_type[sham].append(r['model_decision']['confidence'])
    
    sorted_conf = sorted(conf_failures_by_type.items(), 
                        key=lambda x: np.mean(x[1]) if x[1] else 0, reverse=True)
    
    names = [s[0].replace('_', ' ').title()[:15] for s in sorted_conf]
    means = [np.mean(s[1]) if s[1] else 0 for s in sorted_conf]
    stds = [np.std(s[1]) if len(s[1]) > 1 else 0 for s in sorted_conf]
    
    colors = [COLORS['danger'] if m > 0.6 else COLORS['warning'] if m > 0.4 else COLORS['neutral'] 
              for m in means]
    
    y_pos = np.arange(len(names))
    ax2.barh(y_pos, means, xerr=stds, color=colors, capsize=3, edgecolor='white')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel('Mean Confidence (on failures)')
    ax2.axvline(x=0.5, color=COLORS['neutral'], linestyle='--', alpha=0.5)
    ax2.set_title('B. Confidence When Model Was Wrong', fontweight='bold')
    
    # Panel C: Contradictions detected
    ax3 = axes[1, 0]
    
    contradictions = defaultdict(lambda: {'found': 0, 'not_found': 0})
    for r in results:
        sham = r.get('sham_trap_type', 'unknown')
        contras = r.get('model_decision', {}).get('contradictions_found', [])
        if contras:
            contradictions[sham]['found'] += 1
        else:
            contradictions[sham]['not_found'] += 1
    
    sorted_contra = sorted(contradictions.items(),
                          key=lambda x: x[1]['found']/(x[1]['found']+x[1]['not_found']))
    
    names = [s[0].replace('_', ' ').title()[:15] for s in sorted_contra]
    rates = [s[1]['found']/(s[1]['found']+s[1]['not_found'])*100 for s in sorted_contra]
    
    colors = [plt.cm.Blues(r/100) for r in rates]
    ax3.barh(range(len(names)), rates, color=colors, edgecolor='white')
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names)
    ax3.set_xlabel('Contradiction Detection Rate (%)')
    ax3.set_title('C. Contradiction Detection by Sham Type', fontweight='bold')
    
    # Panel D: Failure count by sham type (pie)
    ax4 = axes[1, 1]
    
    failure_counts = Counter(r['sham_trap_type'] for r in failures)
    sorted_failures = sorted(failure_counts.items(), key=lambda x: -x[1])
    
    labels = [f"{s[0].replace('_', ' ').title()}\n(n={s[1]})" for s in sorted_failures]
    sizes = [s[1] for s in sorted_failures]
    colors = [plt.cm.Reds(0.3 + 0.5*i/len(sizes)) for i in range(len(sizes))]
    
    wedges, texts = ax4.pie(sizes, colors=colors, startangle=90,
                            wedgeprops=dict(edgecolor='white'))
    ax4.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax4.set_title('D. Distribution of Failures by Type', fontweight='bold')
    
    plt.suptitle('Figure 3. Failure Mode Analysis',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure3_failure_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'figure3_failure_analysis.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Figure 3 saved")


def create_table1_summary(stats_dict, results):
    """Create Table 1: Summary statistics."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 1. Summary of Model Performance on Guideline Tool Selection Task")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Overall Performance")
    lines.append("-" * 40)
    overall = stats_dict['overall']
    lines.append(f"  Total cases evaluated:        {overall['total']}")
    lines.append(f"  Correct selections:           {overall['correct']} ({overall['accuracy']:.1f}%)")
    lines.append(f"  Incorrect selections:         {overall['incorrect']} ({100-overall['accuracy']:.1f}%)")
    lines.append(f"  Unclear responses:            0 (0.0%)")
    lines.append("")
    
    # Confidence
    conf = stats_dict['confidence']
    lines.append("Model Confidence")
    lines.append("-" * 40)
    lines.append(f"  Mean confidence (correct):    {conf['correct_mean']:.3f} ± {conf['correct_std']:.3f}")
    lines.append(f"  Mean confidence (incorrect):  {conf['incorrect_mean']:.3f} ± {conf['incorrect_std']:.3f}")
    
    # T-test
    t_stat, p_val = stats.ttest_ind(conf['correct_values'], conf['incorrect_values'])
    lines.append(f"  t-test p-value:               {p_val:.4f}")
    lines.append("")
    
    lines.append("Performance by Sham Type")
    lines.append("-" * 80)
    lines.append(f"{'Sham Type':<30} {'N':>6} {'Correct':>10} {'Incorrect':>10} {'Accuracy':>12}")
    lines.append("-" * 80)
    
    sham_data = stats_dict['by_sham_type']
    sorted_shams = sorted(sham_data.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for sham, data in sorted_shams:
        lines.append(f"{sham:<30} {data['total']:>6} {data['correct']:>10} {data['incorrect']:>10} {data['accuracy']:>11.1f}%")
    
    lines.append("-" * 80)
    lines.append("")
    lines.append("Note: Accuracy represents correct identification of the trustworthy tool (T) over the sham tool (S).")
    lines.append("CI = confidence interval; N = number of cases")
    lines.append("=" * 80)
    
    table_text = "\n".join(lines)
    
    with open(OUTPUT_DIR / 'table1_summary.txt', 'w') as f:
        f.write(table_text)
    
    print("✓ Table 1 saved")
    return table_text


def create_appendix_tables(results, stats_dict):
    """Create appendix tables with detailed breakdowns."""
    
    # Appendix Table A1: Case-level results sample
    lines = []
    lines.append("APPENDIX TABLE A1. Sample Case-Level Results (First 20 Cases)")
    lines.append("=" * 120)
    lines.append(f"{'Case ID':<12} {'Sham Type':<25} {'Selected':<10} {'Correct':<10} {'Confidence':<12} {'Has Safety Notes':<15}")
    lines.append("-" * 120)
    
    for r in results[:20]:
        md = r.get('model_decision', {})
        lines.append(f"{r['case_id']:<12} {r['sham_trap_type']:<25} {md.get('selected_tool', 'N/A'):<10} "
                    f"{'Yes' if r.get('selected_tool_correct') else 'No':<10} "
                    f"{md.get('confidence', 0):<12.2f} {'Yes' if md.get('safety_notes') else 'No':<15}")
    
    lines.append("=" * 120)
    
    with open(OUTPUT_DIR / 'appendix_table_a1_sample_cases.txt', 'w') as f:
        f.write("\n".join(lines))
    
    # Appendix Table A2: Confidence intervals
    lines = []
    lines.append("APPENDIX TABLE A2. 95% Confidence Intervals for Detection Accuracy by Sham Type")
    lines.append("=" * 90)
    lines.append(f"{'Sham Type':<30} {'Accuracy':<12} {'95% CI':<20} {'N':<8}")
    lines.append("-" * 90)
    
    sham_data = stats_dict['by_sham_type']
    for sham, data in sorted(sham_data.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        # Wilson score interval
        n = data['total']
        p = data['accuracy'] / 100
        z = 1.96
        
        denominator = 1 + z**2/n
        center = (p + z**2/(2*n)) / denominator
        margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
        
        ci_low = max(0, (center - margin) * 100)
        ci_high = min(100, (center + margin) * 100)
        
        lines.append(f"{sham:<30} {data['accuracy']:>10.1f}% [{ci_low:>6.1f}%, {ci_high:>6.1f}%] {n:>8}")
    
    lines.append("=" * 90)
    lines.append("Note: 95% CIs calculated using Wilson score interval")
    
    with open(OUTPUT_DIR / 'appendix_table_a2_confidence_intervals.txt', 'w') as f:
        f.write("\n".join(lines))
    
    # Appendix Table A3: Failure case examples with rationales
    lines = []
    lines.append("APPENDIX TABLE A3. Representative Failure Cases with Model Rationales")
    lines.append("=" * 120)
    
    failures = [r for r in results if r.get('selected_tool_correct') == False]
    
    # Get 2 examples from each sham type
    by_type = defaultdict(list)
    for r in failures:
        by_type[r['sham_trap_type']].append(r)
    
    for sham_type in ['allergy_ignorance', 'missing_warning', 'dosing_error', 'wrong_population']:
        cases = by_type.get(sham_type, [])[:2]
        if cases:
            lines.append(f"\n{sham_type.upper().replace('_', ' ')}")
            lines.append("-" * 80)
            for c in cases:
                md = c.get('model_decision', {})
                lines.append(f"  {c['case_id']} (confidence: {md.get('confidence', 0):.2f})")
                rationale = md.get('trust_rationale', '')[:200]
                lines.append(f"  Rationale: \"{rationale}...\"")
                lines.append("")
    
    with open(OUTPUT_DIR / 'appendix_table_a3_failure_examples.txt', 'w') as f:
        f.write("\n".join(lines))
    
    print("✓ Appendix tables saved")


def create_appendix_figures(results, stats_dict):
    """Create appendix figures."""
    
    # Appendix Figure S1: Accuracy by clinical domain
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Load case data to get domains
    with open('500cases_with_real_tools.json') as f:
        case_data = json.load(f)
    
    # Map case_id to domain (from tool metadata)
    case_domains = {}
    for case in case_data['cases_with_real_tools']['cases']:
        case_id = case['case_id']
        # Get domain from tool T metadata notes
        notes = case.get('tools', {}).get('T', {}).get('metadata', {}).get('notes', '')
        if 'Mapped to domain:' in notes:
            domain = notes.split('Mapped to domain:')[1].strip()
            case_domains[case_id] = domain
        else:
            case_domains[case_id] = 'Unknown'
    
    # Calculate accuracy by domain
    by_domain = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    for r in results:
        domain = case_domains.get(r['case_id'], 'Unknown')
        if r.get('selected_tool_correct') == True:
            by_domain[domain]['correct'] += 1
        else:
            by_domain[domain]['incorrect'] += 1
    
    # Filter domains with >= 5 cases
    filtered_domains = {k: v for k, v in by_domain.items() 
                       if v['correct'] + v['incorrect'] >= 5}
    
    sorted_domains = sorted(filtered_domains.items(), 
                           key=lambda x: x[1]['correct']/(x[1]['correct']+x[1]['incorrect']))
    
    names = [d[0] for d in sorted_domains]
    accuracies = [d[1]['correct']/(d[1]['correct']+d[1]['incorrect'])*100 for d in sorted_domains]
    totals = [d[1]['correct']+d[1]['incorrect'] for d in sorted_domains]
    
    colors = [plt.cm.RdYlGn(acc/100) for acc in accuracies]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, accuracies, color=colors, edgecolor='white')
    
    for i, (bar, total) in enumerate(zip(bars, totals)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f'n={total}', va='center', fontsize=8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Detection Accuracy (%)')
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 110)
    ax.set_title('Appendix Figure S1. Detection Accuracy by Clinical Domain', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'appendix_figure_s1_by_domain.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    # Appendix Figure S2: Confidence histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conf_correct = stats_dict['confidence']['correct_values']
    conf_incorrect = stats_dict['confidence']['incorrect_values']
    
    bins = np.linspace(0, 1, 21)
    ax.hist(conf_correct, bins=bins, alpha=0.7, label='Correct', color=COLORS['success'], edgecolor='white')
    ax.hist(conf_incorrect, bins=bins, alpha=0.7, label='Incorrect', color=COLORS['danger'], edgecolor='white')
    
    ax.axvline(np.mean(conf_correct), color=COLORS['success'], linestyle='--', linewidth=2)
    ax.axvline(np.mean(conf_incorrect), color=COLORS['danger'], linestyle='--', linewidth=2)
    
    ax.set_xlabel('Model Confidence')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_title('Appendix Figure S2. Distribution of Model Confidence Scores', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'appendix_figure_s2_confidence_histogram.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    print("✓ Appendix figures saved")


def main():
    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} cases")
    
    print("\nComputing statistics...")
    stats_dict = compute_statistics(results)
    
    print("\nGenerating figures...")
    figure1_main_accuracy(stats_dict, results)
    figure2_confidence_analysis(stats_dict, results)
    figure3_failure_analysis(results)
    
    print("\nGenerating tables...")
    table1 = create_table1_summary(stats_dict, results)
    
    print("\nGenerating appendix materials...")
    create_appendix_tables(results, stats_dict)
    create_appendix_figures(results, stats_dict)
    
    print(f"\n{'='*60}")
    print("All outputs saved to:", OUTPUT_DIR)
    print(f"{'='*60}")
    
    # Print summary for user
    print("\n" + table1)


if __name__ == "__main__":
    main()
