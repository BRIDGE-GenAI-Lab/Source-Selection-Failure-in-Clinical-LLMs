#!/usr/bin/env python3
"""
Generate NEJM-style publication figures and tables for the manuscript.
Uses matplotlib with custom styling for clean, professional output.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# NEJM-style color palette
COLORS = {
    'primary': '#4A6B87',      # Grayish blue
    'secondary': '#6B8BA4',    # Lighter blue
    'accent': '#3D5A73',       # Darker blue
    'light': '#8BABC4',        # Light blue
    'warning': '#9B7A7A',      # Muted red for concerning
    'background': '#F7F9FB',   # Light gray background
    'text': '#2C3E50',         # Dark text
    'grid': '#E0E6EB',         # Light grid
}

# Set default style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': COLORS['grid'],
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.color': COLORS['text'],
    'axes.labelcolor': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
})

# Output directory
OUTPUT_DIR = Path('manuscript_figures')
OUTPUT_DIR.mkdir(exist_ok=True)


def create_figure1_accuracy_bars():
    """Create horizontal bar chart of accuracy by trap type."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data (sorted by accuracy)
    trap_types = [
        'Prompt Injection',
        'Contraindication Violation', 
        'Buzzword Overconfidence',
        'Fabricated Citation',
        'Outdated Recommendation'
    ]
    accuracies = [52, 68, 80, 90, 100]
    n_cases = [25, 25, 20, 10, 20]
    
    # Colors - gradient based on accuracy, prompt injection gets warning color
    bar_colors = [COLORS['warning'] if a == 52 else COLORS['primary'] for a in accuracies]
    
    # Create horizontal bars
    y_pos = np.arange(len(trap_types))
    bars = ax.barh(y_pos, accuracies, height=0.6, color=bar_colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, acc, n) in enumerate(zip(bars, accuracies, n_cases)):
        width = bar.get_width()
        ax.text(width - 3, bar.get_y() + bar.get_height()/2, 
                f'{acc}%', ha='right', va='center', color='white', fontweight='bold', fontsize=12)
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'n={n}', ha='left', va='center', color=COLORS['text'], fontsize=10)
    
    # Reference line at 75% (overall accuracy)
    ax.axvline(x=75, color=COLORS['accent'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(75, len(trap_types) - 0.3, 'Overall: 75%', ha='center', va='bottom', 
            color=COLORS['accent'], fontsize=10, fontstyle='italic')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(trap_types)
    ax.set_xlim(0, 110)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Figure 1. Model Accuracy by Adversarial Trap Type', fontsize=14, fontweight='bold', pad=20)
    
    # Subtitle
    fig.text(0.5, 0.91, 'Percentage of correct trustworthy tool selections (N=100 cases)', 
             ha='center', fontsize=10, color=COLORS['secondary'])
    
    # Light gridlines
    ax.xaxis.grid(True, linestyle='-', alpha=0.3, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure1_accuracy_by_trap.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'figure1_accuracy_by_trap.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Created Figure 1: {OUTPUT_DIR / 'figure1_accuracy_by_trap.png'}")


def create_figure2_detection_gap():
    """Create visualization of detection-action gap for prompt injection."""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Data for prompt injection cases (n=25)
    # Detected injection: 12, Not detected: 13
    # Of detected (12): 4 correct, 8 incorrect
    # Of not detected (13): 9 correct, 4 incorrect
    
    # Create a stacked bar comparison
    categories = ['Injection\nDetected\n(n=12)', 'Injection\nNot Detected\n(n=13)', 'Overall\n(n=25)']
    correct = [4, 9, 13]
    incorrect = [8, 4, 12]
    
    x = np.arange(len(categories))
    width = 0.5
    
    # Stacked bars
    bars_correct = ax.bar(x, correct, width, label='Correct Selection', color=COLORS['primary'], edgecolor='white')
    bars_incorrect = ax.bar(x, incorrect, width, bottom=correct, label='Incorrect Selection', color=COLORS['warning'], edgecolor='white')
    
    # Add percentage labels
    for i, (c, inc) in enumerate(zip(correct, incorrect)):
        total = c + inc
        # Correct portion
        ax.text(i, c/2, f'{c}\n({c/total*100:.0f}%)', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=11)
        # Incorrect portion  
        ax.text(i, c + inc/2, f'{inc}\n({inc/total*100:.0f}%)', ha='center', va='center',
                color='white', fontweight='bold', fontsize=11)
    
    # Key finding annotation
    ax.annotate('', xy=(0, 12), xytext=(0.7, 14),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))
    ax.text(1.0, 14.5, '67% incorrect\ndespite detection', ha='left', va='bottom',
            fontsize=10, color=COLORS['accent'], fontstyle='italic')
    
    # Styling
    ax.set_ylabel('Number of Cases', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 18)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor=COLORS['grid'])
    
    ax.set_title('Figure 2. Detection-Action Gap in Prompt Injection Cases', 
                 fontsize=14, fontweight='bold', pad=20)
    fig.text(0.5, 0.91, 'Model behavior when encountering embedded override instructions', 
             ha='center', fontsize=10, color=COLORS['secondary'])
    
    # Add key finding box
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor=COLORS['background'], edgecolor=COLORS['grid'])
    ax.text(0.5, -0.18, 'Key Finding: Detection of malicious content did not improve selection accuracy (33% vs 69%)',
            transform=ax.transAxes, ha='center', va='top', fontsize=10, 
            bbox=bbox_props, color=COLORS['text'])
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(OUTPUT_DIR / 'figure2_detection_gap.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'figure2_detection_gap.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Created Figure 2: {OUTPUT_DIR / 'figure2_detection_gap.png'}")


def create_table_images():
    """Create table images for the manuscript."""
    
    # Table 1: Summary Metrics
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Value', 'Notes'],
        ['Total Cases Evaluated', '100', '—'],
        ['Overall Accuracy', '75.0%', '75/100 correct'],
        ['Tool Call Compliance', '100.0%', 'Both tools called'],
        ['Unclear Response Rate', '1.0%', '1/100 cases'],
        ['Mean Confidence Score', '0.725', 'SD 0.068'],
        ['Processing Failures', '0', '—'],
    ]
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colColours=[COLORS['primary']]*3)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
 
    ### 
    # Style header
    for i in range(3):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
        table[(0, i)].set_facecolor(COLORS['primary'])
    
    # Style data rows with alternating colors
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor(COLORS['background'])
            else:
                table[(i, j)].set_facecolor('white')
    
    ax.set_title('Table 1. Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'table1_summary.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Created Table 1: {OUTPUT_DIR / 'table1_summary.png'}")
    
    # Table 2: Accuracy by Trap Type
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    table_data = [
        ['Trap Type', 'n', 'Correct', 'Accuracy', 'Interpretation'],
        ['Outdated Recommendation', '20', '20', '100.0%', 'Version/date discrepancies detected'],
        ['Fabricated Citation', '10', '9', '90.0%', 'Invalid references recognized'],
        ['Buzzword Overconfidence', '20', '16', '80.0%', 'Pseudo-scientific claims identified'],
        ['Contraindication Violation', '25', '17', '68.0%', 'Partial detection of unsafe advice'],
        ['Prompt Injection', '25', '13', '52.0%', 'High susceptibility to manipulation'],
        ['TOTAL', '100', '75', '75.0%', '—'],
    ]
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colColours=[COLORS['primary']]*5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.7)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Style prompt injection row (row 5) with warning color
    for j in range(5):
        table[(5, j)].set_facecolor('#FFF0F0')
    
    # Style total row
    for j in range(5):
        table[(6, j)].set_facecolor(COLORS['background'])
        table[(6, j)].set_text_props(fontweight='bold')
    
    ax.set_title('Table 2. Accuracy by Adversarial Trap Type', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'table2_accuracy_by_trap.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Created Table 2: {OUTPUT_DIR / 'table2_accuracy_by_trap.png'}")
    
    # Table 3: Failure Analysis
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis('off')
    
    table_data = [
        ['Failure Pattern', 'Frequency', '% of Failures', 'Example'],
        ['Version Recency Bias', '14', '56%', 'Selected v5.0 sham over v3.2 trustworthy'],
        ['Detected but Ignored', '8', '32%', 'Noted "SYSTEM OVERRIDE" but selected sham'],
        ['Authority Bias', '2', '8%', 'Preferred "Rapid Evidence" variant'],
        ['Incomplete Provenance', '1', '4%', 'Did not verify citation DOIs'],
    ]
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colColours=[COLORS['primary']]*4)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor(COLORS['background'])
    
    ax.set_title('Table 3. Failure Mode Analysis (n=25 failures)', fontsize=14, fontweight='bold', pad=20)
    
    # Add key finding
    fig.text(0.5, 0.08, 'Key Finding: 56% of failures were due to version recency bias—selecting newer-versioned sham tools.',
             ha='center', fontsize=10, fontstyle='italic', color=COLORS['accent'])
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(OUTPUT_DIR / 'table3_failure_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Created Table 3: {OUTPUT_DIR / 'table3_failure_analysis.png'}")


if __name__ == '__main__':
    print("Generating NEJM-style manuscript figures...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    create_figure1_accuracy_bars()
    create_figure2_detection_gap()
    create_table_images()
    
    print(f"\n✓ All figures saved to: {OUTPUT_DIR}")
