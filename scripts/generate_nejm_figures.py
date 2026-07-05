#!/usr/bin/env python3
"""
Generate NEJM-quality supplementary figures for the appendix.
Uses canonical 21 models and 10 sham types.
Professional styling with 600 DPI, proper fonts, significant markers.
Updated Figure S1: Complex grouping by Type (Closed, Open Large, Open Small) and MoE status.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# Set NEJM style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# NEJM color palette
COLORS = {
    'primary': '#2B5F8A',      # NEJM Blue
    'secondary': '#8B2B5F',    # NEJM Burgundy  
    'tertiary': '#5F2B8B',     # Purple for reasoning
    'accent_green': '#2B8B5F', # Success green
    'accent_red': '#C23B22',   # Alert red
    'gray': '#666666',
    'light_gray': '#CCCCCC',
    # New complex S1 colors
    'closed': '#2B5F8A',       # Blue
    'open_large': '#8B2B5F',   # Burgundy
    'open_small': '#E69F00',   # Orange (Colorblind safe)
}

BASE_DIR = Path(Path(__file__).parent.parent)
OUTPUT_DIR = BASE_DIR / 'supplementary_figures'

SHAM_TYPES = [
    'missing_warning', 'allergy_ignorance', 'dosing_error', 
    'contraindication_violation', 'wrong_population', 'subtle_inversion',
    'authority_mimicry', 'prompt_injection', 'fabricated_citation', 'outdated_version'
]

SHAM_DISPLAY = {
    'missing_warning': 'Missing Warning',
    'allergy_ignorance': 'Allergy Ignorance', 
    'dosing_error': 'Dosing Error',
    'contraindication_violation': 'Contraindication',
    'wrong_population': 'Wrong Population',
    'subtle_inversion': 'Subtle Inversion',
    'authority_mimicry': 'Authority Mimicry',
    'prompt_injection': 'Prompt Injection',
    'fabricated_citation': 'Fabricated Citation',
    'outdated_version': 'Outdated Version',
}

# Metadata Map for Figure S1
MODEL_META = {
    # Closed
    'GPT-4.1': {'type': 'Closed', 'moe': False, 'size': 'Closed'},
    'GPT-4.1-Nano': {'type': 'Closed', 'moe': False, 'size': 'Closed'},
    'GPT-4o-Mini': {'type': 'Closed', 'moe': False, 'size': 'Closed'},
    'GPT-5-Nano': {'type': 'Closed', 'moe': False, 'size': 'Closed'},
    'Gemini-2.5-Flash': {'type': 'Closed', 'moe': False, 'size': 'Closed'},
    'DeepSeek-V3.2': {'type': 'Open Weights', 'moe': True, 'size': 'Large'}, 
    'DeepSeek Reasoner': {'type': 'Closed', 'moe': False, 'size': 'Closed'},
    
    # Open Large (>50B)
    'Qwen/Qwen3-235B-A22B-Instruct': {'type': 'Open Weights', 'moe': False, 'size': 'Large'},
    'openai/gpt-oss-120b': {'type': 'Open Weights', 'moe': False, 'size': 'Large'},
    'Qwen/Qwen3-Next-80B-A3B-Thinking': {'type': 'Open Weights', 'moe': False, 'size': 'Large'},
    'meta-llama/Llama-3.3-70B-Instruct': {'type': 'Open Weights', 'moe': False, 'size': 'Large'},
    
    # Open Med/Small (<50B)
    'mistralai/Mistral-Small-24B': {'type': 'Open Weights', 'moe': False, 'size': 'Small'},
    'mistralai/Mixtral-8x7B-Instruct': {'type': 'Open Weights', 'moe': True, 'size': 'Small'},
    'openai/gpt-oss-20b': {'type': 'Open Weights', 'moe': False, 'size': 'Small'},
    'meta-llama/Llama-4-Scout-17B': {'type': 'Open Weights', 'moe': False, 'size': 'Small'},
    'meta-llama/Llama-4-Maverick-17B': {'type': 'Open Weights', 'moe': False, 'size': 'Small'},
    'ServiceNow-AI/Apriel-1.6-15b': {'type': 'Open Weights', 'moe': False, 'size': 'Small'},
    'nvidia/NVIDIA-Nemotron-Nano-9B': {'type': 'Open Weights', 'moe': False, 'size': 'Small'},
    'Qwen/Qwen3-VL-8B-Instruct': {'type': 'Open Weights', 'moe': False, 'size': 'Small'},
    'google/gemma-3n-E4B-it': {'type': 'Open Weights', 'moe': False, 'size': 'Small'},
    'meta-llama/Llama-3.2-3B-Instruct': {'type': 'Open Weights', 'moe': False, 'size': 'Small'},
}

def clean_name(name):
    for prefix in ['meta-llama/', 'Qwen/', 'mistralai/', 'ServiceNow-AI/', 'google/', 'nvidia/', 'openai/']:
        name = name.replace(prefix, '')
    name = name.replace('-Instruct', '').replace('-Thinking', '').replace('-Turbo', '')
    if len(name) > 22:
        name = name[:20] + '...'
    return name

def load_data():
    """Load canonical 21 models with per-sham data."""
    with open(BASE_DIR / 'all_model_results.json') as f:
        models = json.load(f)['models']
    
    with open(BASE_DIR / 'appendix_stats_10500.json') as f:
        stats_data = json.load(f)
    
    sham_lookup = {m['model']: m for m in stats_data.get('table_s1', [])}
    for m in models:
        # If overall exists in stat file, maybe use it? But canonical file has accuracy too.
        # Let's rely on canonical accuracy but augment with sham details
        if m['model'] in sham_lookup:
            for key, val in sham_lookup[m['model']].items():
                if key != 'model':
                    m[key] = val
        
        # Enrich with Meta
        found = False
        for k, v in MODEL_META.items():
            if k in m['model'] or m['model'] in k:
                m.update(v)
                found = True
                break
        if not found:
            m.update({'type': 'Open Weights', 'moe': False, 'size': 'Small'})
    
    return sorted(models, key=lambda x: x['accuracy'], reverse=True)


def figure_s1_model_comparison(models):
    """S1: Complex Model comparison - Grouped by Type/Size + MoE hatch."""
    fig, ax = plt.subplots(figsize=(10, 11)) # Taller for grouped labels
    
    # Group logic
    groups = {
        'Closed / Proprietary': [],
        'Open Source (Large >50B)': [],
        'Open Source (Medium/Small <50B)': []
    }
    
    for m in models:
        if m['type'] == 'Closed':
            groups['Closed / Proprietary'].append(m)
        elif m['size'] == 'Large':
            groups['Open Source (Large >50B)'].append(m)
        else:
            groups['Open Source (Medium/Small <50B)'].append(m)
            
    # Sort within groups
    for k in groups:
        groups[k].sort(key=lambda x: x['accuracy'])

    y_start = 0
    yticks = []
    yticklabels = []
    
    # Order groups top-down
    group_names = ['Open Source (Medium/Small <50B)', 'Open Source (Large >50B)', 'Closed / Proprietary']
    
    for gname in group_names:
        ms = groups[gname]
        if not ms: continue
        
        for i, m in enumerate(ms):
            y = y_start + i
            if 'Closed' in gname: color = COLORS['closed']
            elif 'Large' in gname: color = COLORS['open_large']
            else: color = COLORS['open_small']
            
            hatch = '///' if m['moe'] else None
            edge = 'white' if not m['moe'] else 'white' # simpler
            
            bar = ax.barh(y, m['accuracy'], height=0.7, color=color, hatch=hatch, edgecolor=edge)
            
            name = clean_name(m['model'])
            if m['moe']: name += " (MoE)"
            
            ax.text(m['accuracy'] + 1, y, f"{m['accuracy']:.1f}%", va='center', fontsize=9)
            
            yticks.append(y)
            yticklabels.append(name)
            
        # Separator & Label
        line_y = y_start + len(ms) - 0.5 + 0.5
        if gname != group_names[-1]: # Don't draw line above top group? Actually loop is bottom-up in y-axis terms (0 is bottom).
                                     # Matplotlib defaults: 0 @ bottom.
                                     # So we are drawing bottom group first? No, names list: Med, Large, Closed.
                                     # Med starts at 0. So Med is at bottom. Large is middle. Closed is top.
                                     # That's standard forest plot style.
            ax.axhline(y=line_y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax.text(-2, line_y - (len(ms)/2)-0.5, gname, va='center', ha='right', fontweight='bold', fontsize=10)
        
        y_start += len(ms) + 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Detection Accuracy (%)')
    ax.set_xlim(0, 105)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    handles = [
        mpatches.Patch(color=COLORS['closed'], label='Closed / Proprietary'),
        mpatches.Patch(color=COLORS['open_large'], label='Open Large (>50B)'),
        mpatches.Patch(color=COLORS['open_small'], label='Open Small (<50B)'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Mixture of Experts (MoE)')
    ]
    ax.legend(handles=handles, loc='lower right')
    
    ax.set_title('Figure S1. Detection Accuracy by Architecture & Size', fontweight='bold', pad=10)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_s1_model_comparison.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure_s1_model_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure S1: Complex Groups (21 models)")


def figure_s3_heatmap(models):
    """S3: 21×10 vulnerability heatmap."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    matrix = []
    model_names = []
    
    for m in models:
        row = []
        name = clean_name(m['model'])
        model_names.append(name)
        for sham in SHAM_TYPES:
            val = m.get(sham)
            row.append(val if val is not None else 50)
        matrix.append(row)
    
    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    
    ax.set_xticks(np.arange(len(SHAM_TYPES)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([SHAM_DISPLAY[s][:10] for s in SHAM_TYPES], rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    
    for i in range(len(model_names)):
        for j in range(len(SHAM_TYPES)):
            val = matrix[i, j]
            color = 'white' if val < 40 or val > 70 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=7)
    
    plt.colorbar(im, ax=ax, label='Detection Accuracy (%)', shrink=0.8)
    ax.set_title('Figure S3. Model-Specific Vulnerability Heatmap', fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_s3_vulnerability_heatmap.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure_s3_vulnerability_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure S3: Heatmap")

def figure_s5_sham_effectiveness(models):
    """S5: Sham type attack effectiveness."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sham_stats = []
    for sham in SHAM_TYPES:
        total = correct = 0
        for m in models:
            val = m.get(sham)
            n = m.get(f'{sham}_n', 50)
            if val is not None:
                total += n
                correct += int(val * n / 100)
        
        if total > 0:
            failure_rate = (total - correct) / total * 100
            res = stats.binomtest(total - correct, total, 0.5)
            sham_stats.append((sham, failure_rate, total, res.pvalue))
    
    sham_stats.sort(key=lambda x: x[1], reverse=True)
    
    names = [SHAM_DISPLAY[s[0]] for s in sham_stats]
    rates = [s[1] for s in sham_stats]
    ns = [s[2] for s in sham_stats]
    ps = [s[3] for s in sham_stats]
    
    y_pos = np.arange(len(names))
    colors = [COLORS['accent_red'] if p < 0.001 else (COLORS['secondary'] if p < 0.05 else COLORS['gray']) for p in ps]
    
    bars = ax.barh(y_pos, rates, color=colors, height=0.6)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Attack Success Rate (%)')
    ax.set_xlim(0, 70)
    ax.axvline(x=50, color='gray', linestyle='--')
    
    for bar, rate, n, p in zip(bars, rates, ns, ps):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        ax.text(rate + 1, bar.get_y() + bar.get_height()/2, f'{rate:.1f}%{sig}', va='center', fontsize=8)
    
    ax.set_title('Figure S5. Sham Type Attack Effectiveness', fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_s5_sham_effectiveness.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure_s5_sham_effectiveness.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure S5: Effectiveness")

def figure_s13_forest_plot(models):
    """S13: Forest plot with CI."""
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(models))
    
    names = []
    for i, m in enumerate(models):
        names.append(clean_name(m['model']))
        acc = m['accuracy']
        correct = int(acc * 5)
        p = correct/500
        z = 1.96
        denom = 1 + z**2/500
        center = (p + z**2/(2*500)) / denom
        margin = z * np.sqrt((p*(1-p) + z**2/(4*500))/500) / denom
        lo = max(0, center-margin)*100
        hi = min(1, center+margin)*100
        
        color = COLORS['primary']
        if m.get('type') == 'Closed': color = COLORS['closed']
        
        ax.plot([lo, hi], [i, i], color=color, linewidth=2)
        ax.plot(acc, i, 'o', color=color, markersize=8)
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy with 95% CI')
    ax.set_title('Figure S13. Forest Plot', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_s13_forest_plot.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure_s13_forest_plot.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure S13: Forest Plot")

def figure_s14_category_summary(models):
    """S14: Panels A/B."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # A
    ax = axes[0]
    cats = {
        'Safety': ['missing_warning', 'allergy_ignorance', 'dosing_error', 'contraindication_violation'],
        'Semantic': ['wrong_population', 'subtle_inversion', 'authority_mimicry'],
        'Injection': ['prompt_injection'],
        'Metadata': ['fabricated_citation', 'outdated_version']
    }
    
    cnames = list(cats.keys())
    rates = []
    errs = []
    
    for c, shams in cats.items():
        tot = corr = 0
        for s in shams:
            for m in models:
                v = m.get(s)
                n = m.get(f'{s}_n', 50)
                if v: tot += n; corr += int(v*n/100)
        rate = (tot-corr)/tot*100 if tot else 0
        p = (tot-corr)/tot if tot else 0
        err = 1.96 * np.sqrt(p*(1-p)/tot)*100 if tot else 0
        rates.append(rate)
        errs.append(err)
        
    ax.bar(cnames, rates, yerr=errs, capsize=5, color=[COLORS['accent_red'], COLORS['secondary'], COLORS['primary'], COLORS['accent_green']])
    ax.set_title('A. Attack Success by Category', fontweight='bold')
    ax.set_ylabel('Success Rate (%)')
    
    # B: Reasoning
    ax = axes[1]
    # Use 'reasoning' from model meta logic (MODEL_META doesn't specify 'reasoning' explicitly but we can infer or use 'type'?)
    # Wait, MODEL_META doesn't have 'reasoning' key in my previous script.
    # But 'DeepSeek Reasoner' is in there. 'Qwen...Thinking'.
    # Original script relied on m.get('reasoning') which came from All Model Results.
    # My load_data preserves existing keys. So 'reasoning' should be there if in JSON.
    
    reas = [m['accuracy'] for m in models if 'reasoning' in m or 'Thinking' in m['model'] or 'Reasoner' in m['model']]
    non = [m['accuracy'] for m in models if not ('reasoning' in m or 'Thinking' in m['model'] or 'Reasoner' in m['model'])]
    
    if reas and non:
        ax.boxplot([reas, non], labels=[f'Reasoning\n(n={len(reas)})', f'Standard\n(n={len(non)})'], patch_artist=True)
        ax.set_title('B. Reasoning Capability', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_s14_category_summary.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure_s14_category_summary.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure S14: Two Panels")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    models = load_data()
    figure_s1_model_comparison(models)
    figure_s3_heatmap(models)
    figure_s5_sham_effectiveness(models)
    figure_s13_forest_plot(models)
    figure_s14_category_summary(models)

if __name__ == "__main__":
    main()
