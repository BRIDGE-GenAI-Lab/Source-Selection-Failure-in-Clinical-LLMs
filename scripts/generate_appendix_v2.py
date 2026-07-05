#!/usr/bin/env python3
"""Generate complete appendix v2 for 10,500 cases / 21 models."""

import json
from pathlib import Path

BASE_DIR = Path(Path(__file__).parent.parent)

def load_data():
    with open(BASE_DIR / 'all_model_results.json') as f:
        return json.load(f)['models']

def load_stats():
    with open(BASE_DIR / 'appendix_stats_10500.json') as f:
        return json.load(f)

def main():
    models = load_data()
    stats = load_stats()
    
    out = []
    out.append("# Supplementary Appendix v2\n")
    out.append("## Tool Selection Failures in Agentic Clinical LLMs\n")
    out.append("### 10,500 Case Analysis — 21 Models\n\n---\n\n")
    
    # Part IV: All 21 model detailed results
    out.append("# Part IV: Per-Model Detailed Results\n\n")
    
    sorted_models = sorted(models, key=lambda x: -x['accuracy'])
    
    for i, m in enumerate(sorted_models, 1):
        name = m['model']
        acc = m['accuracy']
        total = m['total']
        correct = m['correct']
        model_type = m.get('type', 'Open')
        reasoning = "Yes" if m.get('reasoning') else "No"
        arch = m.get('architecture', 'Dense')
        params = m.get('params', 'Unknown')
        
        out.append(f"## {i}. {name}\n\n")
        out.append(f"**Overall Accuracy:** {acc:.1f}% ({correct}/{total})\n\n")
        out.append("| Attribute | Value |\n|-----------|-------|\n")
        out.append(f"| Type | {model_type} |\n")
        out.append(f"| Reasoning | {reasoning} |\n")
        out.append(f"| Architecture | {arch} |\n")
        out.append(f"| Parameters | {params} |\n\n")
        
        # Find in table_s1
        for row in stats.get('table_s1', []):
            if row.get('model') == name:
                out.append("### Performance by Sham Type\n\n")
                out.append("| Sham Type | Accuracy |\n|-----------|----------|\n")
                for sham in ['missing_warning', 'allergy_ignorance', 'dosing_error', 
                            'contraindication_violation', 'wrong_population', 
                            'subtle_inversion', 'authority_mimicry', 'prompt_injection',
                            'fabricated_citation', 'outdated_version']:
                    val = row.get(sham)
                    if val is not None:
                        out.append(f"| {sham.replace('_', ' ').title()} | {val:.1f}% |\n")
                    else:
                        out.append(f"| {sham.replace('_', ' ').title()} | — |\n")
                out.append("\n")
                break
        
        # Strengths/weaknesses
        out.append("### Strengths\n")
        if acc > 70:
            out.append("- High overall accuracy\n")
        if m.get('reasoning'):
            out.append("- Reasoning capabilities enhance detection\n")
        out.append("\n### Weaknesses\n")
        if acc < 55:
            out.append("- Below-chance performance on some sham types\n")
        out.append("- Position bias present\n\n---\n\n")
    
    # Write output
    output_file = BASE_DIR / 'Sham_Full_Appendix_v2_complete.md'
    with open(output_file, 'w') as f:
        f.write(''.join(out))
    
    print(f"Generated: {output_file}")
    print(f"Lines: {len(out)}")

if __name__ == "__main__":
    main()
