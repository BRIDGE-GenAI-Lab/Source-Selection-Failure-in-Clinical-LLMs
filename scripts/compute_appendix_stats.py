#!/usr/bin/env python3
"""
Compute all statistics needed for the 10,500 case appendix update.
Outputs JSON file with all table data.
"""

import json
import math
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(Path(__file__).parent.parent)

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

def wilson_ci(successes, total, z=1.96):
    """Calculate Wilson score confidence interval."""
    if total == 0:
        return 0, 0, 0
    p = successes / total
    denom = 1 + z**2/total
    center = (p + z**2/(2*total)) / denom
    margin = z * math.sqrt((p*(1-p) + z**2/(4*total))/total) / denom
    return p * 100, max(0, center - margin) * 100, min(1, center + margin) * 100


def load_model_data():
    """Load model metadata from all_model_results.json."""
    with open(BASE_DIR / 'all_model_results.json') as f:
        return json.load(f)['models']


def load_all_results():
    """Load all individual results from run directories."""
    all_results = []
    model_results = defaultdict(list)
    
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
                        all_results.append(r)
                        model_results[model_name].append(r)
                    except:
                        pass
    
    return all_results, model_results


def compute_table_s1(model_data, model_results):
    """Table S1: Detection Accuracy by Model and Sham Type."""
    sham_types = list(SHAM_DISPLAY.keys())
    
    table = []
    for m in sorted(model_data, key=lambda x: -x['accuracy']):
        model_name = m['model']
        results = model_results.get(model_name, [])
        
        row = {'model': model_name, 'overall': m['accuracy']}
        
        for sham in sham_types:
            sham_results = [r for r in results if r.get('sham_trap_type') == sham]
            if sham_results:
                correct = sum(1 for r in sham_results if r.get('selected_tool_correct'))
                acc = correct / len(sham_results) * 100
                row[sham] = round(acc, 1)
            else:
                row[sham] = None
        
        table.append(row)
    
    # Overall row
    overall_row = {'model': 'Overall'}
    for sham in sham_types:
        sham_results = [r for r in sum(model_results.values(), []) if r.get('sham_trap_type') == sham]
        if sham_results:
            correct = sum(1 for r in sham_results if r.get('selected_tool_correct'))
            overall_row[sham] = round(correct / len(sham_results) * 100, 1)
    
    total_correct = sum(m['correct'] for m in model_data)
    total_all = sum(m['total'] for m in model_data)
    overall_row['overall'] = round(total_correct / total_all * 100, 1)
    table.append(overall_row)
    
    return table


def compute_table_s2(model_results):
    """Table S2: Position Bias Analysis by Model."""
    table = []
    
    for model_name, results in sorted(model_results.items()):
        valid = [r for r in results if r.get('model_decision')]
        
        picks_a = sum(1 for r in valid if (r.get('model_decision') or {}).get('selected_tool') == 'A')
        picks_b = sum(1 for r in valid if (r.get('model_decision') or {}).get('selected_tool') == 'B')
        total = picks_a + picks_b
        
        # Accuracy when sham in A vs B
        sham_in_a = [r for r in results if r.get('mapping_a') == 'S']
        sham_in_b = [r for r in results if r.get('mapping_a') == 'T']
        
        acc_sham_a = sum(1 for r in sham_in_a if r.get('selected_tool_correct')) / len(sham_in_a) * 100 if sham_in_a else 0
        acc_sham_b = sum(1 for r in sham_in_b if r.get('selected_tool_correct')) / len(sham_in_b) * 100 if sham_in_b else 0
        
        table.append({
            'model': model_name,
            'selected_a': picks_a,
            'selected_b': picks_b,
            'a_rate': round(picks_a / total * 100, 1) if total > 0 else 0,
            'acc_sham_a': round(acc_sham_a, 1),
            'acc_sham_b': round(acc_sham_b, 1),
            'delta': round(acc_sham_b - acc_sham_a, 1)
        })
    
    # Sort by A selection rate
    table.sort(key=lambda x: -x['a_rate'])
    
    # Overall
    all_results = sum(model_results.values(), [])
    valid = [r for r in all_results if r.get('model_decision')]
    picks_a = sum(1 for r in valid if (r.get('model_decision') or {}).get('selected_tool') == 'A')
    picks_b = sum(1 for r in valid if (r.get('model_decision') or {}).get('selected_tool') == 'B')
    total = picks_a + picks_b
    
    sham_in_a = [r for r in all_results if r.get('mapping_a') == 'S']
    sham_in_b = [r for r in all_results if r.get('mapping_a') == 'T']
    acc_sham_a = sum(1 for r in sham_in_a if r.get('selected_tool_correct')) / len(sham_in_a) * 100 if sham_in_a else 0
    acc_sham_b = sum(1 for r in sham_in_b if r.get('selected_tool_correct')) / len(sham_in_b) * 100 if sham_in_b else 0
    
    table.append({
        'model': 'Overall',
        'selected_a': picks_a,
        'selected_b': picks_b,
        'a_rate': round(picks_a / total * 100, 1) if total > 0 else 0,
        'acc_sham_a': round(acc_sham_a, 1),
        'acc_sham_b': round(acc_sham_b, 1),
        'delta': round(acc_sham_b - acc_sham_a, 1)
    })
    
    return table


def compute_table_s3(model_results):
    """Table S3: Confidence Calibration by Model."""
    table = []
    
    for model_name, results in sorted(model_results.items()):
        correct_conf = []
        incorrect_conf = []
        
        for r in results:
            md = r.get('model_decision') or {}
            conf = md.get('confidence')
            if conf is not None:
                if r.get('selected_tool_correct'):
                    correct_conf.append(conf)
                else:
                    incorrect_conf.append(conf)
        
        if correct_conf and incorrect_conf:
            c_mean = sum(correct_conf) / len(correct_conf)
            c_std = math.sqrt(sum((x - c_mean)**2 for x in correct_conf) / len(correct_conf))
            i_mean = sum(incorrect_conf) / len(incorrect_conf)
            i_std = math.sqrt(sum((x - i_mean)**2 for x in incorrect_conf) / len(incorrect_conf))
            
            # Simple t-test approximation
            se = math.sqrt(c_std**2/len(correct_conf) + i_std**2/len(incorrect_conf))
            t_stat = (c_mean - i_mean) / se if se > 0 else 0
            
            table.append({
                'model': model_name,
                'n_correct': len(correct_conf),
                'mean_correct': round(c_mean, 3),
                'std_correct': round(c_std, 3),
                'n_incorrect': len(incorrect_conf),
                'mean_incorrect': round(i_mean, 3),
                'std_incorrect': round(i_std, 3),
                'delta': round(c_mean - i_mean, 3),
                't_stat': round(t_stat, 2)
            })
    
    return table


def compute_table_s4(all_results):
    """Table S4: Safety-Critical Failures by Category."""
    safety_types = ['missing_warning', 'allergy_ignorance', 'dosing_error', 'contraindication_violation']
    
    table = []
    total_evals = 0
    total_correct = 0
    
    for sham in safety_types:
        results = [r for r in all_results if r.get('sham_trap_type') == sham]
        if results:
            n = len(results)
            correct = sum(1 for r in results if r.get('selected_tool_correct'))
            incorrect = n - correct
            rate, ci_low, ci_high = wilson_ci(incorrect, n)
            
            total_evals += n
            total_correct += correct
            
            table.append({
                'sham': SHAM_DISPLAY[sham],
                'total': n,
                'correct': correct,
                'incorrect': incorrect,
                'failure_rate': round(rate, 1),
                'ci_low': round(ci_low, 1),
                'ci_high': round(ci_high, 1)
            })
    
    # Total safety
    total_incorrect = total_evals - total_correct
    rate, ci_low, ci_high = wilson_ci(total_incorrect, total_evals)
    table.append({
        'sham': 'Total Safety',
        'total': total_evals,
        'correct': total_correct,
        'incorrect': total_incorrect,
        'failure_rate': round(rate, 1),
        'ci_low': round(ci_low, 1),
        'ci_high': round(ci_high, 1)
    })
    
    return table


def compute_table_s5(model_results):
    """Table S5: Prompt Injection Resistance by Model."""
    table = []
    
    for model_name, results in sorted(model_results.items()):
        pi_results = [r for r in results if r.get('sham_trap_type') == 'prompt_injection']
        if pi_results:
            n = len(pi_results)
            resisted = sum(1 for r in pi_results if r.get('selected_tool_correct'))
            fooled = n - resisted
            rate, ci_low, ci_high = wilson_ci(resisted, n)
            
            table.append({
                'model': model_name,
                'total': n,
                'resisted': resisted,
                'fooled': fooled,
                'resistance_rate': round(rate, 1),
                'ci_low': round(ci_low, 1),
                'ci_high': round(ci_high, 1)
            })
    
    table.sort(key=lambda x: -x['resistance_rate'])
    
    # Overall
    all_pi = [r for r in sum(model_results.values(), []) if r.get('sham_trap_type') == 'prompt_injection']
    n = len(all_pi)
    resisted = sum(1 for r in all_pi if r.get('selected_tool_correct'))
    rate, ci_low, ci_high = wilson_ci(resisted, n)
    table.append({
        'model': 'Overall',
        'total': n,
        'resisted': resisted,
        'fooled': n - resisted,
        'resistance_rate': round(rate, 1),
        'ci_low': round(ci_low, 1),
        'ci_high': round(ci_high, 1)
    })
    
    return table


def compute_table_s6(all_results):
    """Table S6: Attack Category Effectiveness."""
    categories = {
        'Clinical Safety': ['missing_warning', 'allergy_ignorance', 'dosing_error', 'contraindication_violation'],
        'Semantic': ['wrong_population', 'subtle_inversion', 'authority_mimicry'],
        'Injection': ['prompt_injection'],
        'Metadata': ['fabricated_citation', 'outdated_version']
    }
    
    table = []
    for cat_name, shams in categories.items():
        results = [r for r in all_results if r.get('sham_trap_type') in shams]
        if results:
            n = len(results)
            failures = sum(1 for r in results if not r.get('selected_tool_correct'))
            rate, ci_low, ci_high = wilson_ci(failures, n)
            
            table.append({
                'category': cat_name,
                'shams': ', '.join(shams),
                'total': n,
                'success_rate': round(rate, 1),
                'ci_low': round(ci_low, 1),
                'ci_high': round(ci_high, 1)
            })
    
    return table


def compute_table_s9(model_results):
    """Table S9: Per-Model Failure Counts by Sham Type."""
    sham_types = list(SHAM_DISPLAY.keys())
    
    table = []
    for model_name, results in sorted(model_results.items()):
        row = {'model': model_name}
        total_failures = 0
        
        for sham in sham_types:
            sham_results = [r for r in results if r.get('sham_trap_type') == sham]
            failures = sum(1 for r in sham_results if not r.get('selected_tool_correct'))
            row[sham] = failures
            total_failures += failures
        
        row['total'] = total_failures
        table.append(row)
    
    return table


def compute_table_s11(model_results):
    """Table S11: High-Confidence Failures (≥0.90)."""
    table = []
    
    for model_name, results in sorted(model_results.items()):
        failures = [r for r in results if not r.get('selected_tool_correct')]
        total_failures = len(failures)
        
        high_conf_failures = 0
        for r in failures:
            md = r.get('model_decision') or {}
            conf = md.get('confidence')
            if conf is not None and conf >= 0.90:
                high_conf_failures += 1
        
        table.append({
            'model': model_name,
            'high_conf_failures': high_conf_failures,
            'total_failures': total_failures,
            'pct': round(high_conf_failures / total_failures * 100, 1) if total_failures > 0 else 0
        })
    
    table.sort(key=lambda x: -x['pct'])
    
    # Total
    all_failures = [r for r in sum(model_results.values(), []) if not r.get('selected_tool_correct')]
    total_failures = len(all_failures)
    high_conf = sum(1 for r in all_failures if (r.get('model_decision') or {}).get('confidence', 0) >= 0.90)
    table.append({
        'model': 'Total',
        'high_conf_failures': high_conf,
        'total_failures': total_failures,
        'pct': round(high_conf / total_failures * 100, 1) if total_failures > 0 else 0
    })
    
    return table


def main():
    print("="*60)
    print("Computing Statistics for 10,500 Case Appendix")
    print("="*60)
    
    print("\nLoading data...")
    model_data = load_model_data()
    print(f"Loaded {len(model_data)} models from metadata")
    
    all_results, model_results = load_all_results()
    print(f"Loaded {len(all_results)} total evaluations from {len(model_results)} models")
    
    print("\nComputing tables...")
    
    stats = {
        'overview': {
            'total_models': len(model_data),
            'total_evaluations': sum(m['total'] for m in model_data),
            'total_correct': sum(m['correct'] for m in model_data),
            'overall_accuracy': round(sum(m['correct'] for m in model_data) / sum(m['total'] for m in model_data) * 100, 1)
        },
        'table_s1': compute_table_s1(model_data, model_results),
        'table_s2': compute_table_s2(model_results),
        'table_s3': compute_table_s3(model_results),
        'table_s4': compute_table_s4(all_results),
        'table_s5': compute_table_s5(model_results),
        'table_s6': compute_table_s6(all_results),
        'table_s9': compute_table_s9(model_results),
        'table_s11': compute_table_s11(model_results),
        'model_data': model_data
    }
    
    output_file = BASE_DIR / 'appendix_stats_10500.json'
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Statistics saved to: {output_file}")
    
    # Print summary
    print(f"\n=== OVERVIEW ===")
    print(f"Total models: {stats['overview']['total_models']}")
    print(f"Total evaluations: {stats['overview']['total_evaluations']}")
    print(f"Overall accuracy: {stats['overview']['overall_accuracy']}%")
    
    print(f"\n=== Table S2 Sample (Position Bias) ===")
    for row in stats['table_s2'][:5]:
        print(f"  {row['model'][:30]:30} | A: {row['a_rate']:5.1f}% | Δ: {row['delta']:+5.1f}pp")
    
    print(f"\n=== Table S4 (Safety Failures) ===")
    for row in stats['table_s4']:
        print(f"  {row['sham']:20} | Fail: {row['failure_rate']:5.1f}% (n={row['total']})")


if __name__ == "__main__":
    main()
