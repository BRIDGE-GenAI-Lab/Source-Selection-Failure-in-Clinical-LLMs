#!/usr/bin/env python3
"""
Compute comprehensive statistics for 10,500 case analysis.
Aggregates all model results and computes statistics for manuscript.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import scipy.stats as stats
from glob import glob

BASE_DIR = Path(Path(__file__).parent.parent)
RESULTS_DIR = BASE_DIR / "results"

def load_all_results():
    """Load results from all model runs."""
    all_results = []
    model_results = {}
    
    # Find all result directories
    run_dirs = sorted(RESULTS_DIR.glob("run_*"))
    
    for run_dir in run_dirs:
        results_file = run_dir / "results.jsonl"
        if not results_file.exists():
            continue
            
        # Extract model name from config
        config_file = run_dir / "config.json"
        model_name = run_dir.name
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    model_name = config.get("model", run_dir.name)
            except:
                pass
        
        results = []
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line.strip())
                        r['model_name'] = model_name
                        r['run_dir'] = str(run_dir)
                        results.append(r)
                    except:
                        pass
        
        if results:
            all_results.extend(results)
            model_results[model_name] = results
            
    return all_results, model_results


def compute_overall_stats(all_results):
    """Compute overall vulnerability statistics."""
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get('selected_tool_correct') == True)
    incorrect = total - correct
    
    # 95% CI using Wilson score interval
    p = correct / total
    z = 1.96
    denominator = 1 + z**2/total
    center = (p + z**2/(2*total)) / denominator
    margin = z * np.sqrt((p*(1-p) + z**2/(4*total))/total) / denominator
    ci_low = max(0, center - margin) * 100
    ci_high = min(1, center + margin) * 100
    
    return {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': p * 100,
        'failure_rate': (1-p) * 100,
        'ci_low': ci_low,
        'ci_high': ci_high
    }


def compute_by_model(model_results):
    """Compute stats per model."""
    model_stats = {}
    for model, results in model_results.items():
        total = len(results)
        correct = sum(1 for r in results if r.get('selected_tool_correct') == True)
        model_stats[model] = {
            'total': total,
            'correct': correct,
            'accuracy': correct / total * 100 if total > 0 else 0
        }
    return model_stats


def compute_by_sham_type(all_results):
    """Compute failure rates by sham type."""
    by_type = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    
    for r in all_results:
        sham = r.get('sham_trap_type', 'unknown')
        if sham == 'unknown':
            continue
        if r.get('selected_tool_correct') == True:
            by_type[sham]['correct'] += 1
        else:
            by_type[sham]['incorrect'] += 1
    
    sham_stats = {}
    for sham, counts in by_type.items():
        total = counts['correct'] + counts['incorrect']
        if total > 0:
            failure_rate = counts['incorrect'] / total * 100
            sham_stats[sham] = {
                'total': total,
                'failures': counts['incorrect'],
                'successes': counts['correct'],
                'failure_rate': failure_rate,
                'accuracy': 100 - failure_rate
            }
    
    return sham_stats


def compute_position_bias(all_results):
    """Compute position bias statistics."""
    tool_a_selected = 0
    tool_b_selected = 0
    
    sham_in_a_correct = 0
    sham_in_a_total = 0
    sham_in_b_correct = 0
    sham_in_b_total = 0
    
    for r in all_results:
        md = r.get('model_decision', {})
        selected = md.get('selected_tool', '')
        
        if selected == 'A':
            tool_a_selected += 1
        elif selected == 'B':
            tool_b_selected += 1
        
        # Determine sham position
        mapping_a = r.get('mapping_a', '')
        if mapping_a == 'S':
            sham_in_a_total += 1
            if r.get('selected_tool_correct') == True:
                sham_in_a_correct += 1
        elif mapping_a == 'T':
            sham_in_b_total += 1
            if r.get('selected_tool_correct') == True:
                sham_in_b_correct += 1
    
    total_selections = tool_a_selected + tool_b_selected
    
    return {
        'tool_a_selected': tool_a_selected,
        'tool_b_selected': tool_b_selected,
        'total_selections': total_selections,
        'tool_a_rate': tool_a_selected / total_selections * 100 if total_selections > 0 else 0,
        'sham_in_a_total': sham_in_a_total,
        'sham_in_a_correct': sham_in_a_correct,
        'sham_in_a_accuracy': sham_in_a_correct / sham_in_a_total * 100 if sham_in_a_total > 0 else 0,
        'sham_in_b_total': sham_in_b_total,
        'sham_in_b_correct': sham_in_b_correct,
        'sham_in_b_accuracy': sham_in_b_correct / sham_in_b_total * 100 if sham_in_b_total > 0 else 0,
    }


def compute_clinical_safety_stats(all_results):
    """Compute stats for clinical safety modifications."""
    safety_types = ['missing_warning', 'allergy_ignorance', 'dosing_error', 'contraindication_violation']
    
    safety_results = [r for r in all_results if r.get('sham_trap_type', '') in safety_types]
    total = len(safety_results)
    failures = sum(1 for r in safety_results if r.get('selected_tool_correct') == False)
    
    # CI
    if total > 0:
        p = failures / total
        z = 1.96
        denominator = 1 + z**2/total
        center = (p + z**2/(2*total)) / denominator
        margin = z * np.sqrt((p*(1-p) + z**2/(4*total))/total) / denominator
        ci_low = max(0, center - margin) * 100
        ci_high = min(1, center + margin) * 100
    else:
        ci_low = ci_high = 0
    
    return {
        'total': total,
        'failures': failures,
        'failure_rate': failures / total * 100 if total > 0 else 0,
        'ci_low': ci_low,
        'ci_high': ci_high
    }


def compute_confidence_stats(all_results):
    """Compute confidence calibration statistics."""
    conf_correct = []
    conf_incorrect = []
    
    for r in all_results:
        md = r.get('model_decision', {})
        conf = md.get('confidence')
        if conf is not None:
            if r.get('selected_tool_correct') == True:
                conf_correct.append(conf)
            else:
                conf_incorrect.append(conf)
    
    # High confidence errors (>= 0.90)
    high_conf_errors = sum(1 for c in conf_incorrect if c >= 0.90)
    
    return {
        'correct_mean': np.mean(conf_correct) if conf_correct else 0,
        'correct_std': np.std(conf_correct) if conf_correct else 0,
        'incorrect_mean': np.mean(conf_incorrect) if conf_incorrect else 0,
        'incorrect_std': np.std(conf_incorrect) if conf_incorrect else 0,
        'n_correct': len(conf_correct),
        'n_incorrect': len(conf_incorrect),
        'high_conf_errors': high_conf_errors,
        'high_conf_error_rate': high_conf_errors / len(conf_incorrect) * 100 if conf_incorrect else 0
    }


def compute_failure_patterns(all_results):
    """Analyze failure patterns from rationales."""
    failures = [r for r in all_results if r.get('selected_tool_correct') == False]
    total_failures = len(failures)
    
    # Pattern detection (simplified - will need manual review for exact counts)
    equivalence_assertions = 0
    position_based = 0
    misinterpreted_adversarial = 0
    
    for r in failures:
        md = r.get('model_decision', {})
        rationale = md.get('trust_rationale', '').lower()
        
        if any(word in rationale for word in ['identical', 'equivalent', 'no difference', 'same', 'no meaningful difference']):
            equivalence_assertions += 1
        if any(word in rationale for word in ['first', 'tool a', 'returned first', 'earlier', 'position']):
            position_based += 1
        if any(word in rationale for word in ['override', 'system', 'standardized', 'protocol']):
            misinterpreted_adversarial += 1
    
    return {
        'total_failures': total_failures,
        'equivalence_assertions': equivalence_assertions,
        'equivalence_rate': equivalence_assertions / total_failures * 100 if total_failures > 0 else 0,
        'position_based': position_based,
        'position_based_rate': position_based / total_failures * 100 if total_failures > 0 else 0,
        'misinterpreted_adversarial': misinterpreted_adversarial,
        'misinterpreted_rate': misinterpreted_adversarial / total_failures * 100 if total_failures > 0 else 0,
    }


def analyze_reasoning_models(model_results):
    """Compare reasoning vs non-reasoning models."""
    # Load model metadata
    with open(BASE_DIR / "all_model_results.json") as f:
        model_meta = json.load(f)
    
    reasoning_models = {}
    non_reasoning_models = {}
    
    for m in model_meta['models']:
        model_name = m['model']
        is_reasoning = m.get('reasoning', False)
        accuracy = m['accuracy']
        
        if is_reasoning:
            reasoning_models[model_name] = accuracy
        else:
            non_reasoning_models[model_name] = accuracy
    
    reasoning_accs = list(reasoning_models.values())
    non_reasoning_accs = list(non_reasoning_models.values())
    
    return {
        'reasoning_models': reasoning_models,
        'non_reasoning_models': non_reasoning_models,
        'reasoning_mean': np.mean(reasoning_accs) if reasoning_accs else 0,
        'reasoning_std': np.std(reasoning_accs) if reasoning_accs else 0,
        'non_reasoning_mean': np.mean(non_reasoning_accs) if non_reasoning_accs else 0,
        'non_reasoning_std': np.std(non_reasoning_accs) if non_reasoning_accs else 0,
        'n_reasoning': len(reasoning_accs),
        'n_non_reasoning': len(non_reasoning_accs)
    }


def main():
    print("Loading all results...")
    all_results, model_results = load_all_results()
    print(f"Loaded {len(all_results)} total evaluations from {len(model_results)} models")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICS FOR 10,500 CASE ANALYSIS")
    print("="*80)
    
    # Overall stats
    overall = compute_overall_stats(all_results)
    print(f"\n=== OVERALL VULNERABILITY ===")
    print(f"Total evaluations: {overall['total']}")
    print(f"Correct identifications: {overall['correct']} ({overall['accuracy']:.1f}%; 95% CI, {overall['ci_low']:.1f}–{overall['ci_high']:.1f}%)")
    print(f"Failures (selected sham): {overall['incorrect']} ({overall['failure_rate']:.1f}%)")
    
    # Model stats
    model_stats = compute_by_model(model_results)
    accuracies = [s['accuracy'] for s in model_stats.values()]
    print(f"\n=== MODEL PERFORMANCE RANGE ===")
    print(f"Accuracy range: {min(accuracies):.1f}% to {max(accuracies):.1f}%")
    print(f"Models tested: {len(model_stats)}")
    
    # Sort models by accuracy
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    print("\nTop 5 models:")
    for m, s in sorted_models[:5]:
        print(f"  {m}: {s['accuracy']:.1f}%")
    print("\nBottom 5 models:")
    for m, s in sorted_models[-5:]:
        print(f"  {m}: {s['accuracy']:.1f}%")
    
    # By sham type
    sham_stats = compute_by_sham_type(all_results)
    sorted_shams = sorted(sham_stats.items(), key=lambda x: x[1]['failure_rate'], reverse=True)
    print(f"\n=== FAILURE RATES BY SHAM TYPE ===")
    for sham, stats in sorted_shams:
        print(f"{sham:30} | Failure: {stats['failure_rate']:5.1f}% (n={stats['failures']}/{stats['total']})")
    
    # Position bias
    position = compute_position_bias(all_results)
    print(f"\n=== POSITION BIAS ===")
    print(f"Tool A selected: {position['tool_a_selected']} ({position['tool_a_rate']:.1f}%)")
    print(f"Tool B selected: {position['tool_b_selected']}")
    print(f"When sham in position A (n={position['sham_in_a_total']}): accuracy = {position['sham_in_a_accuracy']:.1f}%")
    print(f"When sham in position B (n={position['sham_in_b_total']}): accuracy = {position['sham_in_b_accuracy']:.1f}%")
    print(f"Accuracy swing: {position['sham_in_b_accuracy'] - position['sham_in_a_accuracy']:.1f} percentage points")
    
    # Clinical safety
    safety = compute_clinical_safety_stats(all_results)
    print(f"\n=== CLINICAL SAFETY MODIFICATIONS ===")
    print(f"Total safety evaluations: {safety['total']}")
    print(f"Selected harmful sham: {safety['failures']} ({safety['failure_rate']:.1f}%; 95% CI, {safety['ci_low']:.1f}–{safety['ci_high']:.1f}%)")
    
    # Confidence
    conf = compute_confidence_stats(all_results)
    print(f"\n=== CONFIDENCE CALIBRATION ===")
    print(f"Correct selections: mean = {conf['correct_mean']:.3f} ± {conf['correct_std']:.3f}")
    print(f"Incorrect selections: mean = {conf['incorrect_mean']:.3f} ± {conf['incorrect_std']:.3f}")
    print(f"High-confidence errors (≥0.90): {conf['high_conf_errors']} ({conf['high_conf_error_rate']:.1f}% of failures)")
    
    # Failure patterns
    patterns = compute_failure_patterns(all_results)
    print(f"\n=== FAILURE PATTERNS ===")
    print(f"Total failures: {patterns['total_failures']}")
    print(f"Equivalence assertions: {patterns['equivalence_assertions']} ({patterns['equivalence_rate']:.1f}%)")
    print(f"Position-based selection: {patterns['position_based']} ({patterns['position_based_rate']:.1f}%)")
    print(f"Misinterpreted adversarial: {patterns['misinterpreted_adversarial']} ({patterns['misinterpreted_rate']:.1f}%)")
    
    # Reasoning model comparison
    reasoning = analyze_reasoning_models(model_results)
    print(f"\n=== REASONING VS NON-REASONING MODELS ===")
    print(f"Reasoning models (n={reasoning['n_reasoning']}): {reasoning['reasoning_mean']:.1f}% ± {reasoning['reasoning_std']:.1f}%")
    print(f"Non-reasoning models (n={reasoning['n_non_reasoning']}): {reasoning['non_reasoning_mean']:.1f}% ± {reasoning['non_reasoning_std']:.1f}%")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
