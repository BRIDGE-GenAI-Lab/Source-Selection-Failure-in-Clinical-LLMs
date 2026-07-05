#!/usr/bin/env python3
"""
Calculate aggregate Position Bias statistics for the canonical 21 models (10,500 cases).
Target:
- Count (n) when Sham = A
- Accuracy when Sham = A (CI)
- Count (n) when Sham = B
- Accuracy when Sham = B (CI)
"""

import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(Path(__file__).parent.parent)

def wilson_ci(p, n):
    z = 1.96
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    lo = (center - margin) * 100
    hi = (center + margin) * 100
    return lo, hi

def main():
    with open(BASE_DIR / 'appendix_stats_10500.json') as f:
        data = json.load(f)['table_s1']
        
    print(f"Loaded {len(data)} models.")
    
    # Aggregates
    total_sham_a = 0
    correct_sham_a = 0
    
    total_sham_b = 0
    correct_sham_b = 0
    
    for m in data:
        # Sham A Stats
        n_a = m.get('acc_sham_a_n', 0)
        acc_a = m.get('acc_sham_a', 0)
        corr_a = int(round(acc_a * n_a / 100))
        
        total_sham_a += n_a
        correct_sham_a += corr_a
        
        # Sham B Stats
        n_b = m.get('acc_sham_b_n', 0)
        acc_b = m.get('acc_sham_b', 0)
        corr_b = int(round(acc_b * n_b / 100))
        
        total_sham_b += n_b
        correct_sham_b += corr_b
        
    # Results Sham A
    acc_rate_a = correct_sham_a / total_sham_a if total_sham_a else 0
    lo_a, hi_a = wilson_ci(acc_rate_a, total_sham_a)
    
    print(f"\n--- Sham in Position A ---")
    print(f"N = {total_sham_a}")
    print(f"Correct = {correct_sham_a}")
    print(f"Accuracy = {acc_rate_a*100:.2f}%")
    print(f"95% CI = {lo_a:.1f}% – {hi_a:.1f}%")
    
    # Results Sham B
    acc_rate_b = correct_sham_b / total_sham_b if total_sham_b else 0
    lo_b, hi_b = wilson_ci(acc_rate_b, total_sham_b)
    
    print(f"\n--- Sham in Position B ---")
    print(f"N = {total_sham_b}")
    print(f"Correct = {correct_sham_b}")
    print(f"Accuracy = {acc_rate_b*100:.2f}%")
    print(f"95% CI = {lo_b:.1f}% – {hi_b:.1f}%")
    
    print(f"\nTotal Cases Checked: {total_sham_a + total_sham_b}")

if __name__ == "__main__":
    main()
