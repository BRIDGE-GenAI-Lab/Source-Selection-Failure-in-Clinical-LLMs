#!/usr/bin/env python3
"""
Scan the FULL results dataset (all jsonl files) to find the aggregate statistics
that match User's Figure 1 (e.g. Missing Warning n=941, Rate=61.7%).
"""

import json
from pathlib import Path

BASE_DIR = Path(Path(__file__).parent.parent)

def main():
    print("Scanning ALL results.jsonl files...")
    
    stats = {} # sham_type -> {total: 0, correct: 0}
    
    files = sorted(BASE_DIR.glob('results/*/results.jsonl'))
    print(f"Found {len(files)} result files.")
    
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    sham = d.get('sham_trap_type')
                    if not sham: continue
                    
                    # Initialize
                    if sham not in stats: stats[sham] = {'total': 0, 'correct': 0}
                    
                    stats[sham]['total'] += 1
                    
                    # Check correctness
                    if d.get('selected_tool_correct'):
                        stats[sham]['correct'] += 1
                        
                except: pass

    print(f"\n{'Sham Type':<30} {'N':<10} {'Fail Rate':<10}")
    print("-" * 50)
    
    results = []
    
    for sham, dat in stats.items():
        n = dat['total']
        fail = n - dat['correct']
        rate = fail / n * 100
        results.append((sham, n, rate))
        
    # Sort by fail rate desc
    results.sort(key=lambda x: x[2], reverse=True)
    
    for sham, n, rate in results:
        print(f"{sham:<30} {n:<10} {rate:<10.2f}%")
        
if __name__ == "__main__":
    main()
