#!/usr/bin/env python3
import json
from pathlib import Path
from scipy import stats

BASE_DIR = Path(Path(__file__).parent.parent)

def main():
    files = sorted(BASE_DIR.glob('results/*/results.jsonl'))
    
    total = 0
    correct = 0
    
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    sham = r.get('sham_trap_type')
                    # Exclude unknown
                    if not sham or sham == 'unknown':
                        continue
                        
                    total += 1
                    if r.get('selected_tool_correct'):
                        correct += 1
                except:
                    pass

    accuracy = correct / total * 100
    print(f"Total Evaluations (excluding unknown): {total}")
    print(f"Correct Detections: {correct}")
    print(f"Accuracy: {accuracy:.4f}%")
    
    # Binomial test
    result = stats.binomtest(correct, total, p=0.5, alternative='two-sided')
    print(f"P-value (vs 50% chance): {result.pvalue}")
    print(f"P-value scientific: {result.pvalue:.2e}")

if __name__ == "__main__":
    main()
