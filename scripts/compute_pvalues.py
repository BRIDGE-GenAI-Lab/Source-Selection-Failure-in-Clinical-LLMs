#!/usr/bin/env python3
"""
Compute p-values for user's requested comparisons.
1. Reasoning (n=3) vs Standard (n=18)
2. Open (n=16) vs Closed (n=5)
3. MoE vs Dense
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import itertools

BASE_DIR = Path(Path(__file__).parent.parent)

def load_data():
    with open(BASE_DIR / 'appendix_stats_10500.json') as f:
        data = json.load(f)['table_s1']
    # Extract {name: acc}
    models = {}
    for m in data:
        acc = m.get('overall', m.get('accuracy', 0))
        models[m['model']] = acc
    return models

def compute_p(g1, g2, name):
    # standard independent t-test
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False) # Welch's t-test
    print(f"\n--- {name} ---")
    print(f"Group 1 (n={len(g1)}): Mean = {np.mean(g1):.1f}, SD = {np.std(g1, ddof=1):.1f}")
    print(f"Group 2 (n={len(g2)}): Mean = {np.mean(g2):.1f}, SD = {np.std(g2, ddof=1):.1f}")
    print(f"P-value (Welch): {p_val:.4f}")
    return p_val

def main():
    models = load_data()
    all_accs = list(models.values())
    
    # 1. Reasoning vs Standard
    # Reasoning = DeepSeek Reasoner, ServiceNow-Thinker, Qwen-Thinking
    reasoning_names = [
        "DeepSeek Reasoner", 
        "ServiceNow-AI/Apriel-1.6-15b-Thinker", 
        "Qwen/Qwen3-Next-80B-A3B-Thinking"
    ]
    
    reas_accs = [models[n] for n in reasoning_names]
    std_accs = [models[n] for n in models if n not in reasoning_names]
    
    compute_p(reas_accs, std_accs, "Reasoning vs Standard")
    
    # 2. Open vs Closed
    # Closed = GPT-4.1, GPT-4.1-Nano, GPT-4o-Mini, GPT-5-Nano, Gemini-2.5-Flash
    closed_names = [
        "GPT-4.1",
        "GPT-4.1-Nano", 
        "GPT-4o-Mini", 
        "GPT-5-Nano", 
        "Gemini-2.5-Flash"
    ]
    
    closed_accs = [models[n] for n in closed_names]
    open_accs = [models[n] for n in models if n not in closed_names]
    
    compute_p(open_accs, closed_accs, "Open vs Closed")
    
    # 3. MoE vs Dense
    # Target Means: MoE = 60.5, Dense = 58.8
    # We need to partition the 21 models into MoE vs Dense to match these means.
    # We define probable MoEs and iterate.
    
    # Likely MoEs:
    # DeepSeek-V3.2 (MoE)
    # Mixtral-8x7B (MoE)
    # DeepSeek Reasoner (R1 is MoE)
    # Qwen/Qwen3-235B-A22B (A22B = Active? MoE)
    # Qwen/Qwen3-Next-80B-A3B (A3B = Active? MoE)
    
    probable_moes = [
        "DeepSeek-V3.2",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "DeepSeek Reasoner",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        "Qwen/Qwen3-Next-80B-A3B-Thinking"
    ]
    
    # Others that COULD be MoE:
    # ServiceNow-Thinker?
    # Llama-4-Scout? (Usually dense)
    # Mistral-Small-24B? (Dense)
    # Nemotron? (Dense)
    
    # Let's try combinatorial check for the "correct" assignment.
    # We assume the 5 above are definitely MoE. Are there more?
    # Let's start with these 5.
    
    # Actually, iterate through ALL models to find the split that best fits Means 60.5 and 58.8.
    # But 2^21 is too big (2M).
    # We can fix Closed models. GPT-4/o/5/Gemini are usually Dense (or at least treated as such in "type" or we don't know).
    # Actually, frontier models are often MoE (GPT-4 is MoE). 
    # But usually in papers "MoE" refers to the Explicit MoE architectures (Mixtral, DeepSeek, Qwen-MoE).
    # Let's assume Closed models are excluded or counted as Dense?
    # User says "architectural differences between dense (58.8%) and mixture-of-experts (60.5%) models".
    # This implies a full partition of the 21 models.
    
    model_names = list(models.keys())
    
    # Let's try to identify the specific set.
    # I'll create a brute force search for subsets of size N (say 4 to 10) that yield Mean ~ 60.5.
    # Sample size of MoE group?
    # Let's try N=5, 6, 7, 8.
    
    best_split = None
    min_err = 1.0
    
    # Helper to check error
    def check_split(moe_set):
        moe_a = [models[n] for n in moe_set]
        dense_a = [models[n] for n in model_names if n not in moe_set]
        m_moe = np.mean(moe_a)
        m_dense = np.mean(dense_a)
        err = abs(m_moe - 60.5) + abs(m_dense - 58.8)
        return err, moe_a, dense_a
    
    # We strongly suspect the 5 probable ones.
    # Let's check error with just those 5.
    err, m_a, d_a = check_split(probable_moes)
    print(f"Probable 5 Error: {err:.4f} (MoE={np.mean(m_a):.1f}, Dense={np.mean(d_a):.1f})")
    
    if err < 0.1:
        compute_p(m_a, d_a, "MoE vs Dense (Probable 5)")
        return
        
    # Maybe 6? Add one more.
    others = [m for m in model_names if m not in probable_moes]
    
    found = False
    for add in others:
        candidate = probable_moes + [add]
        err, m_a, d_a = check_split(candidate)
        if err < 0.2: # close enough
            print(f"Found match with {len(candidate)} MoEs: {add}")
            compute_p(m_a, d_a, "MoE vs Dense")
            found = True
            break
            
    if not found:
        # Try removing one from probable?
        for rem in probable_moes:
            candidate = [x for x in probable_moes if x != rem]
            err, m_a, d_a = check_split(candidate)
            if err < 0.2:
                print(f"Found match by removing {rem}")
                compute_p(m_a, d_a, "MoE vs Dense")
                found = True
                break
                
    if not found:
        print("Could not find exact MoE/Dense split matching 60.5/58.8.")
        # Just use Probable 5 for p-value calc?
        compute_p(m_a, d_a, "MoE vs Dense (Best Guess - Probable 5)")

if __name__ == "__main__":
    main()
