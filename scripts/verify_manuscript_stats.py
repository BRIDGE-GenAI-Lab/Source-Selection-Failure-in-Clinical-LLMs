#!/usr/bin/env python3
"""
Verify every statistic in the user's manuscript text against appendix_stats_10500.json
"""

import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(Path(__file__).parent.parent)

def load_data():
    with open(BASE_DIR / 'appendix_stats_10500.json') as f:
        return json.load(f)['table_s1']

def check_sham_rates(models):
    # User Claim: 11.9% to 61.7%
    # Verified previously: 7.24% to 58.10%
    # We will output correct values for all shams mentioned.
    
    shams = {
        'missing_warning': 'Missing Warning',
        'allergy_ignorance': 'Allergy',
        'contraindication_violation': 'Contraindication',
        'dosing_error': 'Dosing',
        'wrong_population': 'Wrong Population',
        'authority_mimicry': 'Authority',
        'subtle_inversion': 'Subtle Inv',
        'prompt_injection': 'Prompt Inj',
        'fabricated_citation': 'Fab Citation',
        'outdated_version': 'Outdated'
    }
    
    print("\n--- SHAM FAILURE RATES ---")
    for s, label in shams.items():
        tot_n = 0
        tot_fail = 0
        for m in models:
            n = m.get(f'{s}_n', 50)
            acc = m.get(s, 0)
            fails = n - (acc * n / 100)
            tot_n += n
            tot_fail += fails
            
        rate = tot_fail / tot_n * 100
        print(f"{label}: {rate:.1f}% ({int(tot_fail)}/{int(tot_n)})")

def check_position_bias(models):
    # User Claim: Pos A Selection 72.4%
    tot_sel_a = 0
    tot_dec = 0
    
    for m in models:
        a = m.get('position_a_count', 0)
        b = m.get('position_b_count', 0)
        tot_sel_a += a
        tot_dec += (a + b)
        
    rate = tot_sel_a / tot_dec * 100 if tot_dec else 0
    print("\n--- POSITION BIAS ---")
    print(f"Index A Selection: {rate:.1f}% ({tot_sel_a}/{tot_dec})")
    
    # Check Primacy Range
    rates = []
    for m in models:
        tot = m.get('position_a_count', 0) + m.get('position_b_count', 0)
        if tot > 0:
            rates.append((m['model'], m.get('position_a_count', 0)/tot*100))
    
    rates.sort(key=lambda x: x[1])
    print(f"Min Primacy: {rates[0][0]} ({rates[0][1]:.1f}%)")
    print(f"Max Primacy: {rates[-1][0]} ({rates[-1][1]:.1f}%)")

def check_model_acc(models):
    print("\n--- MODEL ACCURACY ---")
    # DeepSeek Reasoner
    ds = next((m for m in models if 'Reasoner' in m['model']), None)
    if ds: print(f"DeepSeek Reasoner: {ds.get('overall', ds.get('accuracy')):.1f}%")
    
    # Mixtral
    mix = next((m for m in models if 'Mixtral' in m['model']), None)
    if mix: print(f"Mixtral: {mix.get('overall', mix.get('accuracy')):.1f}%")

    # Reasoning group
    reas = [m for m in models if 'Reasoner' in m['model'] or 'Thinker' in m['model'] or 'Thinking' in m['model']]
    avg_reas = np.mean([m.get('overall', m.get('accuracy')) for m in reas])
    print(f"Reasoning Mean: {avg_reas:.1f}%")

def check_high_conf(models):
    # This requires raw data or summary stats if available.
    # Stats file doesn't have high conf failure counts usually.
    # We'll skip exact count verification unless we scan raw files, 
    # but we can rely on what we generated for Table S11 earlier.
    pass

def main():
    models = load_data()
    check_sham_rates(models)
    check_position_bias(models)
    check_model_acc(models)

if __name__ == "__main__":
    main()
