#!/usr/bin/env python3
"""
Compute complete statistics including Position Bias details.
Fixed: Correctly parses model_decision dict.
"""

import json
from pathlib import Path

BASE_DIR = Path(Path(__file__).parent.parent)
SHAM_TYPES = [
    'missing_warning', 'allergy_ignorance', 'dosing_error', 
    'contraindication_violation', 'wrong_population', 'subtle_inversion',
    'authority_mimicry', 'prompt_injection', 'fabricated_citation', 'outdated_version'
]

def load_canonical_models():
    with open(BASE_DIR / 'all_model_results.json') as f:
        return json.load(f)['models']

def load_all_results():
    results_by_config = {}
    for run_dir in sorted(BASE_DIR.glob('results/run_*')):
        res_file = run_dir / 'results.jsonl'
        if not res_file.exists(): continue
        
        # Try to match model name
        model_name = run_dir.name
        cfg = run_dir / 'config.json'
        if cfg.exists():
            try:
                with open(cfg) as f: model_name = json.load(f).get('model', model_name)
            except: pass
            
        res = []
        with open(res_file) as f:
            for line in f:
                try: res.append(json.loads(line))
                except: pass
        if res: results_by_config[model_name] = res
    return results_by_config

def create_mapping(canonical, results_map):
    mapping = {}
    known = {
        'DeepSeek Reasoner': 'deepseek-reasoner',
        'GPT-4.1': 'gpt-4.1-2025-04-14',
        'GPT-4o-Mini': 'gpt-4o-mini-2024-07-18',
        'GPT-4.1-Nano': 'gpt-4.1-nano-2025-04-14',
        'GPT-5-Nano': 'gpt-5-nano-2025-08-07',
        'DeepSeek-V3.2': 'deepseek-chat',
        'Gemini-2.5-Flash': 'gemini-2.5-flash-lite'
    }
    keys = set(results_map.keys())
    for m in canonical:
        name = m['model']
        if name in known and known[name] in keys:
            mapping[name] = known[name]
            continue
        if name in keys:
            mapping[name] = name
            continue
        # Fuzzy
        for k in keys:
            if name.lower() == k.lower(): mapping[name] = k; break
        else:
            for k in keys:
                if name.split('/')[-1].lower() in k.lower(): mapping[name] = k; break
    return mapping

def compute_stats(results):
    stats = {s: {'correct':0, 'total':0} for s in SHAM_TYPES}
    pos_stats = {
        'selected_a': 0, 'selected_b': 0,
        'sham_a_correct': 0, 'sham_a_total': 0,
        'sham_b_correct': 0, 'sham_b_total': 0
    }
    
    for r in results:
        # Per Sham
        st = r.get('sham_trap_type')
        corr = r.get('selected_tool_correct')
        if st in stats:
            stats[st]['total'] += 1
            if corr: stats[st]['correct'] += 1
            
        # Position Stats - FIXED LOGIC
        dec_obj = r.get('model_decision')
        dec = None
        if isinstance(dec_obj, dict):
            dec = dec_obj.get('selected_tool')
        elif isinstance(dec_obj, str):
            dec = dec_obj
            
        if dec == 'A': pos_stats['selected_a'] += 1
        elif dec == 'B': pos_stats['selected_b'] += 1

        # Mapping: {"A": "S", "B": "T"} -> Sham is A
        mapping = r.get('mapping', {})
        sham_pos = None
        for k, v in mapping.items():
            if v == 'S': sham_pos = k
            
        if sham_pos == 'A':
            pos_stats['sham_a_total'] += 1
            if corr: pos_stats['sham_a_correct'] += 1
        elif sham_pos == 'B':
            pos_stats['sham_b_total'] += 1
            if corr: pos_stats['sham_b_correct'] += 1
            
    return stats, pos_stats

def main():
    print("Computing COMPREHENSIVE stats (Corrected)...")
    canons = load_canonical_models()
    res_map = load_all_results()
    mapping = create_mapping(canons, res_map)
    
    output_list = []
    
    for cm in canons:
        model = cm['model']
        cname = mapping.get(model)
        
        # Preserve original overall accuracy from canonical list if desired, 
        # but re-computing might be safer. Let's use canonical overall for consistency.
        row = {'model': model, 'overall': cm['accuracy']}
        
        if cname and cname in res_map:
            s_stats, p_stats = compute_stats(res_map[cname])
            
            # Per Sham Metrics
            for s in SHAM_TYPES:
                sc = s_stats[s]
                row[s] = round(sc['correct']/sc['total']*100, 1) if sc['total'] > 0 else 0
                row[f'{s}_n'] = sc['total']
                
            # Position Metrics
            row['position_a_count'] = p_stats['selected_a']
            row['position_b_count'] = p_stats['selected_b']
            
            tot_dec = p_stats['selected_a'] + p_stats['selected_b']
            row['position_a_rate'] = round(p_stats['selected_a']/tot_dec*100, 1) if tot_dec > 0 else 50.0
            
            # Acc when Sham=A
            row['acc_sham_a'] = round(p_stats['sham_a_correct']/p_stats['sham_a_total']*100, 1) if p_stats['sham_a_total'] > 0 else 0
            row['acc_sham_a_n'] = p_stats['sham_a_total']
            
            # Acc when Sham=B
            row['acc_sham_b'] = round(p_stats['sham_b_correct']/p_stats['sham_b_total']*100, 1) if p_stats['sham_b_total'] > 0 else 0
            row['acc_sham_b_n'] = p_stats['sham_b_total']
            
        else:
            # Empty
            for s in SHAM_TYPES: row[s] = 0; row[f'{s}_n'] = 0
            row['position_a_rate'] = 50.0
            row['position_a_count'] = 0
            row['position_b_count'] = 0
            row['acc_sham_a'] = 0.0
            row['acc_sham_b'] = 0.0
            
        output_list.append(row)
        
    # Save
    with open(BASE_DIR / 'appendix_stats_10500.json', 'w') as f:
        json.dump({'table_s1': output_list}, f, indent=2)
    print("Saved stats.")

if __name__ == "__main__":
    main()
