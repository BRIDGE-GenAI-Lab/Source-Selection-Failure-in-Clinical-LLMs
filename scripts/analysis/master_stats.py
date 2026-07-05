#!/usr/bin/env python3
"""Master statistics table — the single source of truth for every headline number
in the R1 manuscript, figures, supplement, and response letter.
Recomputes from the canonical 10,500-row dataset and writes outputs/master_stats.json."""
import json, os
import numpy as np
from collections import defaultdict
from load_canonical import load_canonical, sham_position

CAT = {'missing_warning': 'safety', 'allergy_ignorance': 'safety', 'dosing_error': 'safety',
       'contraindication_violation': 'safety', 'wrong_population': 'semantic',
       'subtle_inversion': 'semantic', 'authority_mimicry': 'semantic',
       'prompt_injection': 'injection', 'fabricated_citation': 'metadata',
       'outdated_version': 'metadata'}

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (round(max(0, c - m) * 100, 1), round(min(1, c + m) * 100, 1))

def main():
    recs, by_model = load_canonical()
    n = len(recs)
    corr = sum(1 for r in recs if r['selected_tool_correct'] is True)
    out = {'n': n, 'accuracy': round(corr / n * 100, 1),
           'failure': round((n - corr) / n * 100, 1), 'acc_ci': wilson(corr, n),
           'sham_A': sum(1 for r in recs if sham_position(r) == 'A'),
           'sham_B': sum(1 for r in recs if sham_position(r) == 'B')}
    tt = defaultdict(lambda: [0, 0])
    for r in recs:
        t = r['sham_trap_type']; tt[t][1] += 1
        if r['selected_tool_correct'] is False: tt[t][0] += 1
    out['per_type'] = {t: {'fail': f, 'total': n2, 'fail_pct': round(f / n2 * 100, 1),
                           'ci': wilson(f, n2)} for t, (f, n2) in tt.items()}
    cc = defaultdict(lambda: [0, 0])
    for r in recs:
        c = CAT[r['sham_trap_type']]; cc[c][1] += 1
        if r['selected_tool_correct'] is False: cc[c][0] += 1
    out['per_category'] = {c: {'fail': f, 'total': n2, 'fail_pct': round(f / n2 * 100, 1),
                               'ci': wilson(f, n2)} for c, (f, n2) in cc.items()}
    out['safety'] = out['per_category']['safety']
    outdir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, 'master_stats.json'), 'w'), indent=2)
    # GATE
    assert sum(v['total'] for v in out['per_type'].values()) == 10500, "per-type total != 10500"
    assert out['per_category']['safety']['total'] == 4410, "safety total != 4410"
    assert abs(out['accuracy'] - 59.4) < 0.5, "accuracy drift"
    print(json.dumps(out, indent=2))
    print("GATE PASS")

if __name__ == "__main__":
    main()
