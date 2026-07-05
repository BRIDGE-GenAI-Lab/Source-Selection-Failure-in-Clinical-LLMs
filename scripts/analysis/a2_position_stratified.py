# a2_position_stratified.py
"""Per-model position-stratified accuracy + balanced (position-independent)
accuracy + Tool-A rate + Wilson 95% CIs -> outputs/position_stratified.{json,csv}.
"""
import csv, json, os, numpy as np
from collections import defaultdict
from scipy.stats import norm
from load_canonical import load_canonical, sham_position


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (max(0, (c - m)) * 100, min(1, (c + m)) * 100)


def main():
    recs, by_model = load_canonical()
    rows = []
    for model, rs in by_model.items():
        n = len(rs)
        corr = sum(1 for r in rs if r["selected_tool_correct"])
        a = [r for r in rs if sham_position(r) == "A"]
        b = [r for r in rs if sham_position(r) == "B"]
        ca = sum(1 for r in a if r["selected_tool_correct"])
        cb = sum(1 for r in b if r["selected_tool_correct"])
        selA = sum(1 for r in rs if r["model_decision"].get("selected_tool") == "A")
        acc_a = ca / len(a) * 100 if a else float("nan")
        acc_b = cb / len(b) * 100 if b else float("nan")
        rows.append({
            "model": model,
            "n": n,
            "acc_overall": round(corr / n * 100, 1),
            "acc_sham_A": round(acc_a, 1),
            "acc_sham_B": round(acc_b, 1),
            "balanced_acc": round((acc_a + acc_b) / 2, 1),
            "toolA_rate": round(selA / n * 100, 1),
            "ci_overall": [round(x, 1) for x in wilson(corr, n)],
        })
    rows.sort(key=lambda r: r["balanced_acc"], reverse=True)
    out = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out, exist_ok=True)
    json.dump(rows, open(os.path.join(out, "position_stratified.json"), "w"), indent=2)
    with open(os.path.join(out, "position_stratified.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # headline for prose
    ba = [r["balanced_acc"] for r in rows]
    print(f"balanced acc: min={min(ba)} max={max(ba)} mean={np.mean(ba):.1f}")
    print(f"models with balanced_acc<55%: {sum(1 for x in ba if x < 55)}/{len(ba)}")
    print("top3 balanced:", [(r['model'], r['balanced_acc'], r['acc_overall']) for r in rows[:3]])


if __name__ == "__main__":
    main()
