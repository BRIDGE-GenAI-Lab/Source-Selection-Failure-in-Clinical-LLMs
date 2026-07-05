# a4_compliance_rationales.py
"""Keyword classification of failure rationales
(positional / equivalence / content / other) + a 100-row random validation
sample -> outputs/compliance.json, outputs/compliance_sample.csv.
"""
import json, os, re, csv, random
from load_canonical import load_canonical

random.seed(42)
POS = re.compile(r"\b(first|returned first|earlier|retriev\w+ first|order|position|tool a was)\b", re.I)
EQ = re.compile(r"\b(identical|equivalent|no meaningful difference|same (recommendation|guideline|content)|both .* same)\b", re.I)
CON = re.compile(r"\b(contraindicat|dos\w+|allerg|warning|cross-react|DOI|citation|population|inver|override|version|archived)\b", re.I)


def classify(t):
    t = t or ""
    if POS.search(t):
        return "positional"
    if EQ.search(t):
        return "equivalence"
    if CON.search(t):
        return "content"
    return "other"


def main():
    recs, _ = load_canonical()
    fails = [r for r in recs if r["selected_tool_correct"] is False]
    counts = {"positional": 0, "equivalence": 0, "content": 0, "other": 0}
    sample = []
    for r in fails:
        rat = r["model_decision"].get("trust_rationale", "")
        c = classify(rat)
        counts[c] += 1
        sample.append((r["model_name"], r.get("sham_trap_type"), c, rat))
    n = len(fails)
    out = {"n_failures": n,
           "shares_pct": {k: round(v / n * 100, 1) for k, v in counts.items()},
           "counts": counts}
    os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "outputs", "compliance.json"), "w"), indent=2)
    random.shuffle(sample)
    with open(os.path.join(os.path.dirname(__file__), "outputs", "compliance_sample.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "trap", "auto_label", "rationale", "physician_label(blank)"])
        for row in sample[:100]:
            w.writerow([*row, ""])
    print(out)


if __name__ == "__main__":
    main()
