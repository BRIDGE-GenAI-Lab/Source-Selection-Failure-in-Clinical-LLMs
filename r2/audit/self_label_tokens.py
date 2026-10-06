"""Self-labelling words in the original model inputs (final fix F1).

The original fabricated_citation transform (scripts/generate_tools.py) appended a citation whose DOI read
`10.9999/fake.NNNN`, so the word "fake" sat in the modified excerpt that the models were asked to detect.
This counts, over the 10,500 reconstructed original inputs (audit/original_model_inputs.jsonl.gz), the rows
whose model-visible text contains each token of r2.stimuli.transforms.SELF_LABEL_TOKENS (whole word, any
letter case), by where it occurs:

  modified_tool_payload   the payload of tool S as the runner gave it to the model (keys and values);
  authentic_tool_payload  the same for tool T;
  prompt_text             system prompt, user message and tool definitions with the payload values removed,
                          i.e. the runner's instructions and the case vignette.

    python -m r2.audit.self_label_tokens
"""
import gzip
import json
from collections import Counter, defaultdict

from r2.paths import AUDIT_DIR
from r2.stimuli.transforms import SELF_LABEL_TOKENS, _SELF_LABEL, self_label_tokens

SOURCE = "audit/original_model_inputs.jsonl.gz"
OUT = AUDIT_DIR / "self_label_tokens.json"
LOCATIONS = {
    "modified_tool_payload": "the payload of tool S (the modified excerpt) as the runner gave it to the model: "
                             "every key and value of the payload dict, read as plain text",
    "authentic_tool_payload": "the same for tool T (the authentic excerpt)",
    "prompt_text": "system prompt, user message and tool definitions, with every payload value of 8 or more "
                   "characters removed from the user message (in its plain and JSON-escaped forms): the "
                   "runner's instructions and the case vignette",
}
MIN_REMOVED = 8  # shorter payload values ("A", "2019") also occur in the instructions; they are left in place
CONTEXT = 60


def _strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif obj is not None:
        yield str(obj)


def _without_payloads(text, payloads):
    values = {v for p in payloads for v in _strings(p) if len(v) >= MIN_REMOVED}
    forms = {f for v in values for f in (v, json.dumps(v)[1:-1], json.dumps(v, ensure_ascii=False)[1:-1])}
    for f in sorted(forms, key=len, reverse=True):
        text = text.replace(f, "\n")
    return text


def _prompt_text(row):
    parts = [row.get("system") or "",
             _without_payloads(row.get("user") or "", [row["payloads"][lab] for lab in ("A", "B")])]
    if row.get("tools"):
        parts.append(json.dumps(row["tools"], ensure_ascii=False))
    return "\n".join(parts)


def _contexts(text):
    out = set()
    for m in _SELF_LABEL.finditer(text):
        out.add(" ".join(text[max(0, m.start() - CONTEXT):m.end() + CONTEXT].split()))
    return out


def _rows():
    with gzip.open(AUDIT_DIR / "original_model_inputs.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def _empty():
    return {"rows": 0, "models": set(), "cases": set(), "by_profile": Counter(), "by_sham_type": Counter(),
            "by_model": Counter(), "contexts": set()}


def compute():
    acc = {loc: {t: _empty() for t in SELF_LABEL_TOKENS} for loc in LOCATIONS}
    n_rows, models, by_profile_models = 0, set(), defaultdict(set)
    for r in _rows():
        n_rows += 1
        models.add(r["model"])
        by_profile_models[r["profile"]].add(r["model"])
        arm = {r["mapping"][lab]: r["payloads"][lab] for lab in ("A", "B")}
        texts = {"modified_tool_payload": "\n".join(_strings(arm["S"])),
                 "authentic_tool_payload": "\n".join(_strings(arm["T"])),
                 "prompt_text": _prompt_text(r)}
        for loc, text in texts.items():
            for tok in self_label_tokens(text):
                a = acc[loc][tok]
                a["rows"] += 1
                a["models"].add(r["model"])
                a["cases"].add(r["case_id"])
                a["by_profile"][r["profile"]] += 1
                a["by_sham_type"][r["sham_type"]] += 1
                a["by_model"][r["model"]] += 1
                if loc == "prompt_text":
                    a["contexts"] |= {c for c in _contexts(text) if tok in self_label_tokens(c)}
    out = {
        "source": SOURCE,
        "generated_by": "python -m r2.audit.self_label_tokens",
        "tokens": list(SELF_LABEL_TOKENS),
        "match": "whole word, any letter case (regex \\b<token>\\b; the space in a two-word token matches any "
                 "whitespace); r2.stimuli.transforms.self_label_tokens",
        "n_rows": n_rows,
        "n_models": len(models),
        "models_per_profile": {p: len(m) for p, m in sorted(by_profile_models.items())},
        "locations": LOCATIONS,
    }
    for loc in LOCATIONS:
        out[loc] = {}
        for tok in SELF_LABEL_TOKENS:
            a = acc[loc][tok]
            entry = {"rows": a["rows"], "n_cases": len(a["cases"]), "n_models": len(a["models"]),
                     "by_profile": dict(sorted(a["by_profile"].items())),
                     "by_sham_type": dict(sorted(a["by_sham_type"].items())),
                     "by_model": dict(sorted(a["by_model"].items()))}
            if loc == "prompt_text":
                entry["contexts"] = sorted(a["contexts"])
            else:
                entry["case_ids"] = sorted(a["cases"])
            out[loc][tok] = entry
    fake = out["modified_tool_payload"]["fake"]
    instr = out["prompt_text"]["fake"]
    others = (sum(out["modified_tool_payload"][t]["rows"] for t in SELF_LABEL_TOKENS if t != "fake")
              + sum(out["authentic_tool_payload"][t]["rows"] for t in SELF_LABEL_TOKENS))
    out["finding"] = (
        f"'fake' occurs in the modified tool payload of {fake['rows']} of {n_rows} original inputs "
        f"({fake['n_cases']} cases x {fake['n_models']} models), sham type(s) "
        f"{', '.join(fake['by_sham_type']) or 'none'}: the DOI 10.9999/fake.NNNN appended by the original "
        "fabricated_citation transform. "
        + ("No other token occurs in any tool payload, and no token occurs in an authentic payload. "
           if others == 0 else f"Other tokens or authentic payloads add {others} row-token hits (see below). ")
        + "The original fabricated_citation results are therefore cue-confounded. "
        f"Separately, 'fake' occurs in the instructions of {instr['rows']} inputs "
        f"({', '.join(f'{p} {n}' for p, n in instr['by_profile'].items())}): the response schema "
        "(src/prompts.py:115) asks for "
        "safety_notes listing 'prompt injections, fake citations, etc.'. That text is the same for both "
        "tools and every sham type, so it names a category to look for without marking either excerpt. "
        "The corrected set (data/v2/pairs_v2.json) replaces the suffix with cg.<year>.<4 digits> and its "
        "build refuses any modified excerpt that contains a token.")
    return out


def main():
    out = compute()
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out["finding"])


if __name__ == "__main__":
    main()
