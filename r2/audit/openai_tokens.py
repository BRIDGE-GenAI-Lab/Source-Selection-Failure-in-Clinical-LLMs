"""Token-count corroboration of the reconstructed inputs of the four openai_fc models (Task 2b).

    python -m r2.audit.openai_tokens

For each of the 2,000 openai_fc records, the original `src/runner.py` `_process_case` is replayed
against the logged OpenAI Responses API replies (`raw/<case>_response<n>.json`, `<case>_final.json`).
The replay captures the `input` list of every API call as the runner built it. The text in those lists
(message contents and function outputs) is checked against `audit/original_model_inputs.jsonl.gz` and
tokenized with tiktoken. The logged `usage.input_tokens` minus that count is the offset. The offset
absorbs everything that does not vary between records: role and message framing, the tool schema and
the function-call markup. Two counterfactual inputs add `safety_guardrails` or `exclusions_or_cautions`
to both tool payloads with the runner's own serializer (pydantic `model_dump_json`). Nothing is fitted
apart from reading off the offset.

`logged`, `reconstructed` and `offset` refer to the decision call (`<case>_final.json`), whose input carries
both tool payloads. The first call (`call1_*`) carries the system prompt, user message and tool schema but
no payload, so it cannot show whether a hidden field was sent; it is reported as a check of those parts.
"""
import copy
import csv
import gzip
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from r2.audit.original_inputs import FixedDraw, load_case_models
from r2.original_runs import load_original
from r2.paths import AUDIT_DIR, LEGACY_RESULTS, REPO
from r2.profiles import OPENAI_FC_MODELS

ENCODING = "o200k_base"
K_VALUES = (0, 1, 2, 5, 10)
# a record tests a counterfactual only if the field would add a different number of tokens than it adds to
# most records; a constant addition is absorbed by the offset. 2 tokens allows for JSON-boundary merges.
INFORMATIVE_MARGIN = 2
HIDDEN = {"cf_guardrails": "safety_guardrails", "cf_exclusions": "exclusions_or_cautions"}
COLUMNS = ["model", "case_id", "run_dir", "structure", "n_calls",
           "call1_logged", "call1_reconstructed", "call1_offset", "call1_function_calls",
           "logged", "reconstructed", "offset",
           "cf_guardrails_added", "cf_guardrails_offset", "cf_exclusions_added", "cf_exclusions_offset"]
INT_COLUMNS = [c for c in COLUMNS if c not in ("model", "case_id", "run_dir", "structure")]


def _src():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    os.environ.setdefault("OPENAI_API_KEY", "unused-for-replay")


def encoder():
    import tiktoken
    return tiktoken.get_encoding(ENCODING).encode_ordinary


# ------------------------------------------------------------------ replay of src/runner.py

class _Reply:
    def __init__(self, doc):
        self._doc = doc

    def model_dump(self):
        return self._doc


class _ReplayClient:
    """Stands in for OpenAI(): returns the logged replies in order and keeps a copy of each input."""

    def __init__(self, docs):
        self._docs = list(docs)
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        # the runner appends to one messages list after each call; copy it as sent
        self.calls.append({"input": copy.deepcopy(kwargs["input"]), "tools": "tools" in kwargs})
        if not self._docs:
            raise LookupError("replay asked for more calls than were logged")
        return _Reply(self._docs.pop(0))


def replay_calls(case_model, mapping, docs, logged_names):
    """Run the original ExperimentRunner._process_case on the logged replies and return each call's input.

    Raises RuntimeError unless the replay makes exactly the logged calls (same file names, same mapping,
    no runner error)."""
    _src()
    from src.prompts import get_system_prompt
    from src.runner import ExperimentRunner
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.model = "replay"
    runner.prompt_variant = "baseline"
    runner.require_both_tools = True
    runner.rng = FixedDraw(mapping)
    runner.client = _ReplayClient(docs)
    runner.system_prompt = get_system_prompt("baseline")
    with tempfile.TemporaryDirectory() as tmp:
        runner.raw_dir = Path(tmp)  # _process_case writes its raw files here, not to the legacy tree
        result = runner._process_case(case_model)
    got = {"A": result.mapping.A, "B": result.mapping.B}
    if result.errors or result.raw_responses != list(logged_names) or got != dict(mapping) \
            or len(runner.client.calls) != len(docs):
        raise RuntimeError(f"replay of {case_model.case_id} differs from the log: "
                           f"{result.raw_responses} vs {list(logged_names)}; errors {result.errors}")
    return runner.client.calls


# ------------------------------------------------------------------ token counts

def content_texts(items):
    """The free text of an input list: message contents and function outputs. Function-call items carry
    only a fixed name, '{}' arguments and a call id; they belong to the constant offset."""
    out = []
    for it in items:
        if "role" in it:
            out.append(it["content"])
        elif it.get("type") == "function_call_output":
            out.append(it["output"])
    return out


def count_tokens(items, encode):
    return sum(len(encode(t)) for t in content_texts(items))


def payload_json(payload):
    """The function output exactly as src/runner.py sent it: SanitizedTool.model_dump_json()."""
    _src()
    from src.schemas import SanitizedTool
    return SanitizedTool(**payload).model_dump_json()


@lru_cache(maxsize=None)
def _with_field(field):
    _src()
    from pydantic import create_model
    from src.schemas import SanitizedTool
    return create_model(f"SanitizedTool_{field}", __base__=SanitizedTool,
                        **{field: (Optional[list[str]], None)})


def payload_with_field(payload, field, value):
    """Counterfactual function output: the same payload and serializer, with `field` serialized after
    excerpt_text."""
    return _with_field(field)(**payload, **{field: value}).model_dump_json()


def _substitute(items, by_text):
    out = []
    for it in items:
        if it.get("type") == "function_call_output":
            if it["output"] not in by_text:
                raise KeyError("function output matches neither tool payload")
            it = {**it, "output": by_text[it["output"]]}
        out.append(it)
    return out


def n_function_calls(doc):
    return sum(1 for o in doc.get("output") or [] if o.get("type") == "function_call")


def structure(docs):
    """Tool calls per tool-calling response, e.g. 'AB' (both in call 1), 'A|B', '-|AB' (none, then both)."""
    rounds = []
    for doc in docs[:-1]:
        names = [o.get("name") for o in doc.get("output") or [] if o.get("type") == "function_call"]
        rounds.append("".join(n[-1].upper() for n in names) or "-")
    return "|".join(rounds)


def call_details(calls, docs, logged_names, encode):
    """Per API call: logged and reconstructed input tokens, and what the request and reply contained."""
    out = []
    for name, call, doc in zip(logged_names, calls, docs):
        rec = count_tokens(call["input"], encode)
        logged = doc["usage"]["input_tokens"]
        out.append({"file": name, "input_items": len(call["input"]), "tool_schema_sent": call["tools"],
                    "function_calls_emitted": n_function_calls(doc), "logged": logged, "reconstructed": rec,
                    "offset": logged - rec, "twice_logged_minus_reconstructed": 2 * logged - rec})
    return out


def record_row(gz_row, case_model, docs, logged_names, encode, calls=None):
    mapping = gz_row["mapping"]
    calls = calls or replay_calls(case_model, mapping, docs, logged_names)
    first, final = calls[0]["input"], calls[-1]["input"]
    pj = {x: payload_json(gz_row["payloads"][x]) for x in "AB"}
    assert first[0]["content"] == gz_row["system"], "system prompt differs from the audit file"
    assert first[1]["content"] == gz_row["user"], "user message differs from the audit file"
    outputs = [it["output"] for it in final if it.get("type") == "function_call_output"]
    assert sorted(outputs) == sorted(pj.values()), "tool payloads differ from the audit file"
    tools = {"T": case_model.tools.T, "S": case_model.tools.S}
    rec = count_tokens(final, encode)
    logged = docs[-1]["usage"]["input_tokens"]
    c1_rec = count_tokens(first, encode)
    c1_log = docs[0]["usage"]["input_tokens"]
    row = {"model": gz_row["model"], "case_id": gz_row["case_id"], "run_dir": gz_row["run_dir"],
           "structure": structure(docs), "n_calls": len(docs),
           "call1_logged": c1_log, "call1_reconstructed": c1_rec, "call1_offset": c1_log - c1_rec,
           "call1_function_calls": n_function_calls(docs[0]),
           "logged": logged, "reconstructed": rec, "offset": logged - rec}
    for key, field in HIDDEN.items():
        cf = {pj[x]: payload_with_field(gz_row["payloads"][x], field, getattr(tools[mapping[x]], field))
              for x in "AB"}
        cf_rec = count_tokens(_substitute(final, cf), encode)
        row[f"{key}_added"] = cf_rec - rec
        row[f"{key}_offset"] = logged - cf_rec
    return row


# ------------------------------------------------------------------ statistics

def _num(x):
    x = float(x)
    return int(x) if x.is_integer() else x


def offset_stats(values, ks=K_VALUES):
    v = np.asarray(values, dtype=float)
    med = float(np.median(v))
    q1, q3 = (float(q) for q in np.percentile(v, [25, 75]))
    return {"n": int(v.size), "median": _num(med), "q1": _num(q1), "q3": _num(q3), "iqr": _num(q3 - q1),
            "min": _num(v.min()), "max": _num(v.max()),
            "within_k_of_median": {str(k): float(np.mean(np.abs(v - med) <= k)) for k in ks}}


def slope(y, x):
    """OLS slope of y on x; None when x does not vary."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.all(x == x[0]):
        return None
    return _num(round(float(np.cov(x, y, bias=True)[0, 1] / np.var(x)), 12))


def _decision(rows, ks):
    out = {"offset": offset_stats([r["offset"] for r in rows], ks)}
    for key in HIDDEN:
        out[f"{key}_offset"] = offset_stats([r[f"{key}_offset"] for r in rows], ks)
    out["median_shift"] = {key: _num(out[f"{key}_offset"]["median"] - out["offset"]["median"]) for key in HIDDEN}
    out["added_tokens"] = {}
    for key in HIDDEN:
        a = [r[f"{key}_added"] for r in rows]
        out["added_tokens"][key] = {"min": _num(min(a)), "median": _num(np.median(a)), "max": _num(max(a))}
    out["slope_offset_on_added"] = {key: slope([r["offset"] for r in rows], [r[f"{key}_added"] for r in rows])
                                    for key in HIDDEN}
    out["informative_subset"] = {}
    for key in HIDDEN:
        counts = Counter(r[f"{key}_added"] for r in rows)
        mode = min(v for v, n in counts.items() if n == max(counts.values()))
        sub = [r for r in rows if abs(r[f"{key}_added"] - mode) > INFORMATIVE_MARGIN]
        rest = [r for r in rows if abs(r[f"{key}_added"] - mode) <= INFORMATIVE_MARGIN]
        info = {"modal_added": mode, "n": len(sub), "predicted_difference": None,
                "offset_median_subset": None, "offset_median_rest": None, "observed_difference": None}
        if sub and rest:
            pred = [r[f"{key}_added"] - mode for r in sub]
            ms, mr = (_num(np.median([r["offset"] for r in v])) for v in (sub, rest))
            info.update({"predicted_difference": {"min": _num(min(pred)), "median": _num(np.median(pred)),
                                                  "max": _num(max(pred))},
                         "offset_median_subset": ms, "offset_median_rest": mr, "observed_difference": _num(ms - mr)})
        out["informative_subset"][key] = info
    return out


def summarize(rows, ks=K_VALUES):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    models = {}
    for model, rs in sorted(by_model.items()):
        strata = defaultdict(list)
        for r in rs:
            strata[r["structure"]].append(r)
        dc = _decision(rs, ks)
        dc["by_structure"] = {st: _decision(v, ks) for st, v in sorted(strata.items())}
        by_fc = defaultdict(list)
        for r in rs:
            by_fc[str(r["call1_function_calls"])].append(r)
        c1 = {"offset": offset_stats([r["call1_offset"] for r in rs], ks), "by_function_calls": {}}
        for nfc, v in sorted(by_fc.items()):
            twice = Counter(2 * r["call1_logged"] - r["call1_reconstructed"] for r in v)
            c1["by_function_calls"][nfc] = {
                "offset": offset_stats([r["call1_offset"] for r in v], ks),
                "twice_logged_minus_reconstructed": {str(k): n for k, n in sorted(twice.items())}}
        models[model] = {"n_records": len(rs), "structures": dict(sorted(Counter(r["structure"] for r in rs).items())),
                         "decision_call": dc, "call1": c1}
    return {"models": models}


def typed_row(r):
    return {k: (int(r[k]) if k in INT_COLUMNS else r[k]) for k in COLUMNS}


# ------------------------------------------------------------------ the 2,000 records

def _gz_rows():
    out = {}
    with gzip.open(AUDIT_DIR / "original_model_inputs.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["profile"] == "openai_fc":
                out[(r["model"], r["case_id"])] = r
    return out


def build(write=True, encode=None):
    import tiktoken
    encode = encode or encoder()
    gz = _gz_rows()
    cases = load_case_models()
    rows, multi = [], []
    for rec in load_original():
        if rec["model_name"] not in OPENAI_FC_MODELS:
            continue
        g = gz[(rec["model_name"], rec["case_id"])]
        assert g["run_dir"] == rec["run_dir"] and g["mapping"] == rec["mapping"]
        names = rec["raw_responses"]
        docs = [json.loads((LEGACY_RESULTS / rec["run_dir"] / "raw" / n).read_text()) for n in names]
        calls = replay_calls(cases[rec["case_id"]], g["mapping"], docs, names)
        rows.append(record_row(g, cases[rec["case_id"]], docs, names, encode, calls=calls))
        if len(docs) > 2:
            multi.append({"model": g["model"], "case_id": g["case_id"], "structure": structure(docs),
                          "calls": call_details(calls, docs, names, encode)})
    rows.sort(key=lambda r: (r["model"], r["case_id"]))
    multi.sort(key=lambda r: (r["model"], r["case_id"]))
    summary = {
        "encoding": {"name": ENCODING, "tiktoken_version": tiktoken.__version__,
                     "encoding_for_model": {m: tiktoken.encoding_for_model(m).name for m in sorted(OPENAI_FC_MODELS)}},
        "k_values": list(K_VALUES),
        "definitions": {
            "decision_call": "the final API call (`<case>_final.json`); its input carries both tool payloads",
            "call1": "the first API call (`<case>_response1.json`): system prompt, user message and tool schema; "
                     "no tool payload",
            "reconstructed": "sum of tiktoken counts of each message content and function output in the call "
                             "input, replayed with src/runner.py",
            "offset": "logged usage.input_tokens minus reconstructed",
            "cf_*_added": "tokens added when the hidden field is serialized into both payloads",
            "cf_*_offset": "logged minus the counterfactual reconstruction (= offset - added)",
            "median_shift": "median(cf offset) - median(offset)",
            "slope_offset_on_added": "OLS slope of the observed offset on the tokens the field would add: "
                                     "0 if the field was absent, 1 if it was present",
            "structure": "tool calls per tool-calling response: AB = both in call 1; A|B = one per call; "
                         "-|AB = none, then both",
            "twice_logged_minus_reconstructed": "call-1 diagnostic, 2 x logged - reconstructed, by number of "
                                                "function calls the model emitted in call 1",
            "informative_subset": f"records whose added tokens differ from the modal added count by more than "
                                  f"{INFORMATIVE_MARGIN}; only these can tell the counterfactual from the "
                                  f"reconstruction, since a constant addition is absorbed by the offset. If the "
                                  f"field had been sent, their observed offset would exceed the rest's by "
                                  f"predicted_difference (added - modal added); observed_difference is the "
                                  f"difference of median observed offsets",
            "multi_call_records": "every API call of the records with more than one tool-calling response"},
        **summarize(rows),
        "multi_call_records": multi,
    }
    if write:
        with open(AUDIT_DIR / "openai_input_tokens.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        (AUDIT_DIR / "openai_input_tokens_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return rows, summary


if __name__ == "__main__":
    _, s = build()
    for m, v in s["models"].items():
        dc = v["decision_call"]
        print(m, v["structures"], "offset", {k: dc["offset"][k] for k in ("median", "iqr", "min", "max")},
              "shift", dc["median_shift"], "slope", dc["slope_offset_on_added"])
