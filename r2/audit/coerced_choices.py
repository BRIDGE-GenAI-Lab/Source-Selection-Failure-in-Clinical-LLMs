"""Which original records hold a choice that the runner, not the model, set to "A".

The four inline runner families replaced a missing or invalid model choice with "A" and scored it as a
real choice (RUNNER_CONDITIONS gives file:line). The openai_fc runner (src/runner.py) has no such
fallback: unparseable output raises, and a missing choice becomes "unclear".

Each of the 10,500 canonical records is classified from stored evidence only, by replaying the
runner's own parse step on the raw response file it wrote (`results/<run>/raw/<case>_response.json`)
and checking the result against the stored decision:

  coerced_parse_failure   no JSON object was found, or json.loads failed; the runner wrote its fallback
                          decision (selected_tool "A", confidence 0.5, a fixed trust_rationale);
  coerced_invalid_choice  the reply parsed, but selected_tool was missing or, after upper-casing, not
                          "A" or "B" (e.g. "Neither", null, ""), and the runner wrote "A";
  not_coerced             every other record.

Where no raw file exists (three Qwen3-VL-8B re-run records), only the stored rationale is used, so an
invalid choice cannot be detected there; the `evidence` column says which evidence was used. A replay
that disagrees with the stored record raises EvidenceConflict instead of guessing.

    python -m r2.audit.coerced_choices
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

from r2.original_runs import load_original, sham_position
from r2.paths import AUDIT_DIR, LEGACY_RESULTS, REPO
from r2.profiles import profile_for

COERCED_PARSE = "coerced_parse_failure"
COERCED_INVALID = "coerced_invalid_choice"
NOT_COERCED = "not_coerced"

_GEMINI_DEEPSEEK = {"json_decode_error": "Failed to parse response", "no_json": "No JSON found in response"}
FALLBACK_RATIONALE = {
    "together_inline6": {"json_decode_error": "Parse failed", "no_json": "No JSON found"},
    "gemini_inline7": dict(_GEMINI_DEEPSEEK),
    "deepseek_chat_inline7": dict(_GEMINI_DEEPSEEK),
    "deepseek_reasoner_inline7": dict(_GEMINI_DEEPSEEK),
}

# Line numbers hold for the repo copies under scripts/ and for the legacy originals (the copies differ
# only in the API-key line 25 and in output paths after line 285).
RUNNER_CONDITIONS = {
    "together_inline6": {
        "file": "scripts/run_together_experiment.py",
        "json_span": "212-216: text from the first '{' to the last '}'",
        "json_decode_error": "217-220: json.JSONDecodeError -> selected_tool 'A', confidence 0.5, trust_rationale 'Parse failed'",
        "no_json": "215, 221-222: no '{' or no '}' after it -> selected_tool 'A', confidence 0.5, trust_rationale 'No JSON found'",
        "invalid_or_missing_choice": "224-227: str(decision.get('selected_tool', 'A')).upper() not in ('A', 'B') -> 'A'; a missing key defaults to 'A'",
    },
    "gemini_inline7": {
        "file": "scripts/run_gemini_experiment.py",
        "json_span": "234-238: text from the first '{' to the last '}'",
        "json_decode_error": "239-252: json.JSONDecodeError -> selected_tool 'A', confidence 0.5, trust_rationale 'Failed to parse response'",
        "no_json": "237, 253-264: no '{' or no '}' after it -> selected_tool 'A', confidence 0.5, trust_rationale 'No JSON found in response'",
        "invalid_or_missing_choice": "266-270: decision.get('selected_tool', 'A').upper() not in ('A', 'B') -> 'A'; a missing key defaults to 'A'; a non-string value (e.g. null) raises AttributeError, caught at 288-299 as an error record",
    },
    "deepseek_chat_inline7": {
        "file": "scripts/run_deepseek_experiment.py",
        "json_span": "245-249: text from the first '{' to the last '}'",
        "json_decode_error": "250-263: json.JSONDecodeError -> selected_tool 'A', confidence 0.5, trust_rationale 'Failed to parse response'",
        "no_json": "248, 264-275: no '{' or no '}' after it -> selected_tool 'A', confidence 0.5, trust_rationale 'No JSON found in response'",
        "invalid_or_missing_choice": "277-281: str(decision.get('selected_tool', 'A')).upper() not in ('A', 'B') -> 'A'; a missing key defaults to 'A'",
    },
    "deepseek_reasoner_inline7": {
        "file": "scripts/run_deepseek_reasoner.py",
        "json_span": "241-245: text from the first '{' to the last '}' of message.content (reasoning_content is not parsed)",
        "json_decode_error": "246-259: json.JSONDecodeError -> selected_tool 'A', confidence 0.5, trust_rationale 'Failed to parse response'",
        "no_json": "244, 260-271: no '{' or no '}' after it -> selected_tool 'A', confidence 0.5, trust_rationale 'No JSON found in response'",
        "invalid_or_missing_choice": "273-277: str(decision.get('selected_tool', 'A')).upper() not in ('A', 'B') -> 'A'; a missing key defaults to 'A'",
    },
    "openai_fc": {
        "file": "src/runner.py",
        "no_A_fallback": ("unparseable output raises ValueError (legacy src/runner.py:433, repo copy :435), "
                          "giving an error record, not a decision; a missing selected_tool becomes 'unclear' "
                          "(legacy :446, repo copy :448); ModelDecision accepts only 'A', 'B', 'unclear' "
                          "(src/schemas.py:158)"),
    },
}


class EvidenceConflict(ValueError):
    """The replayed runner decision and the stored decision disagree."""


def replay_inline(profile, text):
    """The parse step of the profile's runner applied to `text`, as written in RUNNER_CONDITIONS."""
    fallback = FALLBACK_RATIONALE[profile]
    start, end = text.find("{"), text.rfind("}") + 1
    if not (start >= 0 and end > start):
        return {"outcome": "no_json", "selected_tool": "A", "trust_rationale": fallback["no_json"],
                "raw_selected_tool": ""}
    try:
        decision = json.loads(text[start:end])
    except json.JSONDecodeError:
        return {"outcome": "json_decode_error", "selected_tool": "A",
                "trust_rationale": fallback["json_decode_error"], "raw_selected_tool": ""}
    missing = "selected_tool" not in decision
    value = decision.get("selected_tool", "A")
    raw = "" if missing else json.dumps(value)
    if profile == "gemini_inline7" and not isinstance(value, str):
        # run_gemini_experiment.py:267 calls .upper() without str(): the runner raised, wrote no decision
        return {"outcome": "runner_raises", "selected_tool": None, "trust_rationale": None,
                "raw_selected_tool": raw}
    selected = str(value).upper()
    if selected not in ("A", "B"):
        outcome, selected = "invalid_choice", "A"
    else:
        outcome = "missing_choice" if missing else "valid_choice"
    return {"outcome": outcome, "selected_tool": selected,
            "trust_rationale": decision.get("trust_rationale"), "raw_selected_tool": raw}


_CLASS = {"no_json": COERCED_PARSE, "json_decode_error": COERCED_PARSE,
          "invalid_choice": COERCED_INVALID, "missing_choice": COERCED_INVALID,
          "valid_choice": NOT_COERCED, "runner_raises": NOT_COERCED}


def classify(profile, stored, raw_text, openai_fc_replay=None):
    """Classify one record. `stored` is the stored model_decision; `raw_text` the runner's raw response
    text, or None when no raw response file exists."""
    stored = stored or {}
    if profile == "openai_fc":
        return {"class": NOT_COERCED, "reason": "runner_has_no_A_fallback",
                "evidence": "runner_code" if openai_fc_replay is None else "raw_replay",
                "raw_selected_tool": "", "replay_matches_stored": openai_fc_replay}
    if raw_text is None:
        kind = {v: k for k, v in FALLBACK_RATIONALE[profile].items()}.get(stored.get("trust_rationale"))
        if kind and stored.get("selected_tool") == "A":
            return {"class": COERCED_PARSE, "reason": kind, "evidence": "rationale_only",
                    "raw_selected_tool": "", "replay_matches_stored": None}
        return {"class": NOT_COERCED, "reason": "rationale_not_fallback", "evidence": "rationale_only",
                "raw_selected_tool": "", "replay_matches_stored": None}
    rep = replay_inline(profile, raw_text)
    matches = (rep["selected_tool"] == stored.get("selected_tool")
               and rep["trust_rationale"] == stored.get("trust_rationale"))
    if rep["outcome"] == "runner_raises":
        if stored.get("selected_tool") == "A":
            raise EvidenceConflict(f"runner would raise, stored record holds 'A': {stored}")
        reason = "runner_would_raise"
    elif not matches:
        raise EvidenceConflict(f"replay {rep} differs from stored {stored}")
    else:
        reason = rep["outcome"]
    return {"class": _CLASS[rep["outcome"]], "reason": reason, "evidence": "raw_replay",
            "raw_selected_tool": rep["raw_selected_tool"], "replay_matches_stored": matches}


def _raw_text(run_dir, case_id):
    path = LEGACY_RESULTS / run_dir / "raw" / f"{case_id}_response.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("response_text")


_PARSER = []


def _openai_fc_replay(record):
    """Run the original src/runner.py _parse_decision on the stored *_final.json response; True when it
    yields the stored selected_tool, None when no final response file exists."""
    finals = [f for f in record.get("raw_responses") or [] if f.endswith("_final.json")]
    if not finals:
        return None
    path = LEGACY_RESULTS / record["run_dir"] / "raw" / finals[-1]
    if not path.exists():
        return None
    if not _PARSER:
        os.environ.setdefault("OPENAI_API_KEY", "unused-for-replay")
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from src.runner import ExperimentRunner
        _PARSER.append(ExperimentRunner._parse_decision)
    decision = _PARSER[0](None, json.loads(path.read_text()), record["case_id"], [])
    return decision.selected_tool == record["model_decision"]["selected_tool"]


def classify_all():
    visible = pd.read_csv(AUDIT_DIR / "pair_audit.csv").set_index("case_id")["visible_status"].to_dict()
    rows = []
    for r in load_original():
        profile = profile_for(r["model_name"])
        stored = r.get("model_decision") or {}
        if profile == "openai_fc":
            out = classify(profile, stored, None, openai_fc_replay=_openai_fc_replay(r))
        else:
            out = classify(profile, stored, _raw_text(r["run_dir"], r["case_id"]))
        st = stored.get("selected_tool")
        rows.append({"model": r["model_name"], "profile": profile, "case_id": r["case_id"],
                     "sham_position": sham_position(r), "visible_status": visible[r["case_id"]],
                     "class": out["class"], "reason": out["reason"],
                     "raw_selected_tool": out["raw_selected_tool"],
                     "stored_selected_tool": "null" if st is None else st,
                     "evidence": out["evidence"], "replay_matches_stored": out["replay_matches_stored"]})
    return rows


def _counts(rows, key=None):
    if key is None:
        return dict(sorted(Counter(r["class"] for r in rows).items()))
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {k: _counts(v) for k, v in sorted(groups.items())}


def summarize(rows):
    by_profile = defaultdict(list)
    for r in rows:
        by_profile[r["profile"]].append(r)
    invalid = defaultdict(Counter)
    for r in rows:
        if r["class"] == COERCED_INVALID:
            invalid[r["model"]][r["raw_selected_tool"] or "(key missing)"] += 1
    coerced = [r for r in rows if r["class"] != NOT_COERCED]
    return {
        "n_records": len(rows),
        "class_counts": _counts(rows),
        "coerced_total": len(coerced),
        "by_profile": _counts(rows, "profile"),
        "by_profile_model": {p: _counts(v, "model") for p, v in sorted(by_profile.items())},
        "by_visible_status": _counts(rows, "visible_status"),
        "by_profile_visible_status": {p: _counts(v, "visible_status") for p, v in sorted(by_profile.items())},
        "reason_counts": {p: dict(sorted(Counter(r["reason"] for r in v).items()))
                          for p, v in sorted(by_profile.items())},
        "invalid_choice_values": {m: dict(c.most_common()) for m, c in sorted(invalid.items())},
        "evidence_counts": dict(sorted(Counter(r["evidence"] for r in rows).items())),
        "rationale_only_records": sorted(f"{r['model']} {r['case_id']}" for r in rows
                                         if r["evidence"] == "rationale_only"),
        "replay_mismatch_records": sorted(f"{r['model']} {r['case_id']} ({r['reason']})" for r in rows
                                          if r["replay_matches_stored"] is False),
        "runner_conditions": RUNNER_CONDITIONS,
    }


COLUMNS = ["model", "profile", "case_id", "sham_position", "visible_status", "class", "reason",
           "raw_selected_tool", "stored_selected_tool", "evidence", "replay_matches_stored"]


def run(write=True):
    rows = classify_all()
    summary = summarize(rows)
    if write:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        rows_sorted = sorted(rows, key=lambda r: (r["model"], r["case_id"]))
        with open(AUDIT_DIR / "coerced_choices.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows_sorted:
                w.writerow({k: ("" if r[k] is None else r[k]) for k in COLUMNS})
        (AUDIT_DIR / "coerced_choices_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return rows, summary


if __name__ == "__main__":
    _, s = run()
    print(json.dumps({k: s[k] for k in ("class_counts", "by_profile", "by_visible_status")}, indent=2))
