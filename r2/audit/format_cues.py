"""Format cues in the original fabricated_citation inputs (cue fixes A1, A7).

The original generator appended  "\\n\\nEvidence: 'NEJM 2025;392:<p1>–<p2>' (DOI:...)"  to the modified excerpt.
The blank line occurs in no authentic excerpt, and the page span (p2 - p1) is implausible. This counts, over the
reconstructed original inputs (audit/original_model_inputs.jsonl.gz), the fabricated_citation rows in which

  * the modified (S) excerpt carries a blank line, read from the payload dict the runner gave the model and
    from the serialized user message (a JSON-escaped "\\n\\n" counts as the cue too), against the authentic (T)
    excerpt of the same row, which does not;
  * the cited page span falls in each range, and the spans themselves.

    python -m r2.audit.format_cues
"""
import gzip
import json
import re
import statistics
from collections import Counter

from r2.paths import AUDIT_DIR

SOURCE = "audit/original_model_inputs.jsonl.gz"
OUT = AUDIT_DIR / "format_cues.json"
SHAM = "fabricated_citation"
PLAUSIBLE_SPAN = (6, 14)  # the span range of the corrected set (r2.stimuli.transforms.PAGE_SPAN)
CITATION = re.compile(r"Evidence: '?NEJM ((?:19|20)\d{2});(\d+):(\d+)[–-](\d+)'?")
BLANK = re.compile(r"\n\n|\\n\\n")  # the newline pair itself, or as a JSON escape in the serialized message


def _rows():
    with gzip.open(AUDIT_DIR / "original_model_inputs.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def _excerpt(payload):
    return str(payload.get("excerpt_text") or "")


def _serialized_has_cue(user, excerpt):
    """The excerpt as the user message carries it (JSON-escaped or not) holds the blank line."""
    if "\n\n" not in excerpt:
        return False
    for form in (json.dumps(excerpt, ensure_ascii=False)[1:-1], json.dumps(excerpt)[1:-1], excerpt):
        if form in user:
            return bool(BLANK.search(form))
    return False


def compute():
    n_rows, fc_rows = 0, 0
    cue = {"rows": 0, "cases": set(), "by_profile": Counter(), "by_model": Counter()}
    serialized_rows, authentic_rows, no_cite = 0, 0, 0
    span_rows, span_by_profile = Counter(), {}
    spans_by_case, in_range_rows = {}, Counter()
    models, by_profile_models = set(), {}
    for r in _rows():
        n_rows += 1
        if r["sham_type"] != SHAM:
            continue
        fc_rows += 1
        models.add(r["model"])
        by_profile_models.setdefault(r["profile"], set()).add(r["model"])
        arm = {r["mapping"][lab]: r["payloads"][lab] for lab in ("A", "B")}
        s_text, t_text = _excerpt(arm["S"]), _excerpt(arm["T"])
        if BLANK.search(s_text):
            cue["rows"] += 1
            cue["cases"].add(r["case_id"])
            cue["by_profile"][r["profile"]] += 1
            cue["by_model"][r["model"]] += 1
        if _serialized_has_cue(r.get("user") or "", s_text):
            serialized_rows += 1
        if BLANK.search(t_text):
            authentic_rows += 1
        m = CITATION.search(s_text)
        if not m:
            no_cite += 1
            continue
        span = int(m.group(4)) - int(m.group(3))
        span_rows[span] += 1
        span_by_profile.setdefault(r["profile"], Counter())[span] += 1
        spans_by_case[r["case_id"]] = span
        in_range_rows["within" if PLAUSIBLE_SPAN[0] <= span <= PLAUSIBLE_SPAN[1] else "outside"] += 1
    spans = sorted(spans_by_case.values())
    out = {
        "source": SOURCE,
        "generated_by": "python -m r2.audit.format_cues",
        "sham_type": SHAM,
        "n_rows": n_rows,
        "n_fabricated_citation_rows": fc_rows,
        "n_models": len(models),
        "models_per_profile": {p: len(m) for p, m in sorted(by_profile_models.items())},
        "blank_line_cue": {
            "definition": "the modified (S) excerpt of the row, as the runner gave it to the model, contains a "
                          "blank line (two consecutive newlines); a JSON-escaped \\n\\n in the serialized "
                          "message counts too",
            "modified_payload_rows": cue["rows"],
            "modified_serialized_message_rows": serialized_rows,
            "serialized_message_note": "openai_fc rows deliver the payload as a function output, not inside the "
                                       "user message, so only the inline profiles count here",
            "authentic_payload_rows": authentic_rows,
            "n_cases": len(cue["cases"]),
            "n_models": len(cue["by_model"]),
            "by_profile": dict(sorted(cue["by_profile"].items())),
            "by_model": dict(sorted(cue["by_model"].items())),
            "case_ids": sorted(cue["cases"]),
        },
        "page_span": {
            "definition": "p2 - p1 of the citation 'NEJM <year>;<volume>:<p1>-<p2>' in the modified excerpt",
            "rows_with_a_citation": sum(span_rows.values()),
            "rows_without_a_parsable_citation": no_cite,
            "n_cases": len(spans_by_case),
            "min": spans[0] if spans else None,
            "median": statistics.median(spans) if spans else None,
            "max": spans[-1] if spans else None,
            "cases_with_span_above_14": sum(s > PLAUSIBLE_SPAN[1] for s in spans),
            "cases_with_span_above_100": sum(s > 100 for s in spans),
            "plausible_range_of_the_corrected_set": list(PLAUSIBLE_SPAN),
            "rows_within_plausible_range": in_range_rows["within"],
            "rows_outside_plausible_range": in_range_rows["outside"],
            "span_by_case": dict(sorted(spans_by_case.items())),
            "rows_by_span": {str(k): v for k, v in sorted(span_rows.items())},
            "rows_by_profile_and_span_range": {
                p: {"1-100": sum(v for k, v in c.items() if k <= 100),
                    "101-500": sum(v for k, v in c.items() if 100 < k <= 500),
                    "501-1000": sum(v for k, v in c.items() if 500 < k <= 1000),
                    "above 1000": sum(v for k, v in c.items() if k > 1000)}
                for p, c in sorted(span_by_profile.items())},
        },
    }
    b, ps = out["blank_line_cue"], out["page_span"]
    out["finding"] = (
        f"The blank line sits in the modified payload of {b['modified_payload_rows']} of {fc_rows} "
        f"fabricated_citation inputs ({b['n_cases']} cases x {b['n_models']} models; "
        f"{b['modified_serialized_message_rows']} in the serialized message) and in {b['authentic_payload_rows']} "
        f"authentic payloads. The cited page span runs from {ps['min']} to {ps['max']} pages (median "
        f"{ps['median']}; {ps['cases_with_span_above_14']} of {ps['n_cases']} cases above 14 pages), where "
        f"the corrected set uses {PLAUSIBLE_SPAN[0]} to {PLAUSIBLE_SPAN[1]}. Both are format cues on top of the "
        "'fake' DOI (self_label_tokens.json); the original fabricated_citation results are cue-confounded.")
    return out


def main():
    out = compute()
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out["finding"])


if __name__ == "__main__":
    main()
