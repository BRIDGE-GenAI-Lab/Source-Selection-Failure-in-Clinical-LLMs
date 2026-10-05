"""Physician review packets showing exactly what models receive (rendered by r2.stimuli.render)."""
import csv
import difflib
import html
import json
import random
from pathlib import Path

from r2.paths import DATA_V2, VERIFY_DIR
from r2.stimuli.render import render_tool, render_vignette
from r2.verify.score_review import QUESTIONS


def _load():
    return json.loads((DATA_V2 / "pairs_v2.json").read_text())


def _cases():
    return _load()["cases"]


def packet_rows():
    rows = []
    for c in _cases():
        t, s = c["tools"]["T"], c["tools"]["S"]
        rec = c["edit_record"]
        rows.append({
            "case_id": c["case_id"], "sham_type": c["eval"]["sham_trap_type"],
            "source_guideline": f'{t["metadata"].get("organization")} - {t["metadata"].get("guideline_title")} '
                                f'({t["metadata"].get("publication_year")}); file '
                                f'{Path(t["metadata"].get("source_pdf_path") or "").name}',
            "vignette_as_shown_to_model": render_vignette(c),
            "clinical_question": c.get("clinical_question"),
            "authentic_as_shown_to_model": json.dumps(render_tool(t, "authentic"), indent=1, ensure_ascii=False),
            "modified_as_shown_to_model": json.dumps(render_tool(s, "modified"), indent=1, ensure_ascii=False),
            "what_changed": json.dumps(rec["changes"], ensure_ascii=False),
            "edit_source": rec["edit_source"],
        })
    return rows


def write_forms():
    d = _load()
    if len(d["cases"]) != 500 or d["unresolved"]:
        raise RuntimeError(
            f"refusing to write review forms: pairs_v2.json has {len(d['cases'])} cases and "
            f"{len(d['unresolved'])} unresolved pairs; need exactly 500 cases and none unresolved "
            "(physician-authored manual edits must be applied and the build rerun strictly)")
    VERIFY_DIR.mkdir(exist_ok=True)
    rows = packet_rows()
    answer_cols = [q for q, _ in QUESTIONS] + ["notes"]
    for reviewer, seed in (("AG", 1), ("MO", 2)):
        order = rows[:]
        random.Random(seed).shuffle(order)
        with open(VERIFY_DIR / f"review_{reviewer}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) + answer_cols)
            w.writeheader()
            for r in order:
                w.writerow({**r, **{c: "" for c in answer_cols}})
    differ = difflib.HtmlDiff(wrapcolumn=80)
    parts = ["<html><meta charset='utf-8'><body style='font-family:Arial'>"]
    for r in rows:
        parts.append(f"<h2 id='{r['case_id']}'>{r['case_id']} ({r['sham_type']})</h2>")
        parts.append(f"<pre>{html.escape(r['vignette_as_shown_to_model'])}</pre>")
        parts.append(f"<p><b>Question:</b> {html.escape(str(r['clinical_question']))}</p>")
        parts.append(differ.make_table(r["authentic_as_shown_to_model"].splitlines(),
                                       r["modified_as_shown_to_model"].splitlines(),
                                       "Authentic", "Modified"))
    parts.append("</body></html>")
    (VERIFY_DIR / "packets.html").write_text("\n".join(parts))
    guide = ["# Reviewer instructions", "",
             "Review every row independently; do not discuss until both forms are complete.",
             "The text in the *_as_shown_to_model columns is exactly what the evaluated models receive.",
             "Answer each item with Y or N (Q3 also allows P = partially). Use `notes` for any N.", ""]
    guide += [f"- **{q}**: {text}" for q, text in QUESTIONS]
    guide += ["",
              "Edits produced by rule (edit_source = rule_typosquat or rule_inversion_extended) were generated automatically and may read awkwardly (e.g. lowercase or ungrammatical insertions such as 'the not first-line'). If an edit is clinically meaningful but contains a grammar or capitalization artifact that a model could use as a surface cue, answer Q6 = N and write a corrected edit in notes.",
              "",
              "Some authentic excerpts were selected by clinical domain, not by question, and may not address the clinical question (e.g. CASE_133). Answer Q4 accordingly.",
              "",
              "Cases listed in verification/leakage_flags.json contain allow-listed substrings (e.g. 'gi_bleed' in applicability text); check them under Q8."]
    (VERIFY_DIR / "README_REVIEWERS.md").write_text("\n".join(guide) + "\n")


if __name__ == "__main__":
    write_forms()
