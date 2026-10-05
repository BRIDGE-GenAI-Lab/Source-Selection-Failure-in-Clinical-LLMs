"""Physician review packets showing exactly what models receive (rendered by r2.stimuli.render)."""
import csv
import difflib
import hashlib
import html
import json
import random
from pathlib import Path

from r2.paths import DATA_V2, VERIFY_DIR
from r2.stimuli.render import render_tool, render_vignette
from r2.verify.score_review import QUESTIONS

# The packet hash binds each physician-reviewed row to the content the models receive.
PACKET_FIELDS = ("vignette_as_shown_to_model", "clinical_question", "authentic_as_shown_to_model",
                 "modified_as_shown_to_model")
PACKET_SEP = "\n\x1e\n"
REVIEWERS = (("AG", 1), ("MO", 2))
MANIFEST = "packets_manifest.json"


def pairs_path():
    return DATA_V2 / "pairs_v2.json"


def _load():
    return json.loads(pairs_path().read_text())


def _cases():
    return _load()["cases"]


def packet_sha256(row):
    parts = ["" if row[k] is None else str(row[k]) for k in PACKET_FIELDS]
    return hashlib.sha256(PACKET_SEP.join(parts).encode("utf-8")).hexdigest()


def packet_rows(cases=None):
    rows = []
    for c in (_cases() if cases is None else cases):
        t, s = c["tools"]["T"], c["tools"]["S"]
        rec = c["edit_record"]
        row = {
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
        }
        row["packet_sha256"] = packet_sha256(row)
        rows.append(row)
    return rows


def _answered_forms(answer_cols):
    answered = []
    for reviewer, _ in REVIEWERS:
        path = VERIFY_DIR / f"review_{reviewer}.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            if any((row.get(c) or "").strip() for row in csv.DictReader(fh) for c in answer_cols):
                answered.append(path.name)
    return answered


def write_forms():
    raw = pairs_path().read_bytes()
    d = json.loads(raw)
    if len(d["cases"]) != 500 or d["unresolved"]:
        raise RuntimeError(
            f"refusing to write review forms: pairs_v2.json has {len(d['cases'])} cases and "
            f"{len(d['unresolved'])} unresolved pairs; need exactly 500 cases and none unresolved "
            "(physician-authored manual edits must be applied and the build rerun strictly)")
    answer_cols = [q for q, _ in QUESTIONS] + ["notes"]
    answered = _answered_forms(answer_cols)
    if answered:
        raise RuntimeError(f"refusing to overwrite review forms that already contain answers: {answered}")
    VERIFY_DIR.mkdir(exist_ok=True)
    rows = packet_rows(d["cases"])
    (VERIFY_DIR / MANIFEST).write_text(json.dumps(
        {"pairs_v2_sha256": hashlib.sha256(raw).hexdigest(),
         "packets": {r["case_id"]: r["packet_sha256"] for r in rows}}, indent=1) + "\n")
    for reviewer, seed in REVIEWERS:
        order = rows[:]
        random.Random(seed).shuffle(order)
        with open(VERIFY_DIR / f"review_{reviewer}.csv", "w", newline="", encoding="utf-8") as fh:
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
