"""Physician review packets showing exactly what models receive (rendered by r2.stimuli.render)."""
import csv
import difflib
import hashlib
import html
import json
import random
from pathlib import Path

from r2.io import load_original_cases
from r2.paths import DATA_V2, VERIFY_DIR
from r2.stimuli.render import render_tool, render_vignette
from r2.stimuli.transforms import TransformError, apply_manual_edit
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


def _answer_cols():
    return [q for q, _ in QUESTIONS] + ["notes"]


def _answered_forms(answer_cols, directory=None):
    directory = VERIFY_DIR if directory is None else directory
    answered = []
    for reviewer, _ in REVIEWERS:
        path = directory / f"review_{reviewer}.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            if any((row.get(c) or "").strip() for row in csv.DictReader(fh) for c in answer_cols):
                answered.append(path.name)
    return answered


def readme_text():
    guide = ["# Reviewer instructions", "",
             "Review every row independently; do not discuss until both forms are complete.",
             "The text in the *_as_shown_to_model columns is exactly what the evaluated models receive.",
             "Answer each item with Y or N (Q3 also allows P = partially). Use `notes` for any N.", ""]
    guide += [f"- **{q}**: {text}" for q, text in QUESTIONS]
    guide += ["",
              "Edits produced by rule (edit_source = rule_typosquat or rule_inversion_extended) were generated automatically and may read awkwardly (e.g. an ungrammatical or clinically odd insertion). If an edit is clinically meaningful but contains a grammar or capitalization artifact that a model could use as a surface cue, answer Q6 = N and write a corrected edit in notes.",
              "",
              "Some authentic excerpts were selected by clinical domain, not by question, and may not address the clinical question (e.g. CASE_133). Answer Q4 accordingly.",
              "",
              "CASE_335 has no clinical question (rendered 'Not specified'); judge Q2 and Q4 accordingly.",
              "",
              "18 of the 50 allergy_ignorance cases have no recorded allergy (rendered 'No known drug allergies'); check Q7 (does the modification reduce trust for this patient?) explicitly for these.",
              "",
              "Cases listed in verification/leakage_flags.json contain allow-listed substrings (e.g. 'gi_bleed' in applicability text); check them under Q8."]
    return "\n".join(guide) + "\n"


def _write_forms_to(directory, raw, cases, extra_manifest=None):
    answer_cols = _answer_cols()
    answered = _answered_forms(answer_cols, directory)
    if answered:
        raise RuntimeError(f"refusing to overwrite review forms that already contain answers: {answered}")
    directory.mkdir(parents=True, exist_ok=True)
    rows = packet_rows(cases)
    manifest = {"pairs_v2_sha256": hashlib.sha256(raw).hexdigest(),
                "packets": {r["case_id"]: r["packet_sha256"] for r in rows}}
    manifest.update(extra_manifest or {})
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    for reviewer, seed in REVIEWERS:
        order = rows[:]
        random.Random(seed).shuffle(order)
        with open(directory / f"review_{reviewer}.csv", "w", newline="", encoding="utf-8") as fh:
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
    (directory / "packets.html").write_text("\n".join(parts))
    (directory / "README_REVIEWERS.md").write_text(readme_text())


def write_forms(round_label="round1", allow_unresolved=False):
    """Default: the full 500-pair forms in verification/ (refuses unless complete). With
    allow_unresolved=True: forms for the resolved cases only, in verification/<round_label>/."""
    raw = pairs_path().read_bytes()
    d = json.loads(raw)
    if not allow_unresolved:
        if len(d["cases"]) != 500 or d["unresolved"]:
            raise RuntimeError(
                f"refusing to write review forms: pairs_v2.json has {len(d['cases'])} cases and "
                f"{len(d['unresolved'])} unresolved pairs; need exactly 500 cases and none unresolved "
                "(physician-authored manual edits must be applied and the build rerun strictly)")
        _write_forms_to(VERIFY_DIR, raw, d["cases"])
        return
    _write_forms_to(VERIFY_DIR / round_label, raw, d["cases"], {"unresolved": list(d["unresolved"])})


def write_round_forms(round_label, case_ids):
    """Forms for an explicit subset of the current pairs (e.g. round 2)."""
    raw = pairs_path().read_bytes()
    by_id = {c["case_id"]: c for c in json.loads(raw)["cases"]}
    missing = [cid for cid in case_ids if cid not in by_id]
    if missing:
        raise ValueError(f"case_ids not resolved in pairs_v2.json: {missing}")
    _write_forms_to(VERIFY_DIR / round_label, raw, [by_id[cid] for cid in case_ids],
                    {"case_ids": list(case_ids)})


# ---- Excel workbooks ----
INSTRUCTIONS = "Instructions"
REVIEW = "Review"
AUTHOR_EDITS = "Author_edits"
AUTHOR_COLS = ["case_id", "sham_type", "clinical_question", "authentic_organization",
               "authentic_excerpt_text", "field", "find", "replace", "rationale"]
EDIT_FIELDS = ("excerpt_text", "metadata.organization")
AUTHOR_INSTRUCTION = (
    "Write ONE minimal edit per case, of the declared type. subtle_inversion: invert the polarity of one "
    "recommendation in excerpt_text (field = excerpt_text). authority_mimicry: a plausible look-alike of the "
    "organization name (field = metadata.organization). `find` must be copied exactly from the authentic text "
    "(authentic_excerpt_text or authentic_organization) and occur exactly once; `replace` is what it becomes. "
    "Avoid grammar or capitalization artifacts a model could use as a surface cue. Optional: rationale.")
LONG_COLS = {"vignette_as_shown_to_model", "authentic_as_shown_to_model", "modified_as_shown_to_model",
             "what_changed", "source_guideline", "clinical_question", "notes", "authentic_excerpt_text",
             "find", "replace", "rationale"}


def _set(ws, row, col, value):
    from openpyxl.cell.cell import Cell  # noqa: F401
    c = ws.cell(row=row, column=col)
    c.value = value
    if isinstance(value, str):
        c.data_type = "s"  # never let a leading "=" turn text into a formula
    return c


def _dropdown(ws, col_letter, first, last, values):
    from openpyxl.worksheet.datavalidation import DataValidation
    dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Invalid", error=f"Choose one of: {', '.join(values)}")
    dv.add(f"{col_letter}{first}:{col_letter}{max(last, first)}")
    ws.add_data_validation(dv)


def _style_table(ws, header_row, columns):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    for i, name in enumerate(columns, start=1):
        h = ws.cell(row=header_row, column=i)
        h.font = Font(bold=True)
        h.fill = PatternFill("solid", fgColor="D9E1F2")
        letter = get_column_letter(i)
        if name in LONG_COLS:
            ws.column_dimensions[letter].width = 60 if name != "notes" else 40
        elif name == "packet_sha256":
            ws.column_dimensions[letter].width = 18
        else:
            ws.column_dimensions[letter].width = 16 if len(name) < 16 else min(len(name) + 2, 28)
        for r in range(header_row + 1, ws.max_row + 1):
            ws.cell(row=r, column=i).alignment = Alignment(wrap_text=name in LONG_COLS, vertical="top")
    ws.freeze_panes = f"A{header_row + 1}"
    ws.auto_filter.ref = f"A{header_row}:{get_column_letter(len(columns))}{max(ws.max_row, header_row + 1)}"


def export_xlsx(csv_path, xlsx_path, reviewer, round_label, queue_csv=None):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    with open(csv_path, newline="", encoding="utf-8") as fh:
        table = list(csv.reader(fh))
    header, body = table[0], table[1:]
    wb = Workbook()
    ins = wb.active
    ins.title = INSTRUCTIONS
    ins.column_dimensions["A"].width = 130
    lines = [f"Reviewer: {reviewer}    Round: {round_label}",
             "Answer independently; do not discuss until both are returned.",
             "Fill the dropdown answer columns on the Review sheet (Y/N; Q3 also P). Do not edit or reorder other columns.",
             ""] + readme_text().splitlines()
    if queue_csv is not None:
        lines += ["", f"Sheet {AUTHOR_EDITS}: " + AUTHOR_INSTRUCTION]
    for i, line in enumerate(lines, start=1):
        c = _set(ins, i, 1, line)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ins["A1"].font = Font(bold=True, size=13)
    ws = wb.create_sheet(REVIEW)
    for j, name in enumerate(header, start=1):
        _set(ws, 1, j, name)
    for i, row in enumerate(body, start=2):
        for j, val in enumerate(row, start=1):
            if val != "":
                _set(ws, i, j, val)
    _style_table(ws, 1, header)
    for q, _ in QUESTIONS:
        if q in header:
            _dropdown(ws, get_column_letter(header.index(q) + 1), 2, len(body) + 1,
                      ("Y", "N", "P") if q == "q3_authentic_concordant" else ("Y", "N"))
    if queue_csv is not None:
        with open(queue_csv, newline="", encoding="utf-8") as fh:
            queue = list(csv.DictReader(fh))
        ae = wb.create_sheet(AUTHOR_EDITS)
        c = _set(ae, 1, 1, AUTHOR_INSTRUCTION)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ae.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(AUTHOR_COLS))
        ae.row_dimensions[1].height = 75
        for j, name in enumerate(AUTHOR_COLS, start=1):
            _set(ae, 2, j, name)
        for i, q in enumerate(queue, start=3):
            for j, name in enumerate(AUTHOR_COLS, start=1):
                if q.get(name):
                    _set(ae, i, j, q[name])
        _style_table(ae, 2, AUTHOR_COLS)
        _dropdown(ae, get_column_letter(AUTHOR_COLS.index("field") + 1), 3, len(queue) + 2, EDIT_FIELDS)
    wb.save(xlsx_path)


def _cell_text(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v)


def import_review_xlsx(xlsx_path, csv_path):
    from openpyxl import load_workbook
    wb = load_workbook(xlsx_path, read_only=False, data_only=True)
    if REVIEW not in wb.sheetnames:
        raise ValueError(f"{xlsx_path}: no '{REVIEW}' sheet")
    rows = [[_cell_text(c.value) for c in row] for row in wb[REVIEW].iter_rows()]
    while rows and not any(rows[-1]):
        rows.pop()
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)


def import_author_edits(xlsx_path, author, date):
    from openpyxl import load_workbook
    wb = load_workbook(xlsx_path, data_only=True)
    if AUTHOR_EDITS not in wb.sheetnames:
        raise ValueError(f"{xlsx_path}: no '{AUTHOR_EDITS}' sheet")
    rows = [[_cell_text(c.value) for c in row] for row in wb[AUTHOR_EDITS].iter_rows()]
    start = next((i for i, r in enumerate(rows) if r and r[0] == "case_id"), None)
    if start is None:
        raise ValueError(f"{AUTHOR_EDITS}: header row (starting with case_id) not found")
    col = {name: rows[start].index(name) for name in AUTHOR_COLS if name in rows[start]}
    originals, _ = load_original_cases()
    edits, problems = {}, []
    for r in rows[start + 1:]:
        cid = r[col["case_id"]].strip()
        if not cid:
            continue
        get = lambda n: r[col[n]] if n in col and col[n] < len(r) else ""  # noqa: E731
        edit = {"field": get("field").strip(), "find": get("find"), "replace": get("replace"),
                "author": author, "date": date}
        if get("rationale").strip():
            edit["rationale"] = get("rationale").strip()
        if cid in edits:
            problems.append(f"{cid}: appears more than once")
        elif cid not in originals:
            problems.append(f"{cid}: unknown case_id")
        elif not edit["find"].strip() or not edit["replace"].strip():
            problems.append(f"{cid}: blank find or replace")
        elif edit["field"] not in EDIT_FIELDS:
            problems.append(f"{cid}: unknown field {edit['field']!r}")
        else:
            expected = {"subtle_inversion": "excerpt_text", "authority_mimicry": "metadata.organization"}.get(
                originals[cid]["eval"]["sham_trap_type"])
            try:
                if expected and edit["field"] != expected:
                    raise TransformError(f"{originals[cid]['eval']['sham_trap_type']} edits belong in {expected}")
                apply_manual_edit(originals[cid]["tools"]["S"], edit)
            except (TransformError, KeyError) as e:
                problems.append(f"{cid}: {e}")
            else:
                edits[cid] = edit
    if problems:
        raise ValueError("author edits rejected, nothing written:\n  " + "\n  ".join(problems))
    path = DATA_V2 / "manual_edits.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(edits)
    path.write_text(json.dumps(merged, indent=1, ensure_ascii=False) + "\n")
    return edits


def merge_rounds(round_labels, out_dir=None):
    """Concatenate the rounds' forms per reviewer into out_dir/review_<R>.csv and write the top-level
    manifest for the current pairs_v2.json."""
    out_dir = VERIFY_DIR if out_dir is None else Path(out_dir)
    raw = pairs_path().read_bytes()
    current = {r["case_id"]: r["packet_sha256"] for r in packet_rows(json.loads(raw)["cases"])}
    merged = {}
    for reviewer, _ in REVIEWERS:
        header, rows, seen = None, [], {}
        for label in round_labels:
            path = VERIFY_DIR / label / f"review_{reviewer}.csv"
            with open(path, newline="", encoding="utf-8") as fh:
                rd = csv.DictReader(fh)
                if header is None:
                    header = rd.fieldnames
                elif rd.fieldnames != header:
                    raise ValueError(f"{path}: columns differ from the first round")
                for row in rd:
                    cid = row["case_id"]
                    if cid in seen:
                        raise ValueError(f"{reviewer}: case_id {cid} appears in both {seen[cid]} and {label}")
                    seen[cid] = label
                    if current.get(cid) != row["packet_sha256"]:
                        raise ValueError(f"{reviewer}: packet_sha256 for {cid} ({label}) differs from the "
                                         "current render: the pair changed after it was reviewed")
                    rows.append(row)
        merged[reviewer] = (header, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    for reviewer, (header, rows) in merged.items():
        with open(out_dir / f"review_{reviewer}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
    (out_dir / MANIFEST).write_text(json.dumps(
        {"pairs_v2_sha256": hashlib.sha256(raw).hexdigest(), "packets": current,
         "merged_rounds": list(round_labels)}, indent=1) + "\n")


REVIEWER_NAMES = {"AG": "A.G.", "MO": "M.O."}


def _export_round(label, with_author_edits, force=False):
    d = VERIFY_DIR / label
    targets = [d / f"review_{r}.xlsx" for r, _ in REVIEWERS]
    existing = [t.name for t in targets if t.exists()]
    if existing and not force:
        raise RuntimeError(f"refusing to overwrite workbooks that may hold answers: {existing} (use --force)")
    for reviewer, _ in REVIEWERS:
        queue = DATA_V2 / "manual_edit_queue.csv" if (with_author_edits and reviewer == "AG") else None
        export_xlsx(d / f"review_{reviewer}.csv", d / f"review_{reviewer}.xlsx",
                    REVIEWER_NAMES[reviewer], label, queue_csv=queue)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="r2.verify.build_packets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("round1", "round2"):
        p = sub.add_parser(name)
        p.add_argument("--force", action="store_true", help="overwrite existing workbooks")
    p = sub.add_parser("import-edits")
    p.add_argument("xlsx")
    p.add_argument("--author", required=True)
    p.add_argument("--date", required=True)
    p = sub.add_parser("import-review")
    p.add_argument("xlsx")
    p.add_argument("csv")
    p = sub.add_parser("merge")
    p.add_argument("rounds", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "round1":
        write_forms("round1", allow_unresolved=True)
        _export_round("round1", with_author_edits=True, force=a.force)
    elif a.cmd == "round2":
        ids = json.loads((VERIFY_DIR / "round1" / MANIFEST).read_text())["unresolved"]
        write_round_forms("round2", ids)
        _export_round("round2", with_author_edits=False, force=a.force)
    elif a.cmd == "import-edits":
        print(f"{len(import_author_edits(a.xlsx, a.author, a.date))} edits written to data/v2/manual_edits.json")
    elif a.cmd == "import-review":
        import_review_xlsx(a.xlsx, a.csv)
    elif a.cmd == "merge":
        merge_rounds(a.rounds)


if __name__ == "__main__":
    main()
