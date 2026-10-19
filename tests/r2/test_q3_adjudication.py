import pandas as pd
import pytest
from openpyxl import load_workbook

from r2.verify import evaluable as ev
from r2.verify import q3_adjudication as qa


def test_evaluable_pairs_and_discordant_q3():
    p = ev.load_pairs()
    assert len(p) == 333
    assert int(ev.evaluable_mask(p).sum()) == 25
    d = ev.discordant_q3(p)
    assert len(d) == 19
    assert list(d.index) == sorted(d.index)
    assert (d[f"AG_{ev.Q3}"] != d[f"MO_{ev.Q3}"]).all()


def _filled(tmp_path, answers=None, edit=None, drop=None, reverse=False):
    x = tmp_path / "q3.xlsx"
    n = qa.write_workbook(x)
    wb = load_workbook(x)
    ws = wb[qa.SHEET]
    col = qa.COLUMNS.index(f"consensus_{ev.Q3}") + 1
    for r in range(2, n + 2):
        ws.cell(row=r, column=col).value = (answers or {}).get(r, "y ")
    if edit:
        r, name, value = edit
        ws.cell(row=r, column=qa.COLUMNS.index(name) + 1).value = value
    if drop:
        ws.delete_rows(drop)
    if reverse:
        rows = [[c.value for c in row] for row in ws.iter_rows(min_row=2, max_row=ws.max_row)]
        for i, vals in enumerate(reversed(rows), start=2):
            for j, v in enumerate(vals, start=1):
                ws.cell(row=i, column=j).value = v
    wb.save(x)
    return x, n


def test_workbook_rows_and_round_trip(tmp_path):
    x, n = _filled(tmp_path, reverse=True, answers={2: "p", 3: "n", 4: "Y", 5: " p "})
    assert n == 19
    # Extract expected case_id→answer mapping from the saved workbook
    wb = load_workbook(x, data_only=True)
    ws = wb[qa.SHEET]
    consensus_col = qa.COLUMNS.index(f"consensus_{ev.Q3}") + 1
    expected = {}
    for row in range(2, ws.max_row + 1):
        cid = str(ws.cell(row=row, column=1).value).strip()
        ans = str(ws.cell(row=row, column=consensus_col).value).strip().upper()
        expected[cid] = ans
    # Import and verify answers match per case_id
    out = qa.import_workbook(x, "2026-10-20", csv_path=tmp_path / "c.csv")
    assert len(out) == 19
    actual = out.set_index("case_id")[f"consensus_{ev.Q3}"].to_dict()
    assert actual == expected  # all 19 case_id→answer mappings correct
    assert list(out.case_id) == sorted(out.case_id)                         # matched by case_id, not row order
    assert (pd.read_csv(tmp_path / "c.csv").adjudicated_on == "2026-10-20").all()


def _add_extra_cid(tmp_path, x):
    """Add a row with unknown case_id CASE_999 to the workbook."""
    wb = load_workbook(x)
    ws = wb[qa.SHEET]
    row = ws.max_row + 1
    for j, name in enumerate(qa.COLUMNS, start=1):
        val = "CASE_999" if name == "case_id" else "Y" if name == f"consensus_{ev.Q3}" else None
        ws.cell(row=row, column=j).value = val
    wb.save(x)


@pytest.mark.parametrize("kind", ["blank", "unknown", "changed_original", "missing_row", "duplicate_cid", "extra_cid"])
def test_import_refuses(tmp_path, kind):
    if kind == "blank":
        kw = {"answers": {2: None}}
        match = r"row 2.*the consensus must be"
    elif kind == "unknown":
        kw = {"answers": {3: "maybe"}}
        match = r"row 3.*the consensus must be"
    elif kind == "changed_original":
        kw = {"edit": (4, f"AG_{ev.Q3}", "Y")}
        match = r"row 4.*changed from"
    elif kind == "missing_row":
        kw = {"drop": 5}
        match = r"no row for"
    elif kind == "duplicate_cid":
        kw = {}
        match = r"appears twice"
    elif kind == "extra_cid":
        kw = {}
        match = r"is not one of the.*discordant pairs"

    x, _ = _filled(tmp_path, **kw)
    if kind == "duplicate_cid":
        wb = load_workbook(x)
        ws = wb[qa.SHEET]
        row2_cid = ws.cell(row=2, column=1).value
        ws.cell(row=3, column=1).value = row2_cid
        wb.save(x)
    elif kind == "extra_cid":
        _add_extra_cid(tmp_path, x)

    with pytest.raises(ValueError, match=match):
        qa.import_workbook(x, "2026-10-20", csv_path=tmp_path / "c.csv")
    assert not (tmp_path / "c.csv").exists()


def test_import_refuses_bad_date(tmp_path):
    x, _ = _filled(tmp_path)
    with pytest.raises(ValueError):
        qa.import_workbook(x, "20 Oct", csv_path=tmp_path / "c.csv")


def _consensus_csv(path, answers, date="2026-10-19"):
    pd.DataFrame({"case_id": [f"CASE_{i:03d}" for i in range(len(answers))], "sham_type": "prompt_injection",
                  f"AG_{ev.Q3}": "P", f"MO_{ev.Q3}": "Y", f"consensus_{ev.Q3}": answers, "consensus_note": "",
                  "adjudicated_on": date}).to_csv(path, index=False)


def test_status_consensus_paragraph(tmp_path, monkeypatch):
    from r2.verify import round1_summary as rs
    monkeypatch.setattr(rs, "CONSENSUS_CSV", tmp_path / "absent.csv")
    assert rs._consensus_text() == ""          # before the import, README_STATUS.md has no consensus paragraph
    csv_path = tmp_path / "c.csv"
    _consensus_csv(csv_path, ["P", "P", "N", "Y", "P"], date="2026-10-20")
    monkeypatch.setattr(rs, "CONSENSUS_CSV", csv_path)
    text = rs._consensus_text()
    assert "jointly adjudicated Q3 on 2026-10-20 for the 5 clinically evaluable pairs" in text
    assert "(Y 1, P 3, N 1)" in text


def test_data_files_consensus_role(tmp_path, monkeypatch):
    from r2 import data_files_doc as dfd
    _consensus_csv(tmp_path / "c.csv", ["N", "N", "Y"])
    monkeypatch.setattr(dfd, "REPO", tmp_path)
    role = dfd._q3_consensus_role("c.csv")
    assert "(adjudicated 2026-10-19) for the 3 clinically evaluable pairs" in role
    assert "(Y 1, P 0, N 2)" in role
