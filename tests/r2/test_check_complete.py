import json

from r2.run.check_complete import check, export_flat

SPEC = {"original_name": "m", "model_id": "m-1"}


def _rec(cid, order, status, error=None):
    labels = {"A": "T", "B": "S"} if order == "TS" else {"A": "S", "B": "T"}
    r = {"case_id": cid, "order": order, "status": status, "labels": labels, "stimulus_sha256": "s1",
         "selected_tool": "A" if status == "ok" else None, "attempts": [] if error else [{}],
         "harness_commit": "h1", "model_served": "m-1", "system_fingerprint": None}
    r["correct"] = int(labels["A"] == "T") if status == "ok" else None
    if error:
        r["error"] = error
    return r


def _write(path, cells):
    with open(path, "w") as fh:
        for cell in cells:
            fh.write(json.dumps(_rec(*cell)) + "\n")


def test_completeness_check_flags_missing_and_duplicates(tmp_path):
    _write(tmp_path / "m-1.jsonl", [("C1", "TS", "ok"), ("C1", "TS", "ok"), ("C1", "ST", "non_response"), ("C9", "TS", "ok")])
    (row,) = check(tmp_path, [SPEC], ["C1", "C2"])
    assert row["cells"] == 3 and row["missing"] == 2 and row["duplicates"] == 1
    assert row["unexpected"] == 1
    assert row["ok"] == 2 and row["non_response"] == 1  # statuses of the final record per cell
    assert row["superseded_errors"] == 0 and row["errors"] == 0


def test_superseded_error_records_are_not_duplicates(tmp_path):
    _write(tmp_path / "m-1.jsonl", [("C1", "TS", "non_response", "BadRequestError: x"), ("C1", "TS", "ok"),
                                    ("C1", "ST", "ok"),
                                    ("C2", "TS", "non_response", "BadRequestError: y"),
                                    ("C2", "TS", "non_response", "BadRequestError: y"),
                                    ("C2", "ST", "ok")])
    (row,) = check(tmp_path, [SPEC], ["C1", "C2"])
    assert row["duplicates"] == 0 and row["missing"] == 0
    assert row["superseded_errors"] == 2
    assert row["errors"] == 1  # C2/TS still ends in an error record
    assert row["ok"] == 3 and row["non_response"] == 1


def test_two_non_superseded_records_are_a_duplicate(tmp_path):
    _write(tmp_path / "m-1.jsonl", [("C1", "TS", "ok"), ("C1", "TS", "non_response", "BadRequestError: x"),
                                    ("C1", "ST", "ok")])
    (row,) = check(tmp_path, [SPEC], ["C1"])
    assert row["duplicates"] == 1 and row["superseded_errors"] == 0 and row["errors"] == 1


def test_export_flat_tolerates_error_cells(tmp_path, monkeypatch):
    from r2.run import check_complete

    # Monkeypatch DATA_V2 to tmp_path
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(check_complete, "DATA_V2", data_dir)
    monkeypatch.setattr(check_complete, "load_specs", lambda *a, **k: [{"original_name": "m", "model_id": "m-1"}])

    # Create the results directory
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Write an ok record
    ok_record = {
        "original_name": "m",
        "model_id": "m-1",
        "model_served": "m-1",
        "case_id": "C1",
        "order": "TS",
        "labels": {"A": "T", "B": "F"},
        "status": "ok",
        "selected_tool": "A",
        "correct": 1,
        "harness_commit": "abc",
        "stimulus_sha256": "def",
        "finished_at": "2026-10-05T10:00:00Z",
        "decision": {"confidence": 0.8, "trust_rationale": "x", "differences_found": []},
    }

    # Write an error-cell record
    error_record = {
        "original_name": "m",
        "model_id": "m-1",
        "model_served": None,
        "case_id": "C2",
        "order": "ST",
        "labels": {"A": "T", "B": "F"},
        "status": "non_response",
        "selected_tool": None,
        "correct": None,
        "harness_commit": "abc",
        "stimulus_sha256": "def",
        "finished_at": "2026-10-05T10:01:00Z",
        "decision": None,
        "error": "BadRequestError: x",
        "attempts": [],
    }

    results_file = results_dir / "m-1.jsonl"
    with open(results_file, "w") as fh:
        fh.write(json.dumps(ok_record) + "\n")
        fh.write(json.dumps(error_record) + "\n")

    # Call export_flat
    dest = export_flat(results_dir)

    # Verify the CSV exists and has the expected content
    assert dest.exists()
    lines = dest.read_text().strip().split("\n")

    # Should have header + 2 data rows
    assert len(lines) == 3, f"Expected 3 lines (header + 2 data), got {len(lines)}"

    # Check header includes error column at the end
    header = lines[0]
    assert "error" in header, f"Error column not found in header: {header}"

    # Check that the error text appears in the second data row
    assert "BadRequestError: x" in lines[2], f"Error text not found in row: {lines[2]}"


def test_export_flat_uses_last_record_per_cell(tmp_path, monkeypatch):
    import csv

    from r2.run import check_complete
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(check_complete, "DATA_V2", data_dir)
    monkeypatch.setattr(check_complete, "load_specs", lambda *a, **k: [{"original_name": "m", "model_id": "m-1"}])
    results = tmp_path / "results"
    results.mkdir()
    base = {"original_name": "m", "model_id": "m-1", "finished_at": "t", "decision": None}
    _write(results / "m-1.jsonl", [("C1", "TS", "non_response", "BadRequestError: x"), ("C1", "TS", "ok"),
                                   ("C1", "ST", "ok")])
    recs = [json.loads(l) for l in (results / "m-1.jsonl").read_text().splitlines()]
    (results / "m-1.jsonl").write_text("".join(json.dumps({**base, **r}) + "\n" for r in recs))
    rows = list(csv.DictReader(open(export_flat(results))))
    assert [(r["case_id"], r["order"], r["status"], r["error"]) for r in rows] == [
        ("C1", "TS", "ok", ""), ("C1", "ST", "ok", "")]


# ---- I6: orphan files, per-record content checks, provenance, readers follow the specs ----
import pytest  # noqa: E402

from r2.run.check_complete import OrphanResults  # noqa: E402


def test_orphan_result_file_fails_check(tmp_path):
    _write(tmp_path / "m-1.jsonl", [("C1", "TS", "ok"), ("C1", "ST", "ok")])
    (tmp_path / "incidents.jsonl").write_text("{}\n")
    (tmp_path / "._m-1.jsonl").write_bytes(b"\x00\x05")
    (tmp_path / "m-1.lock").write_text("")
    (row,) = check(tmp_path, [SPEC], ["C1"])
    assert row["invalid_records"] == 0
    _write(tmp_path / "dropped-model.jsonl", [("C1", "TS", "ok")])
    with pytest.raises(OrphanResults, match="dropped-model.jsonl"):
        check(tmp_path, [SPEC], ["C1"])


def test_invalid_records_counted(tmp_path):
    recs = [_rec("C1", "TS", "ok"), _rec("C1", "ST", "ok"), _rec("C2", "TS", "ok"), _rec("C2", "ST", "ok"),
            _rec("C3", "TS", "ok"), _rec("C3", "ST", "non_response")]
    recs[1]["labels"] = {"A": "T", "B": "S"}          # ST must put the sham in A
    recs[2]["correct"] = 0                             # TS, selected A (= T) is correct
    recs[3]["stimulus_sha256"] = "s2"                  # stimulus differs within the file
    recs[5]["labels"] = {"A": "T", "B": "S"}           # non-ok records are not content-checked
    (tmp_path / "m-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (row,) = check(tmp_path, [SPEC], ["C1", "C2", "C3"])
    assert row["invalid_records"] == 3


def test_provenance_columns(tmp_path):
    recs = [_rec("C1", "TS", "ok"), _rec("C1", "ST", "ok")]
    recs[1].update(harness_commit="h2", system_fingerprint="fp_1", model_served="m-1-0905")
    (tmp_path / "m-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (row,) = check(tmp_path, [SPEC], ["C1"])
    assert row["harness_commit"] == "h1;h2" and row["system_fingerprint"] == "fp_1"
    assert row["model_served"] == "m-1;m-1-0905"


def test_gate_fails_on_invalid_records(tmp_path, monkeypatch):
    import csv
    import sys

    from r2.run import check_complete
    recs = [_rec("C1", "TS", "ok"), _rec("C1", "ST", "ok")]
    recs[0]["correct"] = 0
    out = tmp_path / "runs"
    out.mkdir()
    (out / "m-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    monkeypatch.setattr(check_complete, "load_v2_cases", lambda *a, **k: ({"C1": {}}, "s1"))
    monkeypatch.setattr(check_complete, "load_specs", lambda *a, **k: [SPEC])
    monkeypatch.setattr(check_complete, "REPO", tmp_path)
    monkeypatch.setattr(check_complete, "DATA_V2", tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_complete", str(out)])
    with pytest.raises(SystemExit, match="invalid records"):
        check_complete.main()
    assert not (tmp_path / "results_repeated.csv").exists()
    report = list(csv.DictReader(open(tmp_path / "analysis_repeated" / "completeness.csv")))
    assert report[0]["invalid_records"] == "1"


def test_export_flat_reads_only_current_spec_files(tmp_path, monkeypatch):
    import csv

    from r2.run import check_complete
    monkeypatch.setattr(check_complete, "DATA_V2", tmp_path)
    monkeypatch.setattr(check_complete, "load_specs", lambda *a, **k: [{"original_name": "m", "model_id": "m-1"},
                                                                        {"original_name": "n", "model_id": "n-1"}])
    results = tmp_path / "results"
    results.mkdir()
    base = {"model_id": "m-1", "finished_at": "t", "decision": None}
    (results / "m-1.jsonl").write_text(json.dumps({**base, **_rec("C1", "TS", "ok"), "original_name": "m"}) + "\n")
    (results / "old-model.jsonl").write_text(json.dumps({**base, **_rec("C1", "TS", "ok"), "original_name": "m"}) + "\n")
    rows = list(csv.DictReader(open(export_flat(results))))
    assert len(rows) == 1
