import csv
import json
from collections import Counter

import pytest

from r2 import profiles
from r2.analysis import supp_tables as st
from r2.paths import AUDIT_DIR, LEGACY_EVAL

RESULT_TREES = [LEGACY_EVAL / "results", LEGACY_EVAL / "results_opensource"] if LEGACY_EVAL else []
MOUNTED = bool(RESULT_TREES) and all(t.exists() for t in RESULT_TREES)
mounted = pytest.mark.skipif(not MOUNTED, reason="legacy results tree not mounted (R2_LEGACY_ROOT)")


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    out = tmp_path_factory.mktemp("tables")
    res = st.write_all(out)
    return out, res


def read(out, name):
    with open(out / f"{name}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_audit_by_type_has_ten_types_and_matches_summary(written):
    out, _ = written
    rows = read(out, "S_audit_by_type")
    summ = json.loads((AUDIT_DIR / "pair_audit_summary.json").read_text())
    assert len(rows) == 10
    assert all(r["root_cause"] for r in rows)
    for col in ("excerpt_differs", "identical", "metadata_only"):
        assert sum(int(r[col]) for r in rows) == summ["visible_status"][col]
    for col in ("visible", "hidden_only", "no_edit"):
        assert sum(int(r[col]) for r in rows) == summ["raw_status"][col]
    assert sum(int(r["pairs"]) for r in rows) == 500


def test_hash_table_sums_to_10500_with_no_mismatch(written):
    out, _ = written
    rows = read(out, "S_hash_verification")
    assert {r["profile"] for r in rows} == set(profiles.PROFILES)
    assert sum(int(r["records"]) for r in rows) == 10500
    assert sum(int(r["verified"]) for r in rows) == 10496
    assert sum(int(r["no_logged_hash"]) for r in rows) == 4
    assert sum(int(r["mismatch"]) for r in rows) == 0
    assert all(r["verification_scope"] for r in rows)


def test_original_pipelines_one_row_per_profile(written):
    out, _ = written
    rows = read(out, "S_original_pipelines")
    assert [r["profile"] for r in rows] == list(profiles.PROFILES)
    for r in rows:
        assert all(r[c] for c in ("models", "tool_presentation", "vignette_fields",
                                  "system_prompt_source", "decoding"))
    by = {r["profile"]: r for r in rows}
    assert by["together_inline6"]["decoding"].startswith("temperature 0.7")
    assert "run_gemini_experiment.py" in by["gemini_inline7"]["system_prompt_source"]


@mounted
def test_run_inventory_covers_every_results_dir(written):
    out, _ = written
    rows = read(out, "S_run_inventory")
    expected = {(t.name, d.name) for t in RESULT_TREES for d in t.iterdir()
                if d.is_dir() and not d.name.startswith("._")}
    assert {(r["results_tree"], r["run_dir"]) for r in rows} == expected
    canon = [r for r in rows if r["canonical"] == "Y"]
    assert len(canon) == 21
    assert sum(int(r["valid_decisions_scored"]) for r in canon) >= 21 * 500
    for r in rows:
        assert r["canonical"] in ("Y", "N")
        if r["canonical"] == "N":
            assert r["reason_not_canonical"]
        assert r["input_file_hash"]
    q = [r for r in rows if "Qwen2.5_1.5B" in r["run_dir"]][0]
    assert q["records"] == "500" and q["valid_decisions_scored"] == "0"
    assert "non-serverless" in q["reason_not_canonical"]


@mounted
def test_inventory_reasons_attributed_and_not_redundant(written):
    out, _ = written
    for r in read(out, "S_run_inventory"):
        reason = r["reason_not_canonical"]
        if r["canonical"] == "N":
            assert "(per " in reason, r["run_dir"]
            if reason.count("; ") and "21-model" in reason:
                raise AssertionError(reason)
    assert (out / "README.txt").read_text().count("non-null") == 1


def test_original_reanalysis_labeled_exploratory(written):
    out, _ = written
    rows = read(out, "S_original_reanalysis")
    assert {r["analysis"] for r in rows} == {"exploratory"}
    types = [r for r in rows if r["stratum"] == "sham_type"]
    models = [r for r in rows if r["stratum"] == "model"]
    assert len(types) == 10 and len(models) == 21
    miss = [r for r in types if r["level"] == "missing_warning"][0]
    assert miss["discriminable_n"] == "0"
    assert miss["all_n"] == "840"


def test_modifications_by_type(written):
    out, _ = written
    rows = read(out, "S_modifications_by_type")
    assert len(rows) == 10
    assert sum(int(r["pairs_total"]) for r in rows) == 500
    assert sum(int(r["pairs_visible_in_original_runs"]) for r in rows) == 333
    for r in rows:
        assert r["transform"] and r["fields_changed"]


def test_write_all_skips_the_run_inventory_without_the_legacy_root(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "LEGACY_EVAL", None)
    res = st.write_all(tmp_path)
    assert set(res["skipped"]) == {"S_run_inventory"} and "R2_LEGACY_ROOT" in res["skipped"]["S_run_inventory"]
    assert set(res["written"]) == set(st.AVAILABLE_TABLES) - {"S_run_inventory"}
    assert not (tmp_path / "S_run_inventory.csv").exists()


def test_no_board_certified_anywhere(written):
    out, _ = written
    for p in out.glob("*.csv"):
        text = p.read_text().lower()
        assert "board-certified" not in text and "board certified" not in text


@pytest.mark.skipif(LEGACY_EVAL is None or not LEGACY_EVAL.exists(),
                    reason="legacy scripts not mounted (R2_LEGACY_ROOT)")
def test_system_prompt_cites_point_at_the_prompt_string():
    import re
    for r in st.original_pipelines():
        m = re.match(r"(\S+?):(\d+)-(\d+)", r["system_prompt_source"])
        path, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        lines = (LEGACY_EVAL / path).read_text().splitlines()
        assert "You are a clinical decision support evaluator" in lines[lo - 1], r["profile"]
        assert lines[hi - 1].rstrip().endswith('"""'), r["profile"]
        assert not any("defended" in l.lower() or "POTENTIALLY MALICIOUS" in l for l in lines[lo - 1:hi]), r["profile"]


ALLOWED_SOURCES = {"manifest", "results.jsonl", "logged errors", "run logs", "author record"}


@mounted
def test_inventory_reasons_use_plain_sources_not_internal_labels(written):
    import re
    out, _ = written
    for r in read(out, "S_run_inventory"):
        reason = r["reason_not_canonical"]
        assert "AUTHOR_FACTS" not in reason, r["run_dir"]
        assert set(re.findall(r"\(per ([^)]*)\)", reason)) <= ALLOWED_SOURCES, reason
    readme = (out / "README.txt").read_text()
    assert "AUTHOR_FACTS" not in readme
    assert "run logs" in readme and "author record" in readme


def test_quote_error_keeps_short_messages_whole():
    assert st._quote_error("Request timed out.") == "Request timed out."
    msg = "x " * 80  # exactly 160 characters
    assert st._quote_error(msg) == msg


def test_quote_error_cuts_long_messages_at_a_word_boundary():
    msg = "404 models/gemini-2.5-flash-lite-preview-06-17 is not found for API version v1beta, or is " \
          "not supported for generateContent. Call ListModels to see the list of available models and " \
          "their supported methods."
    q = st._quote_error(msg)
    assert q.endswith("…") and len(q) <= 161
    stem = q[:-1]
    assert msg.startswith(stem) and msg[len(stem)].isspace() and not stem[-1].isspace()
    assert q == msg[:msg.rfind(" ", 0, 161)] + "…"


def test_quote_error_never_cuts_a_single_long_token():
    token = "a" * 200
    assert st._quote_error(token) == token


@mounted
def test_inventory_quotes_logged_errors_whole_or_at_a_word_boundary(written):
    out, _ = written
    seen = 0
    for r in read(out, "S_run_inventory"):
        reason = r["reason_not_canonical"]
        if "logged the error: " not in reason:
            continue
        seen += 1
        quoted = reason.split("logged the error: ", 1)[1].split("; ")[0]
        lines = (LEGACY_EVAL / r["results_tree"] / r["run_dir"] / "results.jsonl").read_text().splitlines()
        recs = [json.loads(l) for l in lines if l.strip()]
        fulls = Counter(str(x["errors"][0]) for x in recs if x.get("errors") and not
                        (x.get("model_decision") is not None and x.get("selected_tool_correct") is not None))
        full = fulls.most_common(1)[0][0]
        if quoted.endswith("…"):
            stem = quoted[:-1]
            assert len(full) > 160 and len(stem) <= 160, r["run_dir"]
            assert full.startswith(stem) and full[len(stem)].isspace(), r["run_dir"]
        else:
            assert quoted == full, r["run_dir"]
    assert seen >= 10


# ---------------------------------------------------------------- invalid-reply handling and coerced choices

def test_original_pipelines_state_invalid_reply_handling(written):
    out, _ = written
    by = {r["profile"]: r for r in read(out, "S_original_pipelines")}
    assert all(by[p]["invalid_reply_handling"] for p in profiles.PROFILES)
    assert "unclear" in by["openai_fc"]["invalid_reply_handling"]
    for p in profiles.PROFILES:
        if p != "openai_fc":
            assert "records A" in by[p]["invalid_reply_handling"], p
    assert by["deepseek_reasoner_inline7"]["decoding"] == "max_tokens 8192; no temperature parameter (provider default)"


def test_invalid_reply_cites_point_at_the_fallback_code():
    import re
    for r in st.original_pipelines():
        text = r["invalid_reply_handling"]
        if r["profile"] == "openai_fc":
            lines = (st.REPO / "src/runner.py").read_text().splitlines()
            assert "raise ValueError" in lines[435 - 1]
            assert 'setdefault("selected_tool", "unclear")' in lines[448 - 1]
            continue
        path, lo, hi = re.search(r"(scripts/\S+\.py):(\d+)-(\d+)", text).groups()
        lines = (st.REPO / path).read_text().splitlines()
        assert 'json_start = response_text.find("{")' in lines[int(lo) - 1], r["profile"]
        assert 'decision["selected_tool"] = selected' in lines[int(hi) - 1], r["profile"]
        block = "\n".join(lines[int(lo) - 1:int(hi)])
        assert '"selected_tool": "A"' in block and 'selected = "A"' in block, r["profile"]


def test_coerced_choices_table_matches_the_audit_summary(written):
    out, _ = written
    rows = read(out, "S_coerced_choices")
    summary = json.loads((AUDIT_DIR / "coerced_choices_summary.json").read_text())
    assert len(rows) == 21 and sum(int(r["records"]) for r in rows) == summary["n_records"]
    tot = lambda col: sum(int(r[col]) for r in rows)
    assert tot("coerced_total") == summary["coerced_total"]
    assert tot("coerced_no_json") + tot("coerced_json_decode_error") == summary["class_counts"]["coerced_parse_failure"]
    assert tot("coerced_invalid_choice") == summary["class_counts"]["coerced_invalid_choice"]
    ident = summary["by_visible_status"]["identical"]
    assert tot("coerced_on_identical_pairs") == ident.get("coerced_parse_failure", 0) + ident.get("coerced_invalid_choice", 0)
    assert all(int(r["coerced_total"]) == 0 for r in rows if r["profile"] != "together_inline6")
    for r in rows:
        assert int(r["coerced_total"]) + int(r["not_coerced"]) == int(r["records"])
        assert int(r["parse_failures_at_token_limit"]) <= int(r["coerced_no_json"]) + int(r["coerced_json_decode_error"])


def test_coerced_parse_failures_at_the_token_limit(written):  # from audit/coerced_choices.csv; no volume needed
    out, _ = written
    rows = {r["model"]: r for r in read(out, "S_coerced_choices")}
    assert sum(int(r["parse_failures_at_token_limit"]) for r in rows.values()) == 381
    q = rows["Qwen/Qwen3-Next-80B-A3B-Thinking"]
    assert int(q["parse_failures_at_token_limit"]) == int(q["coerced_no_json"]) + int(q["coerced_json_decode_error"]) == 329
    assert rows["Qwen/Qwen3-Next-80B-A3B-Thinking"]["runner_max_tokens"] == "2048"
    assert rows["gpt-4.1-2025-04-14"]["runner_max_tokens"] == "not set"


def test_coerced_choices_reads_token_limit_from_the_audit_csv_not_raw_files(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "LEGACY_EVAL", tmp_path / "not-mounted")
    monkeypatch.setattr(st, "LEGACY_RESULTS", tmp_path / "not-mounted" / "results", raising=False)
    rows = {r["model"]: r for r in st.coerced_choices()}
    assert sum(r["parse_failures_at_token_limit"] for r in rows.values()) == 381
    assert rows["Qwen/Qwen3-Next-80B-A3B-Thinking"]["parse_failures_at_token_limit"] == 329
    assert rows["ServiceNow-AI/Apriel-1.6-15b-Thinker"]["parse_failures_at_token_limit"] == 37
    assert rows["deepseek-reasoner"]["runner_max_tokens"] == 8192
    assert rows["gemini-2.5-flash-lite"]["runner_max_tokens"] == "not set"


@mounted
def test_inventory_error_counts_come_from_the_logs(written):
    out, _ = written
    rows = {r["run_dir"]: r for r in read(out, "S_run_inventory")}
    for tree in RESULT_TREES:
        for d in tree.iterdir():
            if d.name.startswith("._") or not d.is_dir():
                continue
            recs = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()] \
                if (d / "results.jsonl").exists() else []
            n429 = sum(1 for r in recs if r.get("model_decision") is None
                       and str((r.get("errors") or [""])[0]).startswith("429"))
            row = rows[d.name]
            if n429:
                assert f"(run logs) {n429} of {len(recs)} rows returned 429 rate-limit errors" in \
                    row["reason_not_canonical"], d.name
            n_err = sum(1 for r in recs if r.get("model_decision") is None and r.get("errors"))
            if row["canonical"] == "N" and n_err:
                assert f"(run logs) {n_err} of {len(recs)} rows" in row["reason_not_canonical"], d.name
    g = rows["run_20260107_181353"]["reason_not_canonical"]
    assert "28 of 59 rows returned 429 rate-limit errors" in g and "no stop reason logged" in g
    assert "74 of 116 rows returned 429" in rows["run_20260107_181634"]["reason_not_canonical"]


def test_write_csv_refuses_an_empty_table(tmp_path):
    with pytest.raises(ValueError, match="no rows"):
        st.write_csv(tmp_path / "S_empty.csv", [])
    assert not (tmp_path / "S_empty.csv").exists()


def test_fabricated_citation_transform_discloses_the_self_label_token():
    """Final fix F1: the original DOI suffix read 'fake'. The description covers the original transform only."""
    text = st.TRANSFORMS["fabricated_citation"][0]
    assert "10.9999/fake.NNNN" in text and "fake DOI" not in text
    assert "10.9999/cg." not in text and "corrected" not in text


def test_cue_descriptions_quote_the_audit(written):
    """The fabricated_citation and outdated_version rows state the original transform and the audit findings only:
    the format cues, the notes phrase that no configuration showed and the years that were not older, with numbers
    from audit/format_cues.json. Nothing about the corrected set of the repeated experiment (prepared, not run)."""
    out, _ = written
    rows = {r["sham_type"]: r["transform"] for r in read(out, "S_modifications_by_type")}
    fc = json.loads((AUDIT_DIR / "format_cues.json").read_text())
    span = fc["page_span"]
    text = rows["fabricated_citation"]
    assert "after a blank line that no authentic excerpt has" in text
    assert f"page range {span['min']:,} to {span['max']:,} pages long" in text
    assert "ending in the DOI 10.9999/fake.NNNN (the word 'fake' was visible to the models)" in text
    assert "'Supported by recent mega-trial data' only to metadata.notes, a field no configuration showed" in text
    text = rows["outdated_version"]
    assert "'Historical guideline version for reference only' only to metadata.notes, a field no configuration " \
           "showed" in text
    assert ("In 2 pairs the year was not older than the authentic one (CASE_387: 2012 against an authentic 2010; "
            "CASE_488: 2014 against an authentic 2014).") in text
    for t in rows.values():
        assert "{" not in t and "corrected" not in t.lower() and "now 20" not in t and "cue_fixes" not in t


def test_cue_values_refuse_an_audit_that_contradicts_the_descriptions(tmp_path, monkeypatch):
    fc = json.loads((AUDIT_DIR / "format_cues.json").read_text())
    fc["notes_phrases"]["phrases"]["Historical guideline version for reference only"]["model_visible_rows"] = 1
    (tmp_path / "format_cues.json").write_text(json.dumps(fc))
    monkeypatch.setattr(st, "AUDIT_DIR", tmp_path)
    with pytest.raises(ValueError, match="notes phrases reached model inputs"):
        st.cue_values()
    fc["notes_phrases"]["phrases"]["Historical guideline version for reference only"]["model_visible_rows"] = 0
    fc["blank_line_cue"]["authentic_payload_rows"] = 3
    (tmp_path / "format_cues.json").write_text(json.dumps(fc))
    with pytest.raises(ValueError, match="blank line"):
        st.cue_values()


@mounted
def test_run_inventory_says_preliminary_run_not_pilot(written):
    out, _ = written
    rows = read(out, "S_run_inventory")
    notes = " ".join(" ".join(r.values()) for r in rows)
    assert "pilot" not in notes.lower()
    assert any(n.startswith("Local preliminary run with") for n in st.RUN_NOTES.values())
    assert any(n.startswith("Preliminary run of") for n in st.RUN_NOTES.values())


def test_readme_text_describes_the_exclusion_only_reanalysis(written):
    """README_TEXT (written to tables/README.txt on every run) carries none of the vocabulary of the repeated
    experiment, which was prepared, not run (R2 decision 2026-10-18)."""
    out, _ = written
    text = st.README_TEXT
    assert (out / "README.txt").read_text() == text
    stale = ("repeated experiment", "repeated-experiment", "21,000", "corrected stimulus set", "corrected set",
             "round 2", "freeze", "frozen", "pilot", "harness", "physician-authored", "re-rendered", "pre-results")
    assert not [t for t in stale if t in text.lower()]
    assert text.startswith("Notes on the supplement tables (R2, exclusion-only reanalysis)")
    assert "S_data_files.csv (Supplementary Table S14" in text
