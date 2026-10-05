import json

import pytest

from r2.io import load_original_cases
from r2.run.harness import (CorruptResults, ModelMismatch, ModelUnavailable, StaleRecords, _create,
                            execute_cell, iter_result_files, parse_decision, results_path, run_model)
from r2.stimuli.render import render_messages

CASES, _ = load_original_cases()
SUBSET = {k: CASES[k] for k in ("CASE_001", "CASE_002")}
SPEC = {"original_name": "m", "model_id": "prov/m-1", "provider": "together",
        "temperature": 0, "max_tokens": 64, "token_param": "max_tokens"}
CTX = {"harness_commit": "abc", "stimulus_sha256": "def"}
GOOD = '{"selected_tool": "B", "confidence": 0.8, "trust_rationale": "x", "differences_found": ["y"]}'


class _Resp:
    def __init__(self, text, model):
        self._d = {"model": model, "choices": [{"message": {"content": text}}], "usage": {}}

    def model_dump(self):
        return self._d


class FakeClient:
    def __init__(self, replies, served=None):
        self.replies, self.calls, self.served = list(replies), [], served
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        return _Resp(self.replies.pop(0), self.served or kw["model"])


def test_messages_sent_equal_rendered():
    client = FakeClient([GOOD])
    rec = execute_cell(client, SPEC, CASES["CASE_001"], "ST", CTX, sleep=lambda s: None)
    expected = render_messages(CASES["CASE_001"], "ST")
    sent = client.calls[0]["messages"]
    assert sent == [{"role": "system", "content": expected["system"]},
                    {"role": "user", "content": expected["user"]}]
    assert client.calls[0]["temperature"] == 0 and client.calls[0]["max_tokens"] == 64
    assert rec["status"] == "ok" and rec["selected_tool"] == "B" and rec["correct"] == 1


def test_reasoning_spec_omits_temperature_and_uses_token_param():
    spec = {**SPEC, "temperature": None, "token_param": "max_completion_tokens", "max_tokens": 8192}
    client = FakeClient([GOOD])
    execute_cell(client, spec, CASES["CASE_001"], "TS", CTX, sleep=lambda s: None)
    assert "temperature" not in client.calls[0] and client.calls[0]["max_completion_tokens"] == 8192


def test_parse_think_tags_and_fences():
    text = "<think>maybe {\"selected_tool\": \"A\"}</think>\n```json\n" + GOOD + "\n```"
    assert parse_decision(text)["selected_tool"] == "B"
    assert parse_decision("") is None
    assert parse_decision('{"selected_tool": "C"}') is None


def test_refusal_becomes_non_response_after_one_retry():
    client = FakeClient(["I cannot provide medical advice.", "Sorry, I can't help."])
    rec = execute_cell(client, SPEC, CASES["CASE_001"], "TS", CTX, sleep=lambda s: None)
    assert rec["status"] == "non_response" and rec["correct"] is None
    assert len(client.calls) == 2 and len(rec["attempts"]) == 2


def test_served_model_mismatch_stops(tmp_path):
    client = FakeClient([GOOD] * 4, served="prov/other-model")
    with pytest.raises(ModelMismatch):
        run_model(SPEC, SUBSET, tmp_path, client, CTX, sleep=lambda s: None)


def test_both_orders_and_resume(tmp_path):
    client = FakeClient([GOOD] * 4)
    path = run_model(SPEC, SUBSET, tmp_path, client, CTX, sleep=lambda s: None)
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    assert sorted((r["case_id"], r["order"]) for r in recs) == [
        ("CASE_001", "ST"), ("CASE_001", "TS"), ("CASE_002", "ST"), ("CASE_002", "TS")]
    client2 = FakeClient([])
    run_model(SPEC, SUBSET, tmp_path, client2, CTX, sleep=lambda s: None)
    assert client2.calls == []


def test_resume_skips_completed(tmp_path):
    client = FakeClient([GOOD] * 2)
    run_model(SPEC, SUBSET, tmp_path, client, CTX, limit=1, sleep=lambda s: None)
    client2 = FakeClient([GOOD] * 2)
    run_model(SPEC, SUBSET, tmp_path, client2, CTX, sleep=lambda s: None)
    assert len(client2.calls) == 2
    lines = results_path(SPEC, tmp_path).read_text().splitlines()
    assert len(lines) == 4 and len({(json.loads(l)["case_id"], json.loads(l)["order"]) for l in lines}) == 4


def test_loader_skips_appledouble(tmp_path):
    (tmp_path / "m.jsonl").write_text("")
    (tmp_path / "._m.jsonl").write_bytes(b"\x00\x05\x16\x07")
    assert [p.name for p in iter_result_files(tmp_path)] == ["m.jsonl"]


def _fake_exc(name):
    return type(name, (Exception,), {})


class RaisingClient(FakeClient):
    def __init__(self, exc, replies=()):
        super().__init__(replies)
        self.exc = exc

    def create(self, **kw):
        if self.exc is not None:
            raise self.exc("boom")
        return super().create(**kw)


def _first_record(tmp_path):
    run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD] * 4), CTX, limit=1, sleep=lambda s: None)
    path = results_path(SPEC, tmp_path)
    return path, [json.loads(l) for l in path.read_text().splitlines()]


def test_stale_stimulus_sha_raises(tmp_path):
    path, recs = _first_record(tmp_path)
    recs[0]["stimulus_sha256"] = "other"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(StaleRecords):
        run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD] * 4), CTX, sleep=lambda s: None)


def test_stale_payload_sha_raises(tmp_path):
    path, recs = _first_record(tmp_path)
    recs[0]["payload_sha256"] = {"A": "x", "B": "y"}
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(StaleRecords):
        run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD] * 4), CTX, sleep=lambda s: None)


def test_bad_request_becomes_non_response_with_error():
    client = RaisingClient(_fake_exc("BadRequestError"))
    rec = execute_cell(client, SPEC, CASES["CASE_001"], "TS", CTX, sleep=lambda s: None)
    assert rec["status"] == "non_response" and rec["correct"] is None
    assert rec["error"].startswith("BadRequestError") and rec["attempts"] == []


def test_not_found_raises_model_unavailable(tmp_path):
    client = RaisingClient(_fake_exc("NotFoundError"))
    with pytest.raises(ModelUnavailable):
        run_model(SPEC, SUBSET, tmp_path, client, CTX, sleep=lambda s: None)


def test_other_errors_propagate():
    with pytest.raises(KeyError):
        execute_cell(RaisingClient(KeyError), SPEC, CASES["CASE_001"], "TS", CTX, sleep=lambda s: None)


def test_empty_first_reply_resends_original_messages():
    client = FakeClient(["", GOOD])
    rec = execute_cell(client, SPEC, CASES["CASE_001"], "TS", CTX, sleep=lambda s: None)
    assert client.calls[1]["messages"] == client.calls[0]["messages"]
    assert rec["status"] == "ok"


def test_unparseable_reply_gets_retry_prompt():
    client = FakeClient(["nope", GOOD])
    execute_cell(client, SPEC, CASES["CASE_001"], "TS", CTX, sleep=lambda s: None)
    assert client.calls[1]["messages"][-2:] == [
        {"role": "assistant", "content": "nope"},
        {"role": "user", "content": "Return only the JSON object described above, with no other text."}]


def test_think_block_handling():
    A = '{"selected_tool": "A"}'
    B = '{"selected_tool": "B"}'
    assert parse_decision("<think> reasoning " + A) is None
    assert parse_decision("draft " + A + "</think>\n" + B)["selected_tool"] == "B"
    assert parse_decision("draft " + A + "</think>") is None


def test_torn_final_line_dropped_and_resumes(tmp_path):
    path, recs = _first_record(tmp_path)
    with open(path, "a") as fh:
        fh.write('{"case_id": "CASE_00')
    client = FakeClient([GOOD] * 4)
    run_model(SPEC, SUBSET, tmp_path, client, CTX, sleep=lambda s: None)
    lines = path.read_text().splitlines()
    assert len(lines) == 4 and all(json.loads(l) for l in lines)
    assert len(client.calls) == 2


def test_corrupt_middle_line_raises(tmp_path):
    path, recs = _first_record(tmp_path)
    path.write_text("garbage\n" + json.dumps(recs[0]) + "\n")
    with pytest.raises(CorruptResults):
        run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD] * 4), CTX, sleep=lambda s: None)


class _Seq:
    def __init__(self, items):
        self.items = list(items)

    def create(self, **kw):
        it = self.items.pop(0)
        if isinstance(it, Exception):
            raise it
        return it

    @property
    def chat(self):
        return self

    completions = chat


def test_create_retries_then_succeeds():
    sleeps = []
    out = _create(_Seq([_fake_exc("RateLimitError")("x"), _fake_exc("APIConnectionError")("y"), "ok"]),
                  {}, sleeps.append)
    assert out == "ok" and sleeps == [2, 4]


def test_create_six_failures_then_success_on_seventh_call():
    sleeps = []
    out = _create(_Seq([_fake_exc("RateLimitError")("z")] * 6 + ["ok"]), {}, sleeps.append)
    assert out == "ok" and sleeps == [2, 4, 8, 16, 32, 64]


def test_create_gives_up_after_ten():
    sleeps = []
    with pytest.raises(Exception, match="z"):
        _create(_Seq([_fake_exc("RateLimitError")("z")] * 10), {}, sleeps.append)
    assert sleeps == [2, 4, 8, 16, 32, 64, 128, 256, 300]


def test_create_non_transient_raises_immediately():
    sleeps = []
    with pytest.raises(Exception, match="bad"):
        _create(_Seq([_fake_exc("BadRequestError")("bad")]), {}, sleeps.append)
    assert sleeps == []


# ---- I1: circuit breaker, error-only records retried on resume ----
SUBSET3 = {k: CASES[k] for k in ("CASE_001", "CASE_002", "CASE_003")}


class ScriptedClient(FakeClient):
    """Each call pops one item: an exception instance is raised, a string is returned as content."""

    def create(self, **kw):
        self.calls.append(kw)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Resp(item, self.served or kw["model"])


def _bad(text="boom"):
    return _fake_exc("BadRequestError")(text)


def _recs(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_breaker_trips_after_three_consecutive_cell_errors(tmp_path):
    client = ScriptedClient([_bad("a"), _bad("b"), _bad("c"), GOOD])
    with pytest.raises(ModelUnavailable):
        run_model(SPEC, SUBSET, tmp_path, client, CTX, sleep=lambda s: None)
    recs = _recs(results_path(SPEC, tmp_path))
    assert len(recs) == 3 and all("error" in r for r in recs)
    assert len(client.calls) == 3


def test_same_text_counter_resets_on_ok_record(tmp_path):
    client = ScriptedClient([_bad("same"), GOOD, _bad("same"), GOOD, _bad("same"), GOOD])
    run_model(SPEC, SUBSET3, tmp_path, client, CTX, sleep=lambda s: None)
    assert len(_recs(results_path(SPEC, tmp_path))) == 6


def test_same_text_counter_not_reset_by_non_response_without_error(tmp_path):
    # A parse failure (non_response, no error key) resets the consecutive rule but not the same-text rule.
    client = ScriptedClient([_bad("same"), "nope", "nope", _bad("same"), "nope", "nope", _bad("same"), GOOD])
    with pytest.raises(ModelUnavailable, match="same error text 3x"):
        run_model(SPEC, SUBSET3, tmp_path, client, CTX, sleep=lambda s: None)
    recs = _recs(results_path(SPEC, tmp_path))
    assert len(recs) == 5 and [("error" in r) for r in recs] == [True, False, True, False, True]


def test_breaker_not_tripped_by_distinct_interleaved_errors(tmp_path):
    client = ScriptedClient([_bad("a"), GOOD, _bad("b"), GOOD, _bad("c"), GOOD])
    run_model(SPEC, SUBSET3, tmp_path, client, CTX, sleep=lambda s: None)
    assert len(_recs(results_path(SPEC, tmp_path))) == 6


def test_error_only_record_is_reattempted_on_resume(tmp_path):
    run_model(SPEC, SUBSET, tmp_path, ScriptedClient([_bad(), GOOD, GOOD, GOOD]), CTX, sleep=lambda s: None)
    client2 = FakeClient([GOOD])
    run_model(SPEC, SUBSET, tmp_path, client2, CTX, sleep=lambda s: None)
    assert len(client2.calls) == 1
    recs = _recs(results_path(SPEC, tmp_path))
    assert len(recs) == 5
    first, last = recs[0], recs[-1]
    assert (first["case_id"], first["order"]) == (last["case_id"], last["order"])
    assert "error" in first and first["attempts"] == [] and last["status"] == "ok"
    client3 = FakeClient([])
    run_model(SPEC, SUBSET, tmp_path, client3, CTX, sleep=lambda s: None)
    assert client3.calls == []


# ---- Step 4b N1: a repeated identical cell error is final; re-attempted cells are outside the breaker ----
SUBSET6 = {k: CASES[k] for k in ("CASE_001", "CASE_002", "CASE_003", "CASE_004", "CASE_005", "CASE_006")}


class CellClient(FakeClient):
    """Deterministic per cell: a cell in `failing` always raises BadRequestError(text), the rest reply GOOD."""

    def __init__(self, cases, failing, text="Invalid prompt: your prompt was flagged"):
        super().__init__([])
        self.cell_of = {render_messages(c, o)["user"]: (cid, o) for cid, c in cases.items() for o in ("TS", "ST")}
        self.failing, self.text = set(failing), text

    def create(self, **kw):
        self.calls.append(kw)
        if self.cell_of[kw["messages"][1]["content"]] in self.failing:
            raise _bad(self.text)
        return _Resp(GOOD, kw["model"])


def _run_tolerant(*args, **kw):
    """One harness invocation of run_model; a breaker stop ends the invocation, as STOPPED does in main()."""
    try:
        run_model(*args, sleep=lambda s: None, **kw)
    except ModelUnavailable:
        pass


def test_persistent_identical_errors_do_not_deadlock_the_model(tmp_path):
    from r2.run.check_complete import check
    from r2.run.harness import final_records
    failing = {("CASE_001", "ST"), ("CASE_003", "ST"), ("CASE_005", "ST")}  # 3 non-adjacent of 12 cells
    for _ in range(2):  # a first run plus one resume
        _run_tolerant(SPEC, SUBSET6, tmp_path, CellClient(SUBSET6, failing), CTX)
    fin = final_records(results_path(SPEC, tmp_path))
    assert len(fin) == 12
    assert sum(r["status"] == "ok" for r in fin) == 9
    persistent = [r for r in fin if r.get("persistent_error")]
    assert {(r["case_id"], r["order"]) for r in persistent} == failing
    assert all(r["status"] == "non_response" and r["persistent_error"] is True and r["attempts"] == []
               for r in persistent)
    client3 = CellClient(SUBSET6, failing)
    run_model(SPEC, SUBSET6, tmp_path, client3, CTX, sleep=lambda s: None)
    assert client3.calls == []  # a cell whose last record is persistent_error is done
    (row,) = check(tmp_path, [SPEC], sorted(SUBSET6))
    assert (row["cells"], row["missing"], row["duplicates"]) == (12, 0, 0)
    assert (row["ok"], row["non_response"], row["errors"], row["superseded_errors"]) == (9, 3, 3, 3)


@pytest.mark.parametrize("first,second,final", [
    ("x", "x", True),
    ("x", "y", False),
    ("e" * 300 + "1", "e" * 300 + "2", True),  # compared on the first 200 characters of the error text
], ids=["same-text", "different-text", "same-first-200-chars"])
def test_reattempt_is_final_only_on_the_same_error_text(tmp_path, first, second, final):
    run_model(SPEC, SUBSET, tmp_path, ScriptedClient([_bad(first), GOOD]), CTX, limit=1, sleep=lambda s: None)
    run_model(SPEC, SUBSET, tmp_path, ScriptedClient([_bad(second)]), CTX, limit=1, sleep=lambda s: None)
    last = _recs(results_path(SPEC, tmp_path))[-1]
    assert last["status"] == "non_response" and last.get("persistent_error", False) is final
    client = FakeClient([GOOD])
    run_model(SPEC, SUBSET, tmp_path, client, CTX, limit=1, sleep=lambda s: None)
    assert len(client.calls) == (0 if final else 1)


def test_repeat_of_any_earlier_error_text_is_final(tmp_path, capsys):
    run_model(SPEC, SUBSET, tmp_path, ScriptedClient([_bad("x"), GOOD]), CTX, limit=1, sleep=lambda s: None)
    run_model(SPEC, SUBSET, tmp_path, ScriptedClient([_bad("y")]), CTX, limit=1, sleep=lambda s: None)
    run_model(SPEC, SUBSET, tmp_path, ScriptedClient([_bad("x")]), CTX, limit=1, sleep=lambda s: None)
    recs = _recs(results_path(SPEC, tmp_path))
    assert [r.get("persistent_error", False) for r in recs if r["order"] == "TS"] == [False, False, True]
    assert "FINAL m CASE_001 TS" in capsys.readouterr().out
    client = FakeClient([])
    run_model(SPEC, SUBSET, tmp_path, client, CTX, limit=1, sleep=lambda s: None)
    assert client.calls == []


def test_reattempted_cells_do_not_count_toward_the_breaker(tmp_path):
    # Run 1 trips on 3 consecutive distinct errors. On resume the same 3 cells fail again with new texts:
    # they are re-attempts, so they neither trip the breaker nor become final, and the fresh cells run.
    with pytest.raises(ModelUnavailable):
        run_model(SPEC, SUBSET3, tmp_path, ScriptedClient([_bad("a"), _bad("b"), _bad("c")]), CTX,
                  sleep=lambda s: None)
    client = ScriptedClient([_bad("d"), _bad("e"), _bad("f"), GOOD, GOOD, GOOD])
    run_model(SPEC, SUBSET3, tmp_path, client, CTX, sleep=lambda s: None)
    assert len(client.calls) == 6
    recs = _recs(results_path(SPEC, tmp_path))
    assert len(recs) == 9 and not any(r.get("persistent_error") for r in recs)


def test_reattempt_ok_does_not_reset_the_fresh_cell_breaker(tmp_path):
    # Re-attempted cells are invisible to the breaker: an ok re-attempt between fresh errors does not reset it.
    run_model(SPEC, SUBSET3, tmp_path, ScriptedClient([GOOD, GOOD, _bad("a"), GOOD, GOOD, GOOD]), CTX,
              sleep=lambda s: None)  # CASE_002/TS is error-only
    path = results_path(SPEC, tmp_path)
    keep = [r for r in _recs(path) if r["order"] == "TS" and r["case_id"] in ("CASE_001", "CASE_002")]
    _rewrite(path, keep)  # resume order: 001/ST fresh, 002/TS re-attempt, 002/ST fresh, 003/TS fresh
    client = ScriptedClient([_bad("p"), GOOD, _bad("q"), _bad("r")])
    with pytest.raises(ModelUnavailable, match="3 consecutive"):
        run_model(SPEC, SUBSET3, tmp_path, client, CTX, sleep=lambda s: None)
    assert len(client.calls) == 4


def test_final_records_takes_last_record_per_cell(tmp_path):
    from r2.run.harness import final_records
    run_model(SPEC, SUBSET, tmp_path, ScriptedClient([_bad(), GOOD, GOOD, GOOD]), CTX, sleep=lambda s: None)
    run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD]), CTX, sleep=lambda s: None)
    fin = final_records(results_path(SPEC, tmp_path))
    assert len(fin) == 4 and all(r["status"] == "ok" and "error" not in r for r in fin)


# ---- I2: resume validates prompt, spec mapping and decoding ----
def _rewrite(path, recs):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))


@pytest.mark.parametrize("field,value", [("system", "old system prompt"), ("user", "old user message"),
                                         ("model_id", "prov/m-0"), ("original_name", "m-old")])
def test_resume_rejects_changed_prompt_or_mapping(tmp_path, field, value):
    path, recs = _first_record(tmp_path)
    recs[0][field] = value
    _rewrite(path, recs)
    with pytest.raises(StaleRecords):
        run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD] * 4), CTX, sleep=lambda s: None)


@pytest.mark.parametrize("change", [{"temperature": 0.7}, {"temperature": None},
                                    {"max_tokens": 128}, {"token_param": "max_completion_tokens"}])
def test_resume_rejects_changed_decoding(tmp_path, change):
    _first_record(tmp_path)
    with pytest.raises(StaleRecords):
        run_model({**SPEC, **change}, SUBSET, tmp_path, FakeClient([GOOD] * 4), CTX, sleep=lambda s: None)


def test_decoding_check_skipped_for_error_only_records(tmp_path):
    run_model(SPEC, SUBSET, tmp_path, ScriptedClient([_bad("x"), _bad("y")]), CTX, limit=1, sleep=lambda s: None)
    client = FakeClient([GOOD] * 4)
    run_model({**SPEC, "temperature": 0.7}, SUBSET, tmp_path, client, CTX, sleep=lambda s: None)
    assert len(client.calls) == 4


def test_accept_spec_change_keeps_records_and_logs_incidents(tmp_path, capsys):
    _first_record(tmp_path)
    client = FakeClient([GOOD] * 2)
    run_model({**SPEC, "temperature": 0.7}, SUBSET, tmp_path, client, CTX, sleep=lambda s: None,
              accept_spec_change=True)
    assert len(client.calls) == 2
    incidents = [json.loads(l) for l in (tmp_path / "prov_m-1.incidents.jsonl").read_text().splitlines()]
    assert not (tmp_path / "incidents.jsonl").exists()
    assert len(incidents) == 2 and {i["field"] for i in incidents} == {"temperature"}
    assert all(i["kind"] == "accepted_spec_change" for i in incidents)
    assert "ACCEPTED" in capsys.readouterr().out


@pytest.mark.parametrize("models", [
    [{"original_name": "a", "model_id": "x"}, {"original_name": "b", "model_id": "x"}],
    [{"original_name": "a", "model_id": "p/x"}, {"original_name": "b", "model_id": "p_x"}],
    [{"original_name": "a", "model_id": "x"}, {"original_name": "a", "model_id": "y"}],
])
def test_load_specs_rejects_duplicates(tmp_path, models):
    from r2.run.harness import load_specs
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": models}))
    with pytest.raises(ValueError):
        load_specs(path)


def test_load_specs_current_file_is_unique():
    from r2.run.harness import load_specs
    assert len(load_specs()) == 21


# ---- I3 / M6: per-model lock, atomic torn-tail repair, byte-level line split ----
def test_second_lock_attempt_raises_run_locked(tmp_path):
    import fcntl

    from r2.run.harness import RunLocked, safe_name
    holder = open(tmp_path / f"{safe_name(SPEC['model_id'])}.lock", "a")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        client = FakeClient([GOOD] * 4)
        with pytest.raises(RunLocked):
            run_model(SPEC, SUBSET, tmp_path, client, CTX, sleep=lambda s: None)
        assert client.calls == [] and not results_path(SPEC, tmp_path).exists()
    finally:
        holder.close()
    run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD] * 4), CTX, sleep=lambda s: None)


def test_lock_released_when_run_model_raises(tmp_path):
    with pytest.raises(ModelMismatch):
        run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD], served="prov/other"), CTX, sleep=lambda s: None)
    run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD] * 4), CTX, sleep=lambda s: None)


def test_torn_tail_repair_keeps_earlier_bytes_identical(tmp_path):
    path, _ = _first_record(tmp_path)
    before = path.read_bytes()
    with open(path, "ab") as fh:
        fh.write(b'{"case_id": "CASE_00')
    run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD] * 2), CTX, sleep=lambda s: None)
    after = path.read_bytes()
    assert after.startswith(before) and b'{"case_id": "CASE_00{' not in after
    assert len(after.splitlines()) == 4


def test_torn_multibyte_tail_is_handled_as_torn(tmp_path):
    path, recs = _first_record(tmp_path)
    before = path.read_bytes()
    tail = json.dumps({**recs[0], "note": "• bullet"}, ensure_ascii=False).encode("utf-8")
    cut = tail.index("•".encode("utf-8")) + 1  # inside the 3-byte bullet
    with open(path, "ab") as fh:
        fh.write(tail[:cut])
    run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD] * 2), CTX, sleep=lambda s: None)
    assert path.read_bytes().startswith(before)
    assert all(json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())


# ---- I4: load_v2_cases checks; main run guarded by the freeze tag and the verified stimulus hash ----
from r2.run import harness  # noqa: E402

IDS500 = [f"CASE_{i:03d}" for i in range(1, 501)]


@pytest.mark.parametrize("kw,ids", [({"unresolved": ["CASE_001"]}, IDS500), ({}, IDS500[:499]),
                                    ({"identical": ["CASE_007"]}, IDS500)])
def test_load_v2_cases_raises_value_error(tmp_path, make_pairs, kw, ids):
    path = tmp_path / "pairs_v2.json"
    path.write_text(json.dumps(make_pairs(ids, **kw)))
    with pytest.raises(ValueError):
        harness.load_v2_cases(path)


def test_load_v2_cases_accepts_complete_set(tmp_path, make_pairs):
    import hashlib
    path = tmp_path / "pairs_v2.json"
    path.write_text(json.dumps(make_pairs(IDS500)))
    cases, sha = harness.load_v2_cases(path)
    assert len(cases) == 500 and sha == hashlib.sha256(path.read_bytes()).hexdigest()


def test_parse_args_out_defaults():
    from r2.paths import RUNS_V2
    assert harness.parse_args([]).out == str(RUNS_V2 / "main")
    assert harness.parse_args(["--pilot"]).out == str(RUNS_V2 / "pilot")
    assert harness.parse_args(["--pilot", "--out", "/x/y"]).out == "/x/y"
    assert harness.parse_args(["--accept-spec-change"]).accept_spec_change


@pytest.fixture
def no_side_effects(monkeypatch):
    monkeypatch.setattr(harness, "load_v2_cases", lambda *a, **k: ({}, "stim-sha"))
    monkeypatch.setattr(harness, "_clean_commit", lambda: "0" * 40)
    monkeypatch.setattr(harness, "load_specs", lambda *a, **k: [])
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)


def test_main_refuses_without_freeze_tag(no_side_effects, monkeypatch):
    monkeypatch.setattr(harness, "_tag_exists", lambda tag: False)
    with pytest.raises(SystemExit, match="r2-analysis-plan-frozen"):
        harness.main([])


@pytest.mark.parametrize("excluded", [None, {"excluded": []}, {"excluded": [], "pairs_v2_sha256": "other"}])
def test_main_refuses_unverified_stimulus(no_side_effects, monkeypatch, tmp_path, excluded):
    monkeypatch.setattr(harness, "_tag_exists", lambda tag: True)
    monkeypatch.setattr(harness, "VERIFY_DIR", tmp_path)
    if excluded is not None:
        (tmp_path / "excluded_pairs.json").write_text(json.dumps(excluded))
    with pytest.raises(SystemExit, match="pairs_v2_sha256"):
        harness.main([])


def test_main_runs_when_frozen_and_verified(no_side_effects, monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "_tag_exists", lambda tag: True)
    monkeypatch.setattr(harness, "VERIFY_DIR", tmp_path)
    (tmp_path / "excluded_pairs.json").write_text(json.dumps({"excluded": [], "pairs_v2_sha256": "stim-sha"}))
    harness.main([])


def test_pilot_skips_freeze_checks(no_side_effects, monkeypatch):
    def boom(tag):
        raise AssertionError("pilot must not check the freeze tag")
    monkeypatch.setattr(harness, "_tag_exists", boom)
    harness.main(["--pilot"])


def test_pilot_refuses_main_out_dir(no_side_effects):
    from r2.paths import RUNS_V2
    with pytest.raises(SystemExit, match="pilot"):
        harness.main(["--pilot", "--out", str(RUNS_V2 / "main")])


# ---- Step 4b N4: the pilot guard compares with samefile when both exist, else casefolded resolved paths ----
@pytest.mark.parametrize("make_main", [False, True], ids=["main-absent", "main-exists"])
@pytest.mark.parametrize("out", ["Main", "MAIN", "main/../main", "./main/"])
def test_pilot_guard_catches_case_and_spelling_variants(no_side_effects, monkeypatch, tmp_path, make_main, out):
    runs = tmp_path / "r2_runs"
    monkeypatch.setattr(harness, "RUNS_V2", runs)
    if make_main:
        (runs / "main").mkdir(parents=True)
    with pytest.raises(SystemExit, match="refusing a --pilot run into the main out dir"):
        harness.main(["--pilot", "--out", str(runs / out)])


def test_pilot_guard_follows_a_symlink_to_main(no_side_effects, monkeypatch, tmp_path):
    runs = tmp_path / "r2_runs"
    (runs / "main").mkdir(parents=True)
    (tmp_path / "elsewhere").symlink_to(runs / "main")
    monkeypatch.setattr(harness, "RUNS_V2", runs)
    with pytest.raises(SystemExit, match="refusing a --pilot run"):
        harness.main(["--pilot", "--out", str(tmp_path / "elsewhere")])


@pytest.mark.parametrize("make_dirs", [False, True])
def test_pilot_guard_allows_a_distinct_dir(no_side_effects, monkeypatch, tmp_path, make_dirs):
    runs = tmp_path / "r2_runs"
    if make_dirs:
        (runs / "main").mkdir(parents=True)
        (runs / "pilot").mkdir()
    monkeypatch.setattr(harness, "RUNS_V2", runs)
    harness.main(["--pilot", "--out", str(runs / "pilot")])
    harness.main(["--pilot", "--out", str(runs / "main-pilot")])


# ---- M1-M5, M17 ----
def _rate_limited(retry_after):
    exc = _fake_exc("RateLimitError")("slow down")
    exc.response = type("R", (), {"headers": {"retry-after": retry_after}})()
    return exc


@pytest.mark.parametrize("header,expected", [("30", 30.0), ("1", 2), ("not-a-number", 2)])
def test_create_honors_retry_after(header, expected):
    sleeps = []
    assert _create(_Seq([_rate_limited(header), "ok"]), {}, sleeps.append) == "ok"
    assert sleeps == [expected]


# ---- Step 4b N2: Retry-After capped at 900 s, non-finite or negative values fall back, every sleep logged ----
def _http_date(seconds_from_now):
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime
    return format_datetime(datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now), usegmt=True)


@pytest.mark.parametrize("header,expected", [
    ("899", 899.0), ("900", 900.0), ("901", 900), ("86400", 900), ("1e19", 900),
    ("inf", 2), ("Infinity", 2), ("-inf", 2), ("nan", 2), ("1e400", 2), ("-5", 2), ("0", 2),
])
def test_create_caps_retry_after(header, expected):
    sleeps = []
    assert _create(_Seq([_rate_limited(header), "ok"]), {}, sleeps.append) == "ok"
    assert sleeps == [expected]


def test_create_caps_retry_after_http_date():
    sleeps = []
    assert _create(_Seq([_rate_limited(_http_date(7200)), _rate_limited(_http_date(-60)), "ok"]), {},
                   sleeps.append) == "ok"
    assert sleeps == [900, 4]  # far future → cap; past date → backoff of the second try


def test_create_prints_a_line_before_each_sleep(capsys):
    def sleep(seconds):
        print(f"<slept {seconds}>")
    _create(_Seq([_rate_limited("86400"), _fake_exc("APIConnectionError")("reset"), "ok"]),
            {"model": "prov/m-1"}, sleep)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("RETRY prov/m-1 try 1/10: RateLimitError: slow down; sleeping 900 s")
    assert "Retry-After 86400" in lines[0] and "capped" in lines[0]
    assert lines[1] == "<slept 900>"
    assert lines[2].startswith("RETRY prov/m-1 try 2/10: APIConnectionError: reset; sleeping 4 s")
    assert lines[3] == "<slept 4>"


@pytest.mark.parametrize("served,ok", [("prov/m-1", True), ("other-org/M-1", True), ("m-1", True),
                                       ("prov/m-1-preview", False), ("prov/m-1-FP8", False), ("prov/m", False)])
def test_check_served_exact_after_normalization(served, ok):
    from r2.run.harness import check_served
    if ok:
        check_served(SPEC, served)
    else:
        with pytest.raises(ModelMismatch):
            check_served(SPEC, served)


def test_check_served_prefix_opt_in():
    from r2.run.harness import check_served
    spec = {**SPEC, "model_id": "gemini-2.5-flash-lite", "served_prefix": "models/gemini-2.5-flash-lite"}
    check_served(spec, "gemini-2.5-flash-lite")
    with pytest.raises(ModelMismatch):
        check_served(spec, "gemini-2.5-flash-lite-preview-06-17")


def test_model_mismatch_response_logged_to_incidents(tmp_path):
    with pytest.raises(ModelMismatch):
        run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD], served="prov/m-1-preview"), CTX, sleep=lambda s: None)
    (inc,) = [json.loads(l) for l in (tmp_path / "prov_m-1.incidents.jsonl").read_text().splitlines()]
    assert inc["kind"] == "model_mismatch" and inc["response"]["model"] == "prov/m-1-preview"
    assert inc["case_id"] == "CASE_001" and inc["request"]["model"] == "prov/m-1"
    assert not results_path(SPEC, tmp_path).read_text()


class _Proc:
    def __init__(self, rc, out):
        self.returncode, self.stdout = rc, out


@pytest.mark.parametrize("status,head,ok", [((0, ""), (0, "a" * 40 + "\n"), True),
                                            ((128, ""), (0, "a" * 40), False),
                                            ((0, ""), (128, ""), False),
                                            ((0, ""), (0, "HEAD"), False),
                                            ((0, " M r2/x.py"), (0, "a" * 40), False)])
def test_clean_commit_requires_git_success_and_sha(monkeypatch, status, head, ok):
    def fake_run(cmd, **kw):
        return _Proc(*status) if "status" in cmd else _Proc(*head)
    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    if ok:
        assert harness._clean_commit() == "a" * 40
    else:
        with pytest.raises(SystemExit):
            harness._clean_commit()


def test_parse_decision_keeps_raw_confidence():
    d = parse_decision('{"selected_tool": "A", "confidence": 85}')
    assert d["confidence"] is None and d["confidence_raw"] == 85
    d = parse_decision('{"selected_tool": "A", "confidence": "0.7"}')
    assert d["confidence"] == 0.7 and d["confidence_raw"] == "0.7"
    d = parse_decision('{"selected_tool": "A", "confidence": -0.1}')
    assert d["confidence"] is None and d["confidence_raw"] == -0.1
    d = parse_decision('{"selected_tool": "A", "confidence": 1}')
    assert d["confidence"] == 1.0
    d = parse_decision('{"selected_tool": "A"}')
    assert d["confidence"] is None and d["confidence_raw"] is None


def test_parse_decision_wraps_string_differences():
    d = parse_decision('{"selected_tool": "B", "differences_found": "B cites 2010, A cites 2019"}')
    assert d["differences_found"] == ["B cites 2010, A cites 2019"]
    assert parse_decision('{"selected_tool": "B", "differences_found": ["x"]}')["differences_found"] == ["x"]
    assert parse_decision('{"selected_tool": "B", "differences_found": null}')["differences_found"] == []


SPECS = [{"original_name": "a", "provider": "openai", "model_id": "a"},
         {"original_name": "b", "provider": "together", "model_id": "b"}]


def test_select_known_names_and_providers():
    assert harness._select(SPECS, "all") == SPECS
    assert harness._select(SPECS, "provider:together") == SPECS[1:]
    assert harness._select(SPECS, "a,b") == SPECS


@pytest.mark.parametrize("selector,name", [("provider:gemini", "gemini"), ("a,zz", "zz")])
def test_select_rejects_unknown(selector, name):
    with pytest.raises(SystemExit, match=name):
        harness._select(SPECS, selector)


# ---- Step 4b N5: per-model incident files, written under the model lock ----
def test_incidents_are_per_model_files(tmp_path):
    other = {**SPEC, "original_name": "n", "model_id": "prov/n-1"}
    with pytest.raises(ModelMismatch):
        run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD], served="prov/m-1-preview"), CTX, sleep=lambda s: None)
    with pytest.raises(ModelMismatch):
        run_model(other, SUBSET, tmp_path, FakeClient([GOOD], served="prov/n-2"), CTX, sleep=lambda s: None)
    from r2.run.harness import incidents_path
    assert incidents_path(SPEC, tmp_path).name == "prov_m-1.incidents.jsonl"
    m = [json.loads(l) for l in incidents_path(SPEC, tmp_path).read_text().splitlines()]
    n = [json.loads(l) for l in incidents_path(other, tmp_path).read_text().splitlines()]
    assert [i["model_id"] for i in m] == ["prov/m-1"] and [i["model_id"] for i in n] == ["prov/n-1"]
    assert not (tmp_path / "incidents.jsonl").exists()
    assert [p.name for p in iter_result_files(tmp_path)] == ["prov_m-1.jsonl", "prov_n-1.jsonl"]


def test_incidents_are_written_under_the_model_lock(tmp_path, monkeypatch):
    import fcntl

    from r2.run.harness import safe_name
    real, held = harness._log_incident, []

    def probe(out_dir, spec, incident):
        with open(tmp_path / f"{safe_name(spec['model_id'])}.lock", "a") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append(False)
            except BlockingIOError:
                held.append(True)
        real(out_dir, spec, incident)
    monkeypatch.setattr(harness, "_log_incident", probe)
    _first_record(tmp_path)
    run_model({**SPEC, "temperature": 0.7}, SUBSET, tmp_path, FakeClient([GOOD] * 2), CTX, sleep=lambda s: None,
              accept_spec_change=True)  # accepted_spec_change incidents, from _done
    with pytest.raises(ModelMismatch):  # model_mismatch incident, from execute_cell
        run_model({**SPEC, "original_name": "z", "model_id": "prov/z-1"}, SUBSET, tmp_path,
                  FakeClient([GOOD], served="prov/x"), CTX, sleep=lambda s: None)
    assert held == [True, True, True]


# ---- Step 4b: an exhausted transient error stops the model, the job continues; every stop is logged ----
def test_exhausted_transient_error_raises_model_unavailable():
    client = RaisingClient(_fake_exc("RateLimitError"))
    with pytest.raises(ModelUnavailable, match="RateLimitError persisted after 10 tries: boom"):
        execute_cell(client, SPEC, CASES["CASE_001"], "TS", CTX, sleep=lambda s: None)


@pytest.fixture
def two_models(monkeypatch, tmp_path):
    import functools
    a = {**SPEC, "original_name": "a", "model_id": "prov/a-1", "provider": "openai"}
    b = {**SPEC, "original_name": "b", "model_id": "prov/b-1", "provider": "together"}
    clients = {}
    monkeypatch.setattr(harness, "load_v2_cases", lambda *x, **k: (SUBSET, "stim-sha"))
    monkeypatch.setattr(harness, "_clean_commit", lambda: "0" * 40)
    monkeypatch.setattr(harness, "load_specs", lambda *x, **k: [a, b])
    monkeypatch.setattr("dotenv.load_dotenv", lambda *x, **k: None)
    monkeypatch.setattr(harness, "make_client", lambda provider: clients[provider])
    monkeypatch.setattr(harness, "run_model", functools.partial(harness.run_model, sleep=lambda s: None))
    return a, b, clients, tmp_path / "pilot"


def _incidents(spec, out):
    from r2.run.harness import incidents_path
    return [json.loads(l) for l in incidents_path(spec, out).read_text().splitlines()]


def _main_continues_after_stop(two_models, capsys, a_client):
    a, b, clients, out = two_models
    clients["openai"], clients["together"] = a_client, FakeClient([GOOD] * 4)
    stopped = harness.main(["--pilot", "--out", str(out)])
    captured = capsys.readouterr()
    assert stopped == ["a"]
    assert len(_recs(results_path(b, out))) == 4  # the next model ran to completion
    assert "FINISHED 2 model(s); stopped 1: ['a']" in captured.out
    return captured, out, a


def test_main_exhausted_transient_error_stops_model_and_continues(two_models, capsys):
    captured, out, a = _main_continues_after_stop(two_models, capsys, RaisingClient(_fake_exc("RateLimitError")))
    assert "STOPPED a: ModelUnavailable: prov/a-1: RateLimitError persisted after 10 tries: boom" in captured.out
    assert captured.out.count("RETRY prov/a-1 try ") == 9
    assert results_path(a, out).read_text() == ""  # no response, so no record: the cell runs again on resume
    (inc,) = _incidents(a, out)
    assert inc["kind"] == "stopped" and inc["error_type"] == "ModelUnavailable"
    assert (inc["case_id"], inc["order"]) == ("CASE_001", "TS") and "RateLimitError" in inc["error"]
    assert inc["harness_commit"] == "0" * 40 and inc["stimulus_sha256"] == "stim-sha"


def test_main_model_unavailable_stops_model_and_continues(two_models, capsys):
    captured, out, a = _main_continues_after_stop(two_models, capsys, RaisingClient(_fake_exc("NotFoundError")))
    assert "STOPPED a: ModelUnavailable: prov/a-1: NotFoundError: boom" in captured.out
    (inc,) = _incidents(a, out)
    assert inc["kind"] == "stopped" and "NotFoundError" in inc["error"]


def test_main_model_mismatch_stops_model_and_continues(two_models, capsys):
    captured, out, a = _main_continues_after_stop(two_models, capsys, FakeClient([GOOD], served="prov/a-2"))
    assert "STOPPED a: ModelMismatch: requested 'prov/a-1', provider served 'prov/a-2'" in captured.out
    assert [i["kind"] for i in _incidents(a, out)] == ["model_mismatch"]  # logged once, with the response


def test_main_unexpected_error_stops_model_and_continues(two_models, capsys):
    # e.g. an HTTP status the SDK maps to the generic APIStatusError (402), which no rule classifies
    captured, out, a = _main_continues_after_stop(two_models, capsys, RaisingClient(_fake_exc("APIStatusError")))
    assert "STOPPED a: unexpected APIStatusError: boom" in captured.out
    assert "Traceback" in captured.err
    (inc,) = _incidents(a, out)
    assert inc["kind"] == "stopped" and inc["error_type"] == "APIStatusError"


def test_main_stale_records_stop_model_and_continue(two_models, capsys):
    a, b, clients, out = two_models
    run_model(a, SUBSET, out, FakeClient([GOOD] * 2), {**CTX, "stimulus_sha256": "older"}, limit=1,
              sleep=lambda s: None)
    captured, out, a = _main_continues_after_stop(two_models, capsys, FakeClient([GOOD] * 4))
    assert "STOPPED a: StaleRecords: " in captured.out and "stale record for ('CASE_001', 'TS')" in captured.out
    (inc,) = _incidents(a, out)
    assert inc["kind"] == "stopped" and inc["error_type"] == "StaleRecords" and inc["case_id"] is None


def test_main_run_locked_stops_model_without_writing_its_files(two_models, capsys):
    import fcntl

    from r2.run.harness import incidents_path, safe_name
    a, b, clients, out = two_models
    out.mkdir(parents=True)
    holder = open(out / f"{safe_name(a['model_id'])}.lock", "a")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        captured, out, a = _main_continues_after_stop(two_models, capsys, FakeClient([GOOD] * 4))
    finally:
        holder.close()
    assert "STOPPED a: RunLocked: " in captured.out
    assert not incidents_path(a, out).exists() and not results_path(a, out).exists()


def test_breaker_stop_is_logged_as_an_incident(tmp_path):
    with pytest.raises(ModelUnavailable):
        run_model(SPEC, SUBSET, tmp_path, ScriptedClient([_bad("a"), _bad("b"), _bad("c")]), CTX,
                  sleep=lambda s: None)
    (inc,) = _incidents(SPEC, tmp_path)
    assert inc["kind"] == "stopped" and inc["error_type"] == "ModelUnavailable"
    assert "circuit breaker" in inc["error"] and (inc["case_id"], inc["order"]) == ("CASE_002", "TS")


# ---- minor 3: --limit 0 means zero pairs; omit --limit for all ----
def test_limit_zero_runs_no_cells(tmp_path):
    client = FakeClient([GOOD] * 4)
    run_model(SPEC, SUBSET, tmp_path, client, CTX, limit=0, sleep=lambda s: None)
    assert client.calls == []
    path = results_path(SPEC, tmp_path)
    assert not path.exists() or path.read_text() == ""


def test_limit_none_runs_all_cells(tmp_path):
    client = FakeClient([GOOD] * 4)
    run_model(SPEC, SUBSET, tmp_path, client, CTX, limit=None, sleep=lambda s: None)
    assert len(client.calls) == 4


def test_negative_limit_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="limit"):
        run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD] * 4), CTX, limit=-1, sleep=lambda s: None)
    with pytest.raises(SystemExit):
        harness.parse_args(["--limit", "-1"])
    assert harness.parse_args(["--limit", "0"]).limit == 0
    assert harness.parse_args([]).limit is None


# ---- minor 2: _clean_commit checks git's return codes and treats r2/, data/v2/ and tests/ as code ----
def _git(repo, *args):
    import subprocess
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
                    *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def tiny_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    for rel in ("r2/run/harness.py", "data/v2/pairs_v2.json", "tests/r2/test_harness.py", "docs/notes.md"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    monkeypatch.setattr(harness, "REPO", repo)
    return repo


@pytest.mark.parametrize("rel,dirty", [("tests/r2/test_harness.py", True), ("tests/r2/test_new.py", True),
                                       ("r2/run/harness.py", True), ("data/v2/pairs_v2.json", True),
                                       ("docs/notes.md", False)])
def test_clean_commit_dirty_check_covers_r2_data_v2_and_tests(tiny_repo, rel, dirty):
    import re
    (tiny_repo / rel).write_text("changed\n")
    if dirty:
        with pytest.raises(SystemExit, match=rel):
            harness._clean_commit()
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", harness._clean_commit())


def test_clean_commit_fails_outside_a_git_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "REPO", tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    with pytest.raises(SystemExit, match="git status failed"):
        harness._clean_commit()


# ---- minor 4: load_v2_cases raises explicit exceptions (no assert anywhere in r2/run/) ----
def test_no_assert_statements_in_r2_run():
    import ast
    from pathlib import Path
    run_dir = Path(harness.__file__).parent
    found = [f"{p.name}:{n.lineno}" for p in sorted(run_dir.glob("*.py")) if not p.name.startswith("._")
             for n in ast.walk(ast.parse(p.read_text())) if isinstance(n, ast.Assert)]
    assert found == []


@pytest.mark.parametrize("mutate", [
    lambda d: list(d["cases"]),                                  # top level is not an object
    lambda d: {k: v for k, v in d.items() if k != "cases"},      # no "cases"
    lambda d: {**d, "cases": {"CASE_001": {}}},                  # "cases" is not a list
    lambda d: {**d, "cases": [{k: v for k, v in c.items() if k != "case_id"} if i == 3 else c
                              for i, c in enumerate(d["cases"])]},
    lambda d: {**d, "cases": [{**c, "tools": {"T": c["tools"]["T"]}} if i == 3 else c
                              for i, c in enumerate(d["cases"])]},
], ids=["not-an-object", "no-cases", "cases-not-a-list", "case-without-id", "case-without-S"])
def test_load_v2_cases_rejects_malformed_payload(tmp_path, make_pairs, mutate):
    path = tmp_path / "pairs_v2.json"
    path.write_text(json.dumps(mutate(make_pairs(IDS500))))
    with pytest.raises(ValueError, match="pairs_v2.json"):
        harness.load_v2_cases(path)


# ---- minor 5: selected_tool forms; the raw value is kept in selected_raw ----
ACCEPTED = [("A", "A"), ("B", "B"), ("a", "A"), ("b", "B"), (" A ", "A"), ("B\n", "B"),
            ("Excerpt A", "A"), ("Excerpt B", "B"), ("Option A", "A"), ("Option B", "B"),
            ("Tool A", "A"), ("Tool B", "B"), ("Source A", "A"), ("Source B", "B"),
            ("excerpt a", "A"), ("OPTION B", "B"), ("tool b", "B"), ("SOURCE a", "A"),
            ("  Tool  A ", "A"), ("Excerpt\tB", "B")]
REJECTED = ["A or B", "Neither", "", "C", "Excerpt C", "A.", "Tool", "ExcerptA", "Guideline excerpt A",
            "A and B", "Tool A or Tool B", None, 1, ["A"]]


@pytest.mark.parametrize("raw,expected", ACCEPTED)
def test_parse_decision_accepts_selected_forms(raw, expected):
    d = parse_decision(json.dumps({"selected_tool": raw, "confidence": 0.5}))
    assert d["selected_tool"] == expected and d["selected_raw"] == raw


@pytest.mark.parametrize("raw", REJECTED)
def test_parse_decision_rejects_other_selected_values(raw):
    assert parse_decision(json.dumps({"selected_tool": raw, "confidence": 0.5})) is None


def test_selected_form_flows_into_the_record():
    client = FakeClient(['{"selected_tool": "Tool A", "confidence": 0.9}'])
    rec = execute_cell(client, SPEC, CASES["CASE_001"], "ST", CTX, sleep=lambda s: None)
    assert rec["status"] == "ok" and rec["selected_tool"] == "A" and rec["correct"] == 0  # ST: A is the sham
    assert rec["decision"]["selected_raw"] == "Tool A" and rec["decision"]["selected_tool"] == "A"
    client = FakeClient(['{"selected_tool": "Neither"}', '{"selected_tool": "A or B"}'])
    rec = execute_cell(client, SPEC, CASES["CASE_001"], "ST", CTX, sleep=lambda s: None)
    assert rec["status"] == "non_response" and rec["selected_tool"] is None and len(rec["attempts"]) == 2


# ---- minor 1: the mismatched served response, and any earlier attempt of the cell, are logged first ----
class ServedSeq(FakeClient):
    """Each call pops (content, served model)."""

    def create(self, **kw):
        self.calls.append(kw)
        text, served = self.replies.pop(0)
        return _Resp(text, served)


def test_model_mismatch_on_retry_logs_the_earlier_attempt_too(tmp_path):
    from r2.run.harness import incidents_path
    client = ServedSeq([("nope", "prov/m-1"), (GOOD, "prov/m-2")])
    with pytest.raises(ModelMismatch):
        run_model(SPEC, SUBSET, tmp_path, client, CTX, sleep=lambda s: None)
    (inc,) = [json.loads(l) for l in incidents_path(SPEC, tmp_path).read_text().splitlines()]
    assert inc["kind"] == "model_mismatch" and inc["response"]["model"] == "prov/m-2"
    assert inc["request"] == client.calls[1]
    assert [a["response"]["model"] for a in inc["attempts"]] == ["prov/m-1"]
    assert inc["attempts"][0]["response"]["choices"][0]["message"]["content"] == "nope"
    assert results_path(SPEC, tmp_path).read_text() == ""


# ---- minor 7: _done splits bytes before decoding; a tear inside any multibyte character is a torn tail ----
@pytest.mark.parametrize("char", ["é", "•", "🙂"], ids=["2-byte", "3-byte", "4-byte"])
def test_torn_tail_inside_multibyte_char_at_every_offset(tmp_path, char):
    encoded = char.encode("utf-8")
    for cut_in_char in range(1, len(encoded)):
        out = tmp_path / f"cut{cut_in_char}"
        path, recs = _first_record(out)
        before = path.read_bytes()
        tail = json.dumps({**recs[0], "order": "ST", "note": f"x{char}y"}, ensure_ascii=False).encode("utf-8")
        cut = tail.index(encoded) + cut_in_char
        with open(path, "ab") as fh:
            fh.write(tail[:cut])
        client = FakeClient([GOOD] * 2)
        run_model(SPEC, SUBSET, out, client, CTX, sleep=lambda s: None)
        after = path.read_bytes()
        assert after[:len(before)] == before and len(client.calls) == 2
        new = after[len(before):].decode("utf-8").splitlines()
        assert len(new) == 2 and all(json.loads(l)["case_id"] == "CASE_002" for l in new)


def test_invalid_utf8_in_a_middle_line_is_corrupt_not_a_decode_crash(tmp_path):
    path, recs = _first_record(tmp_path)
    good = path.read_bytes()
    path.write_bytes(b'{"case_id": "CASE_001", "note": "\xe2\x80"}\n' + good)
    with pytest.raises(CorruptResults, match="line 1"):
        run_model(SPEC, SUBSET, tmp_path, FakeClient([GOOD] * 4), CTX, sleep=lambda s: None)
