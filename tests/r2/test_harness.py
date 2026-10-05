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


def test_create_gives_up_after_six():
    sleeps = []
    with pytest.raises(Exception, match="z"):
        _create(_Seq([_fake_exc("RateLimitError")("z")] * 6), {}, sleeps.append)
    assert len(sleeps) == 5


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


def test_breaker_trips_on_same_error_text_three_times_in_a_row(tmp_path):
    client = ScriptedClient([_bad("same"), GOOD, _bad("same"), GOOD, _bad("same"), GOOD])
    with pytest.raises(ModelUnavailable):
        run_model(SPEC, SUBSET3, tmp_path, client, CTX, sleep=lambda s: None)
    assert len(_recs(results_path(SPEC, tmp_path))) == 5


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
    incidents = [json.loads(l) for l in (tmp_path / "incidents.jsonl").read_text().splitlines()]
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
