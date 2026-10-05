import json

import pytest

from r2.io import load_original_cases
from r2.run.harness import (ModelMismatch, execute_cell, iter_result_files, parse_decision,
                            results_path, run_model)
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
