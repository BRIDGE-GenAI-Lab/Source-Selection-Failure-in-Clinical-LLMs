"""probe_models sends exactly the request the harness sends, through the same _create retry path."""
import pytest

from r2.run import harness, probe_models

CTX = {"harness_commit": "abc", "stimulus_sha256": "def"}
GOOD = '{"selected_tool": "B", "confidence": 0.8, "trust_rationale": "x", "differences_found": []}'
SPECS = [
    {"original_name": "chat", "provider": "together", "model_id": "org/chat-1", "temperature": 0,
     "max_tokens": 2048, "token_param": "max_tokens"},
    {"original_name": "oai-reasoner", "provider": "openai", "model_id": "r-1", "temperature": None,
     "max_tokens": 8192, "token_param": "max_completion_tokens"},
    {"original_name": "reasoner", "provider": "deepseek", "model_id": "r-2", "temperature": None,
     "max_tokens": 8192, "token_param": "max_tokens"},
]


class _Resp:
    def __init__(self, model):
        self._d = {"model": model, "system_fingerprint": "fp_1",
                   "choices": [{"message": {"content": GOOD}}], "usage": {}}

    def model_dump(self):
        return self._d


class Recorder:
    """Fake OpenAI client: records the kwargs of every call; raises the queued exceptions first."""

    def __init__(self, raises=(), served=None):
        self.calls, self.raises, self.served = [], list(raises), served
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.raises:
            raise self.raises.pop(0)
        return _Resp(self.served or kw["model"])


def _exc(name, text="x"):
    return type(name, (Exception,), {})(text)


@pytest.mark.parametrize("spec", SPECS, ids=[s["original_name"] for s in SPECS])
def test_probe_request_equals_the_harness_request(spec):
    harness_client, probe_client = Recorder(), Recorder()
    harness.execute_cell(harness_client, spec, probe_models.probe_case(), "TS", CTX, sleep=lambda s: None)
    row = probe_models.probe(spec, client=probe_client, sleep=lambda s: None)
    assert probe_client.calls == [harness_client.calls[0]]
    assert [m["role"] for m in probe_client.calls[0]["messages"]] == ["system", "user"]
    assert row == {"status": "available", "served": spec["model_id"], "fingerprint": "fp_1", "error": ""}


def test_probe_uses_the_harness_retry_policy():
    sleeps = []
    row = probe_models.probe(SPECS[0], client=Recorder([_exc("RateLimitError")] * 3), sleep=sleeps.append)
    assert row["status"] == "available" and sleeps == [2, 4, 8]
    sleeps.clear()
    row = probe_models.probe(SPECS[0], client=Recorder([_exc("RateLimitError", "quota")] * 10), sleep=sleeps.append)
    assert row["status"] == "error" and row["error"] == "RateLimitError: quota"
    assert sleeps == [2, 4, 8, 16, 32, 64, 128, 256, 300]


def test_probe_reports_served_identifier_and_errors():
    row = probe_models.probe(SPECS[0], client=Recorder(served="org/chat-1-0905"), sleep=lambda s: None)
    assert row["served"] == "org/chat-1-0905"
    row = probe_models.probe(SPECS[0], client=Recorder([_exc("NotFoundError", "no such model")]),
                             sleep=lambda s: None)
    assert row == {"status": "error", "served": "", "fingerprint": "", "error": "NotFoundError: no such model"}
