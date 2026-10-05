"""M7: one fake run through run_model -> check -> export_flat -> repeated_frame, both orders."""
import csv
import hashlib
import json

import pytest

from r2.analysis import repeated
from r2.io import load_original_cases
from r2.run import check_complete
from r2.run.harness import ORDERS, run_model
from r2.stimuli.build_v2 import edit_record
from r2.stimuli.render import render_messages, visible_content

SPEC = {"original_name": "e2e-model", "provider": "together", "model_id": "prov/e2e-1",
        "temperature": 0, "max_tokens": 64, "token_param": "max_tokens"}


class AuthenticChooser:
    """Replies with the label of the authentic tool (T) for whichever case and order it is sent."""

    def __init__(self, cases):
        self.answer = {}
        for c in cases.values():
            for order in ORDERS:
                msgs = render_messages(c, order)
                self.answer[msgs["user"]] = next(k for k, v in msgs["labels"].items() if v == "T")
        self.calls = []
        self.chat = self.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        label = self.answer[kw["messages"][1]["content"]]
        text = json.dumps({"selected_tool": label, "confidence": 0.9, "trust_rationale": "x",
                           "differences_found": ["y"]})
        resp = {"model": kw["model"], "choices": [{"message": {"content": text}}], "usage": {}}
        return type("Resp", (), {"model_dump": lambda self: resp})()


@pytest.fixture
def world(tmp_path, monkeypatch):
    originals, _ = load_original_cases()
    ids = [cid for cid in sorted(originals)
           if visible_content(originals[cid]["tools"]["T"]) != visible_content(originals[cid]["tools"]["S"])][:2]
    cases = {}
    for cid in ids:
        c = dict(originals[cid])
        c["edit_record"] = edit_record(c["tools"]["T"], c["tools"]["S"], "original_transform")
        cases[cid] = c
    data, verify, out = tmp_path / "data", tmp_path / "verification", tmp_path / "runs"
    data.mkdir()
    verify.mkdir()
    raw = json.dumps({"derived_from_sha256": "test", "n_pairs": 2, "unresolved": [], "cases": list(cases.values())},
                     ensure_ascii=False).encode("utf-8")
    (data / "pairs_v2.json").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    (verify / "excluded_pairs.json").write_text(json.dumps({"rule": "x", "excluded": [], "reasons": {},
                                                            "pairs_v2_sha256": sha}))
    (verify / "flags.json").write_text("{}")
    for mod in (check_complete, repeated):
        monkeypatch.setattr(mod, "load_specs", lambda *a, **k: [SPEC])
        monkeypatch.setattr(mod, "DATA_V2", data)
    monkeypatch.setattr(repeated, "VERIFY_DIR", verify)
    return cases, sha, out, data


def test_fake_run_end_to_end(world):
    cases, sha, out, data = world
    client = AuthenticChooser(cases)
    run_model(SPEC, cases, out, client, {"harness_commit": "0" * 40, "stimulus_sha256": sha}, sleep=lambda s: None)
    assert len(client.calls) == 4

    (row,) = check_complete.check(out, [SPEC], sorted(cases))
    for k in ("missing", "duplicates", "unexpected", "non_response", "errors", "superseded_errors",
              "invalid_records"):
        assert row[k] == 0, k
    assert row["cells"] == 4 and row["ok"] == 4 and row["harness_commit"] == "0" * 40

    flat = list(csv.DictReader(open(check_complete.export_flat(out))))
    assert len(flat) == 4
    for r in flat:
        assert (r["label_A"], r["label_B"]) == (("T", "S") if r["order"] == "TS" else ("S", "T"))
        assert r["selected_tool"] == ("A" if r["order"] == "TS" else "B") and r["correct"] == "1"

    df = repeated.repeated_frame(out).set_index(["case_id", "order"]).sort_index()
    assert sorted(df.index) == sorted((c, o) for c in cases for o in ORDERS)
    for (cid, order), r in df.iterrows():
        assert r.status == "ok"
        assert bool(r.sham_first) is (order == "ST")
        assert bool(r.selected_first) is (order == "TS")
        assert r.correct == 1
