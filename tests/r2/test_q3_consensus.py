import pandas as pd

from r2.verify import evaluable as ev


def test_evaluable_pairs_and_discordant_q3():
    p = ev.load_pairs()
    assert len(p) == 333
    assert int(ev.evaluable_mask(p).sum()) == 25
    d = ev.discordant_q3(p)
    assert len(d) == 19
    assert list(d.index) == sorted(d.index)
    assert (d[f"AG_{ev.Q3}"] != d[f"MO_{ev.Q3}"]).all()


def _consensus_csv(path, answers, date="2026-10-19"):
    pd.DataFrame({"case_id": [f"CASE_{i:03d}" for i in range(len(answers))], "sham_type": "prompt_injection",
                  f"AG_{ev.Q3}": "P", f"MO_{ev.Q3}": "Y", f"consensus_{ev.Q3}": answers, "consensus_note": "",
                  "adjudicated_on": date}).to_csv(path, index=False)


def test_data_files_consensus_role(tmp_path, monkeypatch):
    from r2 import data_files_doc as dfd
    _consensus_csv(tmp_path / "c.csv", ["N", "N", "Y"])
    monkeypatch.setattr(dfd, "REPO", tmp_path)
    role = dfd._q3_consensus_role("c.csv")
    assert "(adjudicated 2026-10-19) for the 3 clinically evaluable pairs" in role
    assert "(Y 1, P 0, N 2)" in role
