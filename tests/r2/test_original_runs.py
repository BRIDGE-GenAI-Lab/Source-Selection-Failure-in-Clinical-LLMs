import pytest

from r2.original_runs import load_original, sham_position
from r2.profiles import profile_for


def test_original_export_reproduces_published_aggregates():
    recs = load_original()
    assert len(recs) == 10500
    assert len({r["model_name"] for r in recs}) == 21
    assert len({(r["model_name"], r["case_id"]) for r in recs}) == 10500
    acc = sum(r["selected_tool_correct"] is True for r in recs) / len(recs)
    assert abs(acc - 0.5938) < 0.0005
    assert sum(sham_position(r) == "A" for r in recs) == 5291


def test_profiles_cover_every_model():
    models = {r["model_name"] for r in load_original()}
    counts = {}
    for m in models:
        counts[profile_for(m)] = counts.get(profile_for(m), 0) + 1
    assert counts == {"openai_fc": 4, "gemini_inline7": 1, "deepseek_chat_inline7": 1,
                      "deepseek_reasoner_inline7": 1, "together_inline6": 14}


# ---- export_original_runs (Task DM item 1) ----
def _export_to(tmp_path, monkeypatch):
    import r2.export_original_runs as ex
    monkeypatch.setattr(ex, "ORIGINAL_RUNS_DIR", tmp_path / "runs")
    return ex


def test_export_manifest_hashes_match_exported_bytes(tmp_path, monkeypatch):
    import hashlib
    import json
    ex = _export_to(tmp_path, monkeypatch)
    ex.main()
    manifest = json.loads((tmp_path / "runs" / "manifest.json").read_text())
    assert len(manifest["runs"]) == 21
    for run in manifest["runs"]:
        data = (tmp_path / "runs" / run["file"]).read_bytes()
        assert run["exported_sha256"] == hashlib.sha256(data).hexdigest()


def test_export_key_mismatch_raises_before_any_write(tmp_path, monkeypatch):
    import types
    ex = _export_to(tmp_path, monkeypatch)
    real = ex._legacy_loader()
    fake = types.SimpleNamespace(**{k: getattr(real, k) for k in dir(real) if not k.startswith("__")})
    fake.load_canonical = lambda: ([], None)  # reference disagrees with the export
    fake.__doc__ = real.__doc__
    monkeypatch.setattr(ex, "_legacy_loader", lambda: fake)
    with pytest.raises(RuntimeError, match="differs from load_canonical"):
        ex.main()
    out = tmp_path / "runs"
    assert not out.exists() or not any(out.iterdir())


def test_committed_manifest_matches_committed_files():
    import hashlib
    import json

    from r2.paths import ORIGINAL_RUNS_DIR
    manifest = json.loads((ORIGINAL_RUNS_DIR / "manifest.json").read_text())
    for run in manifest["runs"]:
        data = (ORIGINAL_RUNS_DIR / run["file"]).read_bytes()
        assert run["exported_sha256"] == hashlib.sha256(data).hexdigest()
