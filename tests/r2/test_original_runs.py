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


def test_committed_manifest_matches_committed_files():
    import hashlib
    import json

    from r2.paths import ORIGINAL_RUNS_DIR
    manifest = json.loads((ORIGINAL_RUNS_DIR / "manifest.json").read_text())
    for run in manifest["runs"]:
        data = (ORIGINAL_RUNS_DIR / run["file"]).read_bytes()
        assert run["exported_sha256"] == hashlib.sha256(data).hexdigest()
