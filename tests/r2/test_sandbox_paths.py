"""R2_SANDBOX: every path the pipeline writes resolves inside the sandbox; read-only inputs stay real."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from r2 import paths

REPO = Path(paths.REPO)

PROBE = """
import json
from r2 import paths
from r2.analysis import figures, repeated, supp_tables
from r2.run import harness
out = {k: str(v) for k, v in paths.WRITE_PATHS.items()}
out.update({"repeated.OUT": str(repeated.OUT), "supp_tables.OUT_DIR": str(supp_tables.OUT_DIR),
            "supp_tables.ANALYSIS_REPEATED": str(supp_tables.ANALYSIS_REPEATED),
            "figures.OUT_DIR": str(figures.OUT_DIR), "harness.RUNS_V2": str(harness.RUNS_V2),
            "harness.DATA_V2": str(harness.DATA_V2), "harness.VERIFY_DIR": str(harness.VERIFY_DIR),
            "FREEZE_RECORD": str(paths.FREEZE_RECORD), "SANDBOX": str(paths.SANDBOX)})
read_only = {"ORIGINAL_CASES": str(paths.ORIGINAL_CASES), "AUDIT_DIR": str(paths.AUDIT_DIR),
             "LEGACY_ROOT": str(paths.LEGACY_ROOT), "ORIGINAL_RUNS_DIR": str(paths.ORIGINAL_RUNS_DIR),
             "REPO": str(paths.REPO)}
print(json.dumps({"write": out, "read_only": read_only}))
"""


def _probe(env_extra):
    env = {k: v for k, v in os.environ.items() if k not in ("R2_SANDBOX", "R2_RUNS_V2")}
    env.update(env_extra)
    res = subprocess.run([sys.executable, "-c", PROBE], cwd=REPO, env=env, capture_output=True, text=True)
    return res


def test_every_written_path_is_inside_the_sandbox(tmp_path):
    box = tmp_path / "box"
    res = _probe({"R2_SANDBOX": str(box)})
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    root = box.resolve()
    for name, p in got["write"].items():
        assert Path(p).resolve().is_relative_to(root), (name, p)
    assert got["read_only"]["ORIGINAL_CASES"] == str(paths.ORIGINAL_CASES)
    assert got["read_only"]["AUDIT_DIR"] == str(paths.AUDIT_DIR)
    assert got["read_only"]["LEGACY_ROOT"] == str(paths.LEGACY_ROOT)
    assert got["read_only"]["REPO"] == str(paths.REPO)


def test_without_sandbox_paths_are_the_real_ones():
    res = _probe({})
    assert res.returncode == 0, res.stderr
    w = json.loads(res.stdout)["write"]
    assert w["DATA_V2"] == str(REPO / "data" / "v2")
    assert w["VERIFY_DIR"] == str(REPO / "verification")
    assert w["ANALYSIS_REPEATED"] == str(REPO / "analysis_repeated")
    assert w["repeated.OUT"] == str(REPO / "analysis_repeated")
    assert w["TABLES_DIR"] == str(Path(paths.LEGACY_ROOT) / "npj_R2_WORKING" / "tables")
    assert w["figures.OUT_DIR"] == str(Path(paths.LEGACY_ROOT) / "npj_R2_WORKING" / "05_Figures")
    assert w["MODELS_FILE"] == str(REPO / "r2" / "run" / "models_v2.json")
    assert w["SANDBOX"] == "None"
    assert "FREEZE_REPO" not in w  # the freeze writes the repository itself, only outside a sandbox


@pytest.mark.parametrize("where", ["repo", "repo_sub", "legacy", "relative"])
def test_a_sandbox_inside_protected_roots_is_refused(where, tmp_path):
    target = {"repo": str(REPO), "repo_sub": str(REPO / "tmp_box"), "legacy": str(Path(paths.LEGACY_ROOT) / "box"),
              "relative": "relative/box"}[where]
    res = _probe({"R2_SANDBOX": target})
    assert res.returncode != 0
    assert "R2_SANDBOX" in res.stderr


def test_the_sandbox_cannot_contain_the_real_run_directory(tmp_path):
    runs = tmp_path / "runs"
    res = _probe({"R2_SANDBOX": str(tmp_path), "R2_RUNS_V2": str(runs)})
    assert res.returncode != 0 and "protected" in res.stderr
