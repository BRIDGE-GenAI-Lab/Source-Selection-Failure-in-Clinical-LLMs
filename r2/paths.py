"""Single source of file locations for the R2 integrity revision."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path(os.environ.get("R2_LEGACY_ROOT", "/Volumes/Lexar/LLM_Tool_use"))
LEGACY_EVAL = LEGACY_ROOT / "safe-guideline-tooling-eval"
LEGACY_RESULTS = LEGACY_EVAL / "results"

# Exact bytes the models saw (as run). The published copy differs only in PDF path strings.
ORIGINAL_CASES = REPO / "data" / "original" / "500cases_with_real_tools.as_run.json"
PUBLISHED_CASES = REPO / "data" / "500cases_with_real_tools.json"
ORIGINAL_CASES_SHA256 = "58651ef5425278f1fbc661367d6a704482a8b5ddfb8850160c1c1a0e4d194436"
ORIGINAL_RUNS_DIR = REPO / "data" / "original_runs"
AUDIT_DIR = REPO / "audit"
DATA_V2 = REPO / "data" / "v2"
VERIFY_DIR = REPO / "verification"
RUNS_V2 = Path(os.environ.get("R2_RUNS_V2", str(LEGACY_ROOT / "r2_runs")))
