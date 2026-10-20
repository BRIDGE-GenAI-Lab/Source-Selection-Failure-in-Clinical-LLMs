"""Single source of file locations.

Every input and output lives in this repository, except the original run directories (the legacy
safe-guideline-tooling-eval/ tree: results/<run>/ with the raw responses, and results_opensource/), which only the
steps that read raw run files need. For those steps set R2_LEGACY_ROOT to the directory that holds
safe-guideline-tooling-eval/; without it they stop with LegacyRootUnset.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_LEGACY = os.environ.get("R2_LEGACY_ROOT", "").strip()
LEGACY_ROOT = Path(_LEGACY) if _LEGACY else None
LEGACY_EVAL = LEGACY_ROOT / "safe-guideline-tooling-eval" if LEGACY_ROOT else None
LEGACY_RESULTS = LEGACY_EVAL / "results" if LEGACY_EVAL else None

# Exact bytes the models saw (as run). The published copy differs only in PDF path strings.
ORIGINAL_CASES = REPO / "data" / "original" / "500cases_with_real_tools.as_run.json"
PUBLISHED_CASES = REPO / "data" / "500cases_with_real_tools.json"
ORIGINAL_CASES_SHA256 = "58651ef5425278f1fbc661367d6a704482a8b5ddfb8850160c1c1a0e4d194436"
ORIGINAL_RUNS_DIR = REPO / "data" / "original_runs"
AUDIT_DIR = REPO / "audit"
VERIFY_DIR = REPO / "verification"
MODELS_FILE = REPO / "r2" / "run" / "models_v2.json"  # read only for the reasoning classification of Figure 4


class LegacyRootUnset(RuntimeError):
    """A step needs the original run directories and R2_LEGACY_ROOT is not set."""


def legacy(path):
    """`path` (LEGACY_EVAL or LEGACY_RESULTS as the caller's module holds it), or LegacyRootUnset when it is None."""
    if path is None:
        raise LegacyRootUnset("R2_LEGACY_ROOT is not set: this step reads the original run directories "
                              "(safe-guideline-tooling-eval/results*/), which are not in this repository; "
                              "set R2_LEGACY_ROOT to the directory that holds safe-guideline-tooling-eval/")
    return path
