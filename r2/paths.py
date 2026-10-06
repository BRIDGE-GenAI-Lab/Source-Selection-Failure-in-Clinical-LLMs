"""Single source of file locations for the R2 integrity revision.

Sandbox mode: with R2_SANDBOX=<dir> set (an absolute directory outside the repository, the legacy root and
the real run directory), every path the pipeline WRITES or keeps state in resolves inside <dir> (DATA_V2,
VERIFY_DIR, RUNS_V2, ANALYSIS_REPEATED, R2_WORKING with its tables and figures, MODELS_FILE, FREEZE_REPO,
PUBLIC_RECORD). Read-only inputs (code, original data, audit, legacy results) stay real. WRITE_PATHS lists
the written locations; r2/pipeline.py and r2/rehearse.py use them. Without R2_SANDBOX every path is the
real one, as before.
"""
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

# The real locations (also when sandboxed: the sandbox guard protects them).
REAL_DATA_V2 = REPO / "data" / "v2"
REAL_VERIFY_DIR = REPO / "verification"
REAL_RUNS_V2 = Path(os.environ.get("R2_RUNS_V2", str(LEGACY_ROOT / "r2_runs")))
REAL_ANALYSIS_REPEATED = REPO / "analysis_repeated"
REAL_R2_WORKING = LEGACY_ROOT / "npj_R2_WORKING"  # the manuscript, supplement and response build folder
REAL_MODELS_FILE = REPO / "r2" / "run" / "models_v2.json"


def _key(path):
    return tuple(part.casefold() for part in Path(path).expanduser().resolve().parts)


def _sandbox():
    """The sandbox root from R2_SANDBOX, or None. Refused (ValueError) when relative, or when it is, contains or
    lies inside the repository, the legacy root or the real run directory (compared casefolded: the volumes are
    case-insensitive). r2.analysis.synthetic.assert_sandbox adds the samefile and other-checkout checks."""
    raw = os.environ.get("R2_SANDBOX", "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        raise ValueError(f"R2_SANDBOX must be an absolute path, got {raw!r}")
    p = p.resolve()
    for root in (REPO, LEGACY_ROOT, REAL_RUNS_V2):
        a, b = _key(p), _key(root)
        if a[:len(b)] == b or b[:len(a)] == a:
            raise ValueError(f"R2_SANDBOX={p} is, contains or lies inside the protected {root}")
    return p


SANDBOX = _sandbox()
if SANDBOX is None:
    DATA_V2 = REAL_DATA_V2
    VERIFY_DIR = REAL_VERIFY_DIR
    RUNS_V2 = REAL_RUNS_V2
    ANALYSIS_REPEATED = REAL_ANALYSIS_REPEATED
    R2_WORKING = REAL_R2_WORKING
    MODELS_FILE = REAL_MODELS_FILE
    FREEZE_REPO = REPO  # where the freeze tag lives
    PUBLIC_RECORD = REPO / "docs" / "r2" / "PUBLIC_RECORD.json"
else:
    DATA_V2 = SANDBOX / "data" / "v2"
    VERIFY_DIR = SANDBOX / "verification"
    RUNS_V2 = SANDBOX / "r2_runs"
    ANALYSIS_REPEATED = SANDBOX / "analysis_repeated"
    R2_WORKING = SANDBOX / "npj_R2_WORKING"
    MODELS_FILE = SANDBOX / "models_v2.json"
    FREEZE_REPO = SANDBOX / "freeze_repo"  # a throwaway git repository holding copies of the frozen inputs
    PUBLIC_RECORD = SANDBOX / "PUBLIC_RECORD.json"
TABLES_DIR = R2_WORKING / "tables"
FIGURES_DIR = R2_WORKING / "05_Figures"
FREEZE_RECORD = FREEZE_REPO / "docs" / "r2" / "FREEZE_RECORD.md"

# Every location the pipeline writes or keeps state in. FREEZE_REPO is REPO itself outside a sandbox (the
# freeze commits and tags the repository), so it is listed only when sandboxed.
WRITE_PATHS = {"DATA_V2": DATA_V2, "VERIFY_DIR": VERIFY_DIR, "RUNS_V2": RUNS_V2,
               "ANALYSIS_REPEATED": ANALYSIS_REPEATED, "R2_WORKING": R2_WORKING, "TABLES_DIR": TABLES_DIR,
               "FIGURES_DIR": FIGURES_DIR, "MODELS_FILE": MODELS_FILE, "PUBLIC_RECORD": PUBLIC_RECORD}
if SANDBOX is not None:
    WRITE_PATHS.update({"FREEZE_REPO": FREEZE_REPO, "FREEZE_RECORD": FREEZE_RECORD})
