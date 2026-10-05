import json

from r2.paths import ORIGINAL_CASES


def load_original_cases():
    """Return (cases_by_id, audit_log_by_id) from the case file the original runs used."""
    data = json.loads(ORIGINAL_CASES.read_text())
    cases = {c["case_id"]: c for c in data["cases_with_real_tools"]["cases"]}
    log = {row["case_id"]: row for row in data["tool_audit_log"]}
    return cases, log
