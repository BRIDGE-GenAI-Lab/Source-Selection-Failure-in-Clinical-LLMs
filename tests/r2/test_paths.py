import copy
import hashlib
import json
import os

from r2.paths import ORIGINAL_CASES, ORIGINAL_CASES_SHA256, PUBLISHED_CASES


def test_repo_case_file_is_the_file_the_models_saw():
    digest = hashlib.sha256(ORIGINAL_CASES.read_bytes()).hexdigest()
    assert digest == ORIGINAL_CASES_SHA256


def _strip(data):
    data = copy.deepcopy(data)
    for c in data["cases_with_real_tools"]["cases"]:
        for t in c["tools"].values():
            t["metadata"].pop("source_pdf_path", None)
    for row in data["tool_audit_log"]:
        row.pop("source_pdf_used", None)
    return data


def test_published_copy_differs_only_in_pdf_paths():
    a = json.loads(ORIGINAL_CASES.read_text())
    p = json.loads(PUBLISHED_CASES.read_text())
    assert _strip(a) == _strip(p)
    ac = a["cases_with_real_tools"]["cases"]
    pc = p["cases_with_real_tools"]["cases"]
    assert len(ac) == len(pc)
    for x, y in zip(ac, pc):
        assert x["case_id"] == y["case_id"]
        for k in ("T", "S"):
            xm, ym = x["tools"][k]["metadata"], y["tools"][k]["metadata"]
            assert os.path.basename(xm["source_pdf_path"]) == ym["source_pdf_path"]
    for x, y in zip(a["tool_audit_log"], p["tool_audit_log"]):
        assert os.path.basename(x["source_pdf_used"]) == y["source_pdf_used"]
