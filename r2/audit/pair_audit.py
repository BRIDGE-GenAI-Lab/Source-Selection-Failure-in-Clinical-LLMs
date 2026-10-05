"""Classify every authentic-sham pair by what the models could see, and reconcile it with the
modification log written by scripts/generate_tools.py."""
import hashlib
import json
from collections import Counter

import pandas as pd

from r2.io import load_original_cases
from r2.paths import AUDIT_DIR

VISIBLE_CONTENT = ("excerpt_text",)
HIDDEN_CONTENT = ("safety_guardrails", "exclusions_or_cautions", "applicability")
VISIBLE_META = ("organization", "guideline_title", "version_label", "publication_year")

ROOT_CAUSE = {
    "hidden_only": "edit written only to fields no original runner serialized "
                   "(safety_guardrails / exclusions_or_cautions / applicability)",
    ("no_edit", "authority_mimicry"): "organization not in the 6-entry spoof table; transform made no change",
    ("no_edit", "subtle_inversion"): "no polarity keyword in the excerpt; transform made no change",
}


def classify_pair(case, log_row, shared_by):
    t, s = case["tools"]["T"], case["tools"]["S"]
    content = [f for f in VISIBLE_CONTENT + HIDDEN_CONTENT if t.get(f) != s.get(f)]
    meta = [f for f in VISIBLE_META if t["metadata"].get(f) != s["metadata"].get(f)]
    hidden = [f for f in HIDDEN_CONTENT if f in content]
    if "excerpt_text" in content or meta:
        raw = "visible"
    elif hidden:
        raw = "hidden_only"
    else:
        raw = "no_edit"
    if raw != "visible":
        visible = "identical"
    elif "excerpt_text" in content:
        visible = "excerpt_differs"
    else:
        visible = "metadata_only"
    sham_type = case["eval"]["sham_trap_type"]
    summary = (log_row or {}).get("tool_S_poison_summary") or {}
    cause = ""
    if raw == "hidden_only":
        cause = ROOT_CAUSE["hidden_only"]
    elif raw == "no_edit":
        cause = ROOT_CAUSE.get(("no_edit", sham_type), "no change found")
    t_hash = hashlib.sha256(t["excerpt_text"].encode("utf-8")).hexdigest()[:16]
    return {
        "case_id": case["case_id"], "sham_type": sham_type,
        "raw_status": raw, "visible_status": visible,
        "content_fields_changed": ";".join(content), "visible_meta_changed": ";".join(meta),
        "hidden_fields_changed": ";".join(hidden),
        "log_where": summary.get("where_in_tool", ""), "log_what": summary.get("what_changed", ""),
        "root_cause": cause, "t_excerpt_hash": t_hash, "t_excerpt_shared_by": shared_by[t_hash],
    }


def run(write=True):
    cases, log = load_original_cases()
    shared = Counter(hashlib.sha256(c["tools"]["T"]["excerpt_text"].encode("utf-8")).hexdigest()[:16]
                     for c in cases.values())
    df = pd.DataFrame([classify_pair(cases[cid], log.get(cid), shared) for cid in sorted(cases)])
    if write:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(AUDIT_DIR / "pair_audit.csv", index=False)
        by_type = (df.groupby(["sham_type", "raw_status"]).size().unstack(fill_value=0)
                     .join(df.groupby(["sham_type", "visible_status"]).size().unstack(fill_value=0)))
        by_type["pairs"] = df.groupby("sham_type").size()
        by_type.to_csv(AUDIT_DIR / "pair_audit_by_type.csv")
        summary = {
            "visible_status": dict(Counter(df.visible_status)),
            "raw_status": dict(Counter(df.raw_status)),
            "distinct_authentic_excerpts": int(df.t_excerpt_hash.nunique()),
            "case_022": df.set_index("case_id").loc["CASE_022"].to_dict(),
        }
        (AUDIT_DIR / "pair_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return df


if __name__ == "__main__":
    print(run().groupby(["sham_type", "raw_status"]).size())
