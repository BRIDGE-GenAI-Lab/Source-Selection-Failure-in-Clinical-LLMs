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
GENERATOR_BOOKKEEPING = {"sham_type", "modification_log", "source_status"}

ROOT_CAUSE = {
    "hidden_only": "edit written only to fields no original runner serialized "
                   "(safety_guardrails / exclusions_or_cautions / applicability)",
    ("no_edit", "authority_mimicry"): "organization not in the 6-entry spoof table; transform made no change",
    ("no_edit", "subtle_inversion"): "no polarity keyword in the excerpt; transform made no change",
}


def parse_log_fields(where: str) -> list[str]:
    """Parse the 'where_in_tool' field from the log.

    Split on commas; for each piece, take the leading token before any space or "("
    and strip it; keep only tokens that are one of excerpt_text, safety_guardrails,
    exclusions_or_cautions, applicability, or start with "metadata."; return sorted
    and deduplicated. An empty string returns [].
    """
    if not where:
        return []

    valid_tokens = {"excerpt_text", "safety_guardrails", "exclusions_or_cautions", "applicability"}
    result = set()

    for piece in where.split(","):
        piece = piece.strip()
        if not piece:
            continue
        # Take the leading token before space or "("
        token = piece.split("(")[0].split()[0].strip()
        if token in valid_tokens or token.startswith("metadata."):
            result.add(token)

    return sorted(list(result))


def observed_fields(t, s) -> list[str]:
    """Get sorted names of every changed content field and metadata field.

    Returns sorted names of every changed content field (excerpt_text, safety_guardrails,
    exclusions_or_cautions, applicability) plus "metadata.<key>" for every metadata key
    whose value differs between T and S, EXCLUDING the generator bookkeeping keys
    sham_type, modification_log, source_status.
    """
    result = set()

    # Check content fields
    for field in VISIBLE_CONTENT + HIDDEN_CONTENT:
        if t.get(field) != s.get(field):
            result.add(field)

    # Check metadata fields, excluding bookkeeping keys
    if "metadata" in t and "metadata" in s:
        for key in set(t["metadata"].keys()) | set(s["metadata"].keys()):
            if key not in GENERATOR_BOOKKEEPING:
                if t["metadata"].get(key) != s["metadata"].get(key):
                    result.add(f"metadata.{key}")

    return sorted(list(result))


def claim_status(claimed, observed):
    """How the generator's logged claim relates to what actually changed between T and S."""
    claimed, observed = set(claimed), set(observed)
    if not claimed:
        return "no_claim"
    if claimed <= observed:
        return "realized"
    if not claimed & observed:
        return "not_realized"
    return "partially_realized"


def unclaimed_changes(claimed, observed):
    """Changes the log does not claim: none, curation-only (metadata keys no model saw), or content."""
    unclaimed = set(observed) - set(claimed)
    if not unclaimed:
        return "none"
    visible_meta = {f"metadata.{k}" for k in VISIBLE_META}
    if all(f.startswith("metadata.") and f not in visible_meta for f in unclaimed):
        return "curation_only"
    return "content"


def _require_fields(case, t, s):
    for label, tool in (("T", t), ("S", s)):
        for key in ("metadata", "excerpt_text"):
            if key not in tool:
                raise ValueError(f"{case.get('case_id')}: tool {label} has no '{key}' field; cannot classify the pair")


def classify_pair(case, log_row, shared_by):
    t, s = case["tools"]["T"], case["tools"]["S"]
    _require_fields(case, t, s)
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
        try:
            cause = ROOT_CAUSE[("no_edit", sham_type)]
        except KeyError:
            raise ValueError(f"{case['case_id']}: no_edit pair of type {sham_type!r} has no root-cause rule; "
                             "add one to ROOT_CAUSE") from None
    t_hash = hashlib.sha256(t["excerpt_text"].encode("utf-8")).hexdigest()[:16]

    # Reconciliation: compare logged fields with observed fields
    log_fields = parse_log_fields(summary.get("where_in_tool", ""))
    obs_fields = observed_fields(t, s)

    claimed_not_observed = sorted(set(log_fields) - set(obs_fields))
    observed_not_claimed = sorted(set(obs_fields) - set(log_fields))

    return {
        "case_id": case["case_id"], "sham_type": sham_type,
        "raw_status": raw, "visible_status": visible,
        "content_fields_changed": ";".join(content), "visible_meta_changed": ";".join(meta),
        "hidden_fields_changed": ";".join(hidden),
        "log_where": summary.get("where_in_tool", ""), "log_what": summary.get("what_changed", ""),
        "root_cause": cause, "t_excerpt_hash": t_hash, "t_excerpt_shared_by": shared_by[t_hash],
        "log_fields_claimed": ";".join(log_fields),
        "observed_fields_changed": ";".join(obs_fields),
        "claimed_not_observed": ";".join(claimed_not_observed),
        "observed_not_claimed": ";".join(observed_not_claimed),
        "claim_status": claim_status(log_fields, obs_fields),
        "unclaimed_changes": unclaimed_changes(log_fields, obs_fields),
    }


def _by_type(df, col):
    return {t: {k: int(n) for k, n in sorted(Counter(g[col]).items())} for t, g in df.groupby("sham_type")}


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

        # Pairs whose log claims a field that did not change, by sham_type
        claimed_not_obs_by_type = {}
        for sham_type in df.sham_type.unique():
            subset = df[df.sham_type == sham_type]
            count = (subset.claimed_not_observed != "").sum()
            if count > 0:
                claimed_not_obs_by_type[sham_type] = int(count)

        summary = {
            "visible_status": dict(Counter(df.visible_status)),
            "raw_status": dict(Counter(df.raw_status)),
            "distinct_authentic_excerpts": int(df.t_excerpt_hash.nunique()),
            "claim_status_by_type": _by_type(df, "claim_status"),
            "unclaimed_changes_by_type": _by_type(df, "unclaimed_changes"),
            "claimed_not_observed_by_type": claimed_not_obs_by_type,
            "case_022": df.set_index("case_id").loc["CASE_022"].to_dict(),
        }
        (AUDIT_DIR / "pair_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return df


if __name__ == "__main__":
    print(run().groupby(["sham_type", "raw_status"]).size())
