"""Rebuild the exact model inputs of the original 10,500 evaluations with the original
runner code, and verify them against the payload hashes logged at run time."""
import ast
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict

from r2.io import load_original_cases
from r2.original_runs import load_original
from r2.paths import AUDIT_DIR, LEGACY_RESULTS, ORIGINAL_CASES, REPO
from r2.profiles import PROFILES, profile_for

MESSAGE_VARIABLE = {
    "gemini_inline7": "full_prompt",
    "deepseek_chat_inline7": "user_message",
    "deepseek_reasoner_inline7": "user_message",
    "together_inline6": "user_message",
}
SCOPE_OPENAI_FC = ("excerpt_text sha256 (logged by src/runner.py); metadata, system, user, tools "
                   "reconstructed from original code, not logged")
SCOPE_INLINE = ("full tool payload dict sha256 (logged) + first 500 chars of the message (logged raw "
                "prompt); remainder reconstructed from the runner's own f-string")
_MODULES = {}
_FSTRINGS = {}


class FixedDraw:
    """Stand-in RNG: every original sanitizer assigns A=T when rng.random() < 0.5."""

    def __init__(self, mapping):
        self.value = 0.0 if mapping["A"] == "T" else 0.99

    def random(self):
        return self.value


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_json(obj):
    return _sha(json.dumps(obj, sort_keys=True))


def load_runner(rel_path):
    if rel_path not in _MODULES:
        for var in ("TOGETHER_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
                    "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
            os.environ.setdefault(var, "unused-for-reconstruction")
        path = REPO / rel_path
        spec = importlib.util.spec_from_file_location("orig_" + path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES[rel_path] = mod
    return _MODULES[rel_path]


def extract_fstring(rel_path, variable):
    """Compile the f-string assigned to `variable` in the runner source, verbatim."""
    key = (rel_path, variable)
    if key not in _FSTRINGS:
        tree = ast.parse((REPO / rel_path).read_text())
        found = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.JoinedStr)
                    and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets)):
                found = node.value
                break
        if found is None:
            raise LookupError(f"no f-string assigned to {variable} in {rel_path}")
        _FSTRINGS[key] = compile(ast.Expression(body=found), str(REPO / rel_path), "eval")
    return _FSTRINGS[key]


def reconstruct_inline(profile, case, mapping):
    rel = PROFILES[profile]
    mod = load_runner(rel)
    tool_a, tool_b, got = mod.sanitize_for_case(case, FixedDraw(mapping))
    assert got == mapping, (case["case_id"], got, mapping)
    try:
        system_prompt = mod.get_system_prompt("baseline")
    except TypeError:
        system_prompt = mod.get_system_prompt()
    names = {"case_prompt": mod.format_case_prompt(case), "tool_a": tool_a,
             "tool_b": tool_b, "system_prompt": system_prompt}
    text = eval(extract_fstring(rel, MESSAGE_VARIABLE[profile]), {"json": json}, names)
    return {
        "system": None if profile == "gemini_inline7" else system_prompt,
        "user": text,
        "payloads": {"A": tool_a, "B": tool_b},
        "sha256": {"A": _sha_json(tool_a), "B": _sha_json(tool_b)},
    }


def load_case_models():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.load_cases import load_cases
    return {c.case_id: c for c in load_cases(ORIGINAL_CASES)}


def reconstruct_openai_fc(case_model, mapping):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.prompts import format_user_prompt, get_system_prompt
    from src.runner import GUIDELINE_TOOLS
    from src.sanitize_tools import sanitize_for_case
    tool_a, tool_b, got = sanitize_for_case(case_model, FixedDraw(mapping))
    assert (got.A, got.B) == (mapping["A"], mapping["B"])
    return {
        "system": get_system_prompt("baseline"),
        "user": format_user_prompt(case_model),
        "tools": GUIDELINE_TOOLS,
        "payloads": {"A": json.loads(tool_a.model_dump_json()), "B": json.loads(tool_b.model_dump_json())},
        # src/runner.py logged sha256(excerpt_text) only
        "sha256": {"A": _sha(tool_a.excerpt_text), "B": _sha(tool_b.excerpt_text)},
    }


def _logged_prefix(run_dir, case_id):
    path = LEGACY_RESULTS / run_dir / "raw" / f"{case_id}_response.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("prompt")


def build_all(write=True):
    cases, _ = load_original_cases()
    case_models = None
    rows = []
    for r in load_original():
        model, cid, mapping = r["model_name"], r["case_id"], r["mapping"]
        profile = profile_for(model)
        if profile == "openai_fc":
            if case_models is None:
                case_models = load_case_models()
            rec = reconstruct_openai_fc(case_models[cid], mapping)
            prefix_ok = None  # Responses API logs do not echo the input
        else:
            rec = reconstruct_inline(profile, cases[cid], mapping)
            logged = _logged_prefix(r["run_dir"], cid)
            prefix_ok = None if logged is None else logged == rec["user"][:500] + "..."
        logged_hashes = r.get("tool_payload_hashes") or {}
        if not logged_hashes:
            status = "no_logged_hash"
        elif logged_hashes == rec["sha256"]:
            status = "verified"
        else:
            status = "MISMATCH"
        a = {k: v for k, v in rec["payloads"]["A"].items() if k != "tool_id"}
        b = {k: v for k, v in rec["payloads"]["B"].items() if k != "tool_id"}
        rows.append({
            "model": model, "profile": profile, "run_dir": r["run_dir"], "case_id": cid,
            "mapping": mapping, "sham_type": r.get("sham_trap_type"),
            "system": rec["system"], "user": rec["user"], "payloads": rec["payloads"],
            "tools": rec.get("tools"),
            "sha256_logged": logged_hashes, "sha256_reconstructed": rec["sha256"],
            "hash_status": status, "prompt_prefix_verified": prefix_ok,
            "identical_except_label": a == b,
            "verification_scope": SCOPE_OPENAI_FC if profile == "openai_fc" else SCOPE_INLINE,
        })
    if write:
        _write(rows)
    return rows


def _write(rows):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_DIR / "original_model_inputs.jsonl.gz", "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for row in rows:
                gz.write((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    per = defaultdict(Counter)
    scope = {row["profile"]: row["verification_scope"] for row in rows}
    for row in rows:
        c = per[(row["model"], row["profile"], row["run_dir"])]
        c["n"] += 1
        c[row["hash_status"]] += 1
        c["prefix_true"] += row["prompt_prefix_verified"] is True
        c["prefix_false"] += row["prompt_prefix_verified"] is False
        c["identical_pairs"] += row["identical_except_label"]
    with open(AUDIT_DIR / "original_inputs_verification.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "profile", "run_dir", "n", "verified", "no_logged_hash", "MISMATCH",
                    "prompt_prefix_verified", "prompt_prefix_failed", "identical_pairs",
                    "verification_scope"])
        for (model, profile, run_dir), c in sorted(per.items()):
            w.writerow([model, profile, run_dir, c["n"], c["verified"], c["no_logged_hash"],
                        c["MISMATCH"], c["prefix_true"], c["prefix_false"], c["identical_pairs"],
                            scope[profile]])


if __name__ == "__main__":
    out = build_all()
    print(Counter(r["hash_status"] for r in out))
