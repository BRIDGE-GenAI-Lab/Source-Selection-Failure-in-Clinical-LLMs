"""One pipeline from "the physicians returned their forms" to "manuscript, supplement and response rebuilt".

    python -m r2.pipeline <command> [options]        (the order, and what to do on failure: docs/r2/RUNBOOK.md)

  import-edits AG_r1.xlsx [--author A.G.] [--date YYYY-MM-DD]
      the 23 physician-written edits -> data/v2/manual_edits.json -> pairs_v2.json rebuilt strictly (all 500
      pairs, re-audited by build_v2) -> round-2 forms for the 23 rebuilt pairs.
  consensus AG_r1.xlsx MO_r1.xlsx AG_r2.xlsx MO_r2.xlsx [--consensus FILE]
      imports the four answered workbooks, merges round 1 and round 2, scores agreement (kappa.csv,
      adjudication_needed.csv) and, once verification/consensus.csv resolves every disagreement, applies the
      exclusion and flag rules (excluded_pairs.json, flags.json, final_review.csv).
  probe --yes | pilot --yes [--models M] | main --yes [--models M] [--retry-persistent] [--accept-spec-change]
                                            [--accept-unpublished-freeze]
      thin wrappers over r2.run.probe_models and r2.run.harness; each prints the exact command it runs and
      spends money only with --yes. main also needs the freeze tag and the public timestamp of the freeze
      (record-public); --accept-unpublished-freeze is the only override and writes an incident line.
  pilot-summary [DIR]
      read-only: per model, the pilot's valid replies, non-responses, API errors, truncations, served
      identifiers and median tokens (the facts PILOT_REVIEW.md records); DIR defaults to r2_runs/pilot.
  freeze [--dry-run | --yes]
      wraps python -m r2.freeze (local commit and tag; never pushes).
  record-public URL [--date YYYY-MM-DD]
      after the authors approved and made the freeze public: records the pushed tag's URL, an OSF registration
      or a Zenodo DOI, the day, and the tag commit in docs/r2/FREEZE_RECORD.md (commit that file).
  record [--release-tag TAG] [--zenodo-doi DOI]
      writes the release facts the documents cite (docs/r2/PUBLIC_RECORD.json) after the authors' release.
  scan-legacy-keys [ROOT]
      read-only: the files under ROOT (default the legacy tree) that hold a plaintext API key, and the provider
      of each key; never any part of a key.
  post-results [--no-render]
      completeness gate -> repeated analyses -> order check (freeze and public timestamp before the first
      main-run call) -> supplement tables -> Table 1 -> Figures 2-4 -> numbers.json -> manuscript, supplement,
      response -> QA scripts (any failure stops) -> PENDING_INDEX.md -> remaining markers by class.
  markers
      the remaining pending markers of the three documents, NUMERIC (must be filled) or PROSE (author text).
  sandbox-init DIR
      a new sandbox (for R2_SANDBOX=DIR) holding copies of the real state; r2.rehearse uses it.

Every command is idempotent and refuses (exit status 2, "REFUSED: ...") when an input is missing. Under
R2_SANDBOX every write goes to the sandbox (r2.paths), every write location is checked first, and no command
calls a real API.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from r2 import paths

PILOT_PAIRS = 10  # the pilot sends the first 10 pairs (sorted case_id) to every model, both orders
PENDING_RE = re.compile(r"⟦PENDING: (.*?)⟧", re.S)
DOCS = {"manuscript": "02_Manuscript.docx", "supplement": "04_Supplement.docx",
        "response": "01_Response_to_Reviewers.docx"}
PY = sys.executable
DOI_RE = re.compile(r"10\.\d{4,9}/\S+")


class Refused(RuntimeError):
    """An input is missing or a precondition fails; nothing (more) was written."""


class NeedsHumans(RuntimeError):
    """The step stopped where people must act (e.g. adjudicate disagreements); rerun it afterwards."""


# ---------------------------------------------------------------- guards
def check_write_paths():
    """Under R2_SANDBOX: the sandbox passes r2.analysis.synthetic.assert_sandbox and every write location of
    r2.paths lies inside it. Outside a sandbox: nothing to check."""
    if paths.SANDBOX is None:
        return None
    from r2.analysis.synthetic import _within, assert_sandbox
    root = assert_sandbox(paths.SANDBOX)
    outside = {k: str(v) for k, v in paths.WRITE_PATHS.items()
               if not _within(Path(v).expanduser().resolve(), root)}
    if outside:
        raise Refused(f"R2_SANDBOX={root}: write locations outside the sandbox: {outside}")
    return root


def _need(*files):
    missing = [str(f) for f in files if not Path(f).exists()]
    if missing:
        raise Refused("missing input: " + ", ".join(missing))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _show(cmd):
    print("command: " + shlex.join(cmd), flush=True)


def _pairs():
    p = paths.DATA_V2 / "pairs_v2.json"
    _need(p)
    return json.loads(p.read_text(encoding="utf-8"))


def _complete(pairs):
    return len(pairs.get("cases", [])) == 500 and not pairs.get("unresolved")


# ---------------------------------------------------------------- sandbox-init
STATE_NOTE = """SANDBOX: copies of the real state plus simulated physicians and a simulated model API (r2/rehearse.py).
Nothing in this directory is data. Never copy anything from here into the repository or npj_R2_WORKING.
"""


def _ignore(_dir, names):
    return [n for n in names if n.startswith("._") or n in ("__pycache__", ".DS_Store")]


def sandbox_init(root, n_models=None):
    """Create the sandbox `root` (empty or absent) and copy the real state in: data/v2/*, verification/round1/*,
    verification/leakage_flags.json, r2/run/models_v2.json (optionally cut to `n_models`, always keeping the
    S5 model) and npj_R2_WORKING (without PDFs over 5 MB under 05_Figures). Reads only the real paths."""
    from r2.analysis.repeated import S5_MODEL
    from r2.analysis.synthetic import assert_sandbox
    root = assert_sandbox(root)
    if root.exists() and any(root.iterdir()):
        raise Refused(f"{root} is not empty; a sandbox starts empty")
    root.mkdir(parents=True, exist_ok=True)
    (root / "SANDBOX_README.txt").write_text(STATE_NOTE)
    shutil.copytree(paths.REAL_DATA_V2, root / "data" / "v2", ignore=_ignore)
    shutil.copytree(paths.REAL_VERIFY_DIR / "round1", root / "verification" / "round1", ignore=_ignore)
    shutil.copy2(paths.REAL_VERIFY_DIR / "leakage_flags.json", root / "verification" / "leakage_flags.json")
    models = json.loads(paths.REAL_MODELS_FILE.read_text())
    if n_models:  # a smaller rehearsal: the S5 model, one reasoning model, then the others in file order
        ms = models["models"]
        first = [m for m in ms if m["original_name"] == S5_MODEL] + [next(m for m in ms if m["reasoning"])]
        keep = (first + [m for m in ms if m not in first])[:max(n_models, 2)]
        order = {m["original_name"]: i for i, m in enumerate(ms)}
        models["models"] = sorted(keep, key=lambda m: order[m["original_name"]])
    (root / "models_v2.json").write_text(json.dumps(models, indent=1) + "\n")

    def ignore_docs(d, names):
        out = set(_ignore(d, names))
        if Path(d).name == "05_Figures":
            out |= {n for n in names if n.endswith(".pdf") and (Path(d) / n).stat().st_size > 5_000_000}
        return list(out)
    shutil.copytree(paths.REAL_R2_WORKING, root / "npj_R2_WORKING", ignore=ignore_docs)
    for d in ("r2_runs", "analysis_repeated"):
        (root / d).mkdir(exist_ok=True)
    return {"root": str(root), "models": [m["original_name"] for m in models["models"]]}


# ---------------------------------------------------------------- scan-legacy-keys
# The key patterns of the global constraints (plus OpenAI's sk-proj- form, which their sk- pattern misses).
# Written with character classes so that this file never matches the secrets grep itself.
KEY_PATTERNS = (("Together", re.compile(r"tgp[_]v1[_][A-Za-z0-9_-]{20,}")),
                ("Groq", re.compile(r"gsk[_][A-Za-z0-9]{20,}")),
                ("Google Gemini", re.compile(r"AIz[a][0-9A-Za-z_-]{20,}")),
                ("OpenAI", re.compile(r"s[k]-proj-[A-Za-z0-9_-]{20,}")),
                ("sk-", re.compile(r"(?<![A-Za-z0-9])s[k]-[A-Za-z0-9]{20,}")))
SCAN_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "site-packages", ".pytest_cache"}
BINARY_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".docx", ".xlsx", ".pptx", ".zip", ".gz", ".tar",
                   ".pyc", ".ashx", ".parquet", ".pkl", ".npz", ".npy", ".mp4", ".mov", ".tif", ".tiff", ".svgz"}


def _sk_provider(key, text):
    """An sk- key is DeepSeek's (sk- + 32 hex) or OpenAI's; the file's text decides when the form does not."""
    low = text.lower()
    if re.fullmatch(r"s[k]-[0-9a-f]{32}", key) or ("deepseek" in low and "openai.com" not in low):
        return "DeepSeek"
    return "OpenAI" if "openai" in low else "OpenAI or DeepSeek"


def scan_legacy_keys(root=None, quiet=False):
    """Plaintext API keys under `root` (default the legacy tree): the files and the provider of each key, never
    any part of a key. Skips ._* files, git repositories below root (check those with the secrets git grep),
    virtual environments and binary files. Returns {files: {path: [providers]}, providers: {name: {files,
    distinct_keys}}, skipped_repositories: [...], files_read: n}."""
    root = Path(root or paths.LEGACY_ROOT)
    _need(root)
    files, seen, repos, n_read = {}, {}, [], 0
    for d, dirs, names in os.walk(root):
        here = Path(d)
        keep = []
        for name in sorted(dirs):
            sub_dir = here / name
            if name in SCAN_SKIP_DIRS or name.startswith("._"):
                continue
            if (sub_dir / ".git").exists():
                repos.append(str(sub_dir))
                continue
            keep.append(name)
        dirs[:] = keep
        for name in sorted(names):
            path = here / name
            if name.startswith("._") or path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:8192]:
                continue
            n_read += 1
            text = data.decode("utf-8", "replace")
            found = set()
            for provider, rx in KEY_PATTERNS:
                for m in rx.finditer(text):
                    name_ = _sk_provider(m.group(0), text) if provider == "sk-" else provider
                    found.add(name_)
                    seen.setdefault(name_, {"files": set(), "keys": set()})
                    seen[name_]["files"].add(str(path))
                    seen[name_]["keys"].add(hashlib.sha256(m.group(0).encode()).hexdigest())  # counted, never shown
            if found:
                files[str(path)] = sorted(found)
    providers = {k: {"files": len(v["files"]), "distinct_keys": len(v["keys"])} for k, v in sorted(seen.items())}
    if not quiet:
        print(f"plaintext API keys under {root} ({n_read} text files read; skipped ._* files, virtual environments, "
              f"binary files and {len(repos)} git repositories):")
        for path, provs in files.items():
            print(f"  {path}: {', '.join(provs)}")
        for name, v in providers.items():
            print(f"  {name}: {v['distinct_keys']} distinct key(s) in {v['files']} file(s)")
        for r in repos:
            print(f"  skipped git repository (run the secrets git grep there): {r}")
        if providers:
            print("Revoke EVERY key of each provider above in that provider's dashboard (not only the keys these "
                  "files hold), create new keys, and put them only in ~/npj_r2/r2_repo/.env "
                  "(docs/r2/RUNBOOK.md, section 1).")
    return {"files": files, "providers": providers, "skipped_repositories": repos, "files_read": n_read}


# ---------------------------------------------------------------- import-edits
def _workbook_edits(xlsx):
    """(case_id -> {field, find, replace}) from the Author_edits sheet, as build_packets reads it."""
    from openpyxl import load_workbook
    from r2.verify import build_packets as BP
    wb = load_workbook(xlsx, data_only=True)
    if BP.AUTHOR_EDITS not in wb.sheetnames:
        raise Refused(f"{xlsx}: no '{BP.AUTHOR_EDITS}' sheet")
    rows = [[BP._cell_text(c.value) for c in row] for row in wb[BP.AUTHOR_EDITS].iter_rows()]
    start = next((i for i, r in enumerate(rows) if r and r[0] == "case_id"), None)
    if start is None:
        raise Refused(f"{xlsx}: {BP.AUTHOR_EDITS} header row not found")
    col = {name: rows[start].index(name) for name in BP.AUTHOR_COLS if name in rows[start]}
    out = {}
    for r in rows[start + 1:]:
        cid = r[col["case_id"]].strip() if col["case_id"] < len(r) else ""
        if cid:
            out[cid] = {k: (r[col[k]] if col[k] < len(r) else "") for k in ("field", "find", "replace")}
            out[cid]["field"] = out[cid]["field"].strip()
    return out


def _round2_ready(pairs_sha):
    from r2.verify import build_packets as BP
    d = paths.VERIFY_DIR / "round2"
    man = d / BP.MANIFEST
    if not man.exists():
        return False
    ok = json.loads(man.read_text(encoding="utf-8")).get("pairs_v2_sha256") == pairs_sha
    have = all((d / f"review_{r}.xlsx").exists() for r, _ in BP.REVIEWERS)
    if have and not ok:
        raise Refused(f"{d} holds round-2 workbooks for another pairs_v2.json; if they were never sent to the "
                      "physicians, delete that folder and rerun")
    return ok and have


def import_edits(xlsx, author="A.G.", date=None):
    from r2.stimuli.build_v2 import build_v2, write_outputs
    from r2.verify import build_packets as BP
    check_write_paths()
    xlsx = Path(xlsx)
    _need(xlsx, paths.VERIFY_DIR / "round1" / BP.MANIFEST)
    pairs = _pairs()
    queue = json.loads((paths.VERIFY_DIR / "round1" / BP.MANIFEST).read_text(encoding="utf-8")).get("unresolved") or []
    manual_path = paths.DATA_V2 / "manual_edits.json"
    result = {"workbook": str(xlsx)}
    if pairs.get("unresolved"):
        try:
            edits = BP.import_author_edits(xlsx, author, date or _today())
        except ValueError as exc:  # a rejected edit: nothing was written; send the row back to the physician
            raise Refused(f"author edit rejected: {exc}") from exc
        result["edits_imported"] = len(edits)
    else:  # already imported: the workbook must agree with manual_edits.json (idempotent rerun)
        _need(manual_path)
        stored = json.loads(manual_path.read_text(encoding="utf-8"))
        given = _workbook_edits(xlsx)
        differ = sorted(c for c in set(stored) | set(given)
                        if {k: stored.get(c, {}).get(k) for k in ("field", "find", "replace")} != given.get(c))
        if differ:
            raise Refused(f"pairs_v2.json already holds physician edits, and {xlsx.name} differs from "
                          f"data/v2/manual_edits.json for {differ[:10]}; change manual_edits.json deliberately "
                          "(and rebuild) rather than re-importing")
        result["edits_imported"] = 0
        result["already_imported"] = len(stored)
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else {}
    missing = sorted(set(queue) - set(manual))
    if missing:
        raise Refused(f"no physician edit for {len(missing)} queued pairs: {missing[:10]}")
    try:
        cases, unresolved = build_v2(manual, strict=True)  # re-audits every pair: identical renders raise
    except ValueError as exc:
        raise Refused(f"the rebuild of pairs_v2.json failed (nothing was written): {exc}") from exc
    write_outputs(cases, unresolved)
    pairs = _pairs()
    sha = _sha(paths.DATA_V2 / "pairs_v2.json")
    from r2.stimuli.render import visible_content
    result.update({"pairs": len(pairs["cases"]), "unresolved": len(pairs["unresolved"]), "pairs_v2_sha256": sha,
                   "identical_visible": sum(visible_content(c["tools"]["T"]) == visible_content(c["tools"]["S"])
                                            for c in pairs["cases"])})
    if _round2_ready(sha):
        result["round2"] = "forms already written for this pairs_v2.json"
    else:
        BP.write_round_forms("round2", queue)
        BP._export_round("round2", with_author_edits=False, force=False)
        result["round2"] = f"forms for {len(queue)} pairs in {paths.VERIFY_DIR / 'round2'}"
    return result


# ---------------------------------------------------------------- consensus
def _mark_synthetic(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("synthetic") is not True:
        path.write_text(json.dumps({**d, "synthetic": True}, indent=2), encoding="utf-8")


def consensus(ag1, mo1, ag2, mo2, consensus_file=None):
    from r2.verify import build_packets as BP
    from r2.verify import score_review as SR
    check_write_paths()
    books = {("round1", "AG"): Path(ag1), ("round1", "MO"): Path(mo1), ("round2", "AG"): Path(ag2),
             ("round2", "MO"): Path(mo2)}
    _need(*books.values())
    if not _complete(_pairs()):
        raise Refused("pairs_v2.json is not complete (500 pairs, none unresolved): run import-edits first")
    for (label, reviewer), book in books.items():
        _need(paths.VERIFY_DIR / label / BP.MANIFEST)
        try:
            BP.import_review_xlsx(book, paths.VERIFY_DIR / label / f"review_{reviewer}.csv")
        except ValueError as exc:
            raise Refused(f"{book}: review form rejected: {exc}") from exc
    BP.merge_rounds(["round1", "round2"])
    a, b = paths.VERIFY_DIR / "review_AG.csv", paths.VERIFY_DIR / "review_MO.csv"
    _, disagreements = SR.score(a, b)
    cons = paths.VERIFY_DIR / "consensus.csv"
    if consensus_file:
        src = Path(consensus_file)
        _need(src)
        if src.resolve() != cons.resolve():
            shutil.copyfile(src, cons)
    result = {"pairs_reviewed": len(SR._load(a)), "disagreements": len(disagreements),
              "kappa": str(paths.VERIFY_DIR / "kappa.csv"),
              "adjudication_needed": str(paths.VERIFY_DIR / "adjudication_needed.csv")}
    if not cons.exists():
        if disagreements:
            raise NeedsHumans(f"{len(disagreements)} disagreements are listed in "
                              f"{paths.VERIFY_DIR / 'adjudication_needed.csv'}; the two physicians agree a "
                              f"consensus answer for each and write {cons} (columns case_id,item,consensus), "
                              "then rerun this command")
        cons.write_text("case_id,item,consensus\n", encoding="utf-8")
    excluded, flags = SR.apply_consensus(a, b, cons)
    if paths.SANDBOX is not None:
        for name in ("excluded_pairs.json", "flags.json"):
            _mark_synthetic(paths.VERIFY_DIR / name)
    result.update({"excluded": len(excluded), "flagged": len(flags),
                   "pairs_v2_sha256": json.loads((paths.VERIFY_DIR / "excluded_pairs.json").read_text())
                   ["pairs_v2_sha256"]})
    return result


# ---------------------------------------------------------------- paid steps (probe, pilot, main)
def _spend_guard(yes, what):
    if not yes:
        raise Refused(f"{what} spends money on API calls: rerun with --yes once the authors have approved the "
                      "budget (docs/r2/RUNBOOK.md)")


def probe(yes=False, client_factory=None, sleep=None):
    check_write_paths()
    cmd = ["python", "-m", "r2.run.probe_models"]
    _show(cmd)
    _spend_guard(yes, "the probe")
    if client_factory is None:
        if paths.SANDBOX is not None:
            raise Refused("R2_SANDBOX is set: a sandbox never calls a real API (use r2.rehearse)")
        return subprocess.run([PY, "-m", "r2.run.probe_models"], cwd=paths.REPO).returncode
    from r2.run import probe_models
    probe_models.main(client_factory=client_factory, **({"sleep": sleep} if sleep else {}))
    return 0


def harness_argv(kind, models="all", retry_persistent=False, accept_spec_change=False):
    argv = ["--models", models]
    if kind == "pilot":
        argv = ["--pilot", "--limit", str(PILOT_PAIRS)] + argv
    if retry_persistent:
        argv.append("--retry-persistent")
    if accept_spec_change:
        argv.append("--accept-spec-change")
    return argv


def run_harness(kind, yes=False, client_factory=None, sleep=None, accept_unpublished_freeze=False, **flags):
    """kind 'pilot' or 'main'. Returns the harness exit status (1: a model stopped or cells need a resume).
    The main run also needs the freeze tag and a public timestamp of the freeze in FREEZE_RECORD.md (record-public);
    the only override, accept_unpublished_freeze, writes an incident line."""
    check_write_paths()
    argv = harness_argv(kind, **flags)
    _show(["python", "-m", "r2.run.harness", *argv])
    _spend_guard(yes, f"the {kind} run")
    if kind == "main":
        from r2.freeze import TAG
        _need(paths.VERIFY_DIR / "excluded_pairs.json")
        if not _tag_commit(TAG):
            raise Refused(f"the analysis plan is not frozen: no tag {TAG} in {paths.FREEZE_REPO} "
                          "(python -m r2.pipeline freeze)")
    if client_factory is None and paths.SANDBOX is not None:
        raise Refused("R2_SANDBOX is set: a sandbox never calls a real API (use r2.rehearse)")
    if kind == "main":
        pub = public_timestamp()
        if pub is not None:
            print(f"public timestamp of the freeze: {pub['kind']} {pub['url']} (public since {pub['since']})")
        elif not accept_unpublished_freeze:
            raise Refused(f"{paths.FREEZE_RECORD} records no public timestamp of the freeze. Pushing the tag needs "
                          "the authors' approval; once the freeze is public (the pushed tag, an OSF registration or a "
                          "Zenodo DOI), run python -m r2.pipeline record-public <url> and rerun. The only override is "
                          "--accept-unpublished-freeze, which writes an incident line")
        else:
            where = _pipeline_incident({"kind": "unpublished_freeze", "freeze_record": str(paths.FREEZE_RECORD),
                                        "argv": argv, "note": "main run started before the freeze was public"})
            print(f"ACCEPTED: main run without a public timestamp of the freeze (--accept-unpublished-freeze); "
                  f"incident written to {where}")
    if client_factory is None:
        return subprocess.run([PY, "-m", "r2.run.harness", *argv], cwd=paths.REPO).returncode
    from r2.run import harness
    return harness.cli(argv, client_factory=client_factory, **({"sleep": sleep} if sleep else {}))


def _median(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    return None if not xs else (xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2)


def pilot_summary(out_dir=None, write=True):
    """Summary of a pilot run, one row per model of the current specs: valid replies, non-responses, API errors,
    truncations, served identifiers, median tokens, first and last call and the parse rate. Reads the pilot
    records only; with write=True it also writes analysis_repeated/pilot_summary.csv (the pilot keys of
    numbers.json, for the authors' pilot sentence; the pilot is never analysed)."""
    from r2.run.harness import final_records, load_specs, results_path
    out_dir = Path(out_dir or paths.RUNS_V2 / "pilot")
    _need(out_dir)
    rows = []
    for s in load_specs(paths.MODELS_FILE):
        path = results_path(s, out_dir)
        final = final_records(path) if path.exists() else []
        last = [(r.get("attempts") or [{}])[-1].get("response") or {} for r in final]
        usage = [x.get("usage") or {} for x in last]
        finish = [((x.get("choices") or [{}])[0] or {}).get("finish_reason") for x in last]
        rows.append({"model": s["original_name"], "cells": len(final),
                     "ok": sum(r.get("status") == "ok" for r in final),
                     "non_response": sum(r.get("status") == "non_response" for r in final),
                     "api_errors": sum("error" in r for r in final),
                     "truncated": sum(f == "length" for f in finish),
                     "served": ";".join(sorted({r["model_served"] for r in final if r.get("model_served")})),
                     "median_prompt_tokens": _median([u.get("prompt_tokens") for u in usage]),
                     "median_completion_tokens": _median([u.get("completion_tokens") for u in usage]),
                     "first_call": min((r["finished_at"] for r in final if r.get("finished_at")), default=""),
                     "last_call": max((r["finished_at"] for r in final if r.get("finished_at")), default=""),
                     "parse_rate": (sum(r.get("status") == "ok" for r in final) / len(final)) if final else ""})
    if write and rows:
        dest = paths.ANALYSIS_REPEATED / "pilot_summary.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        extra = {"synthetic": True} if paths.SANDBOX is not None else {}
        with open(dest, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) + list(extra))
            w.writeheader()
            w.writerows({**r, **extra} for r in rows)
    head = [k for k in (rows[0] if rows else []) if k not in ("first_call", "last_call", "parse_rate")]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for r in rows:
        rate = f"{100 * r['ok'] / r['cells']:.0f}%" if r["cells"] else "no records"
        print("| " + " | ".join(str(r[k]) for k in head) + f" | {rate}")
    return rows


# ---------------------------------------------------------------- freeze
FREEZE_INPUTS = {  # path in the freeze repository -> source (frozen hashes, r2.freeze.HASHED, plus the plan)
    "docs/r2/ANALYSIS_PLAN.md": lambda: paths.REPO / "docs" / "r2" / "ANALYSIS_PLAN.md",
    "data/v2/pairs_v2.json": lambda: paths.DATA_V2 / "pairs_v2.json",
    "verification/excluded_pairs.json": lambda: paths.VERIFY_DIR / "excluded_pairs.json",
    "r2/run/models_v2.json": lambda: paths.MODELS_FILE,
    "r2/stimuli/render.py": lambda: paths.REPO / "r2" / "stimuli" / "render.py",
    "r2/run/harness.py": lambda: paths.REPO / "r2" / "run" / "harness.py",
}
GIT_ID = ["-c", "user.name=R2 sandbox", "-c", "user.email=sandbox@rehearsal.invalid", "-c", "commit.gpgsign=false",
          "-c", "tag.gpgsign=false"]


def _git(repo, *args, check=False):
    """git in `repo`. Under R2_SANDBOX through r2.sandbox_git (no inherited GIT_* variables, no hooks, and the
    repository must lie inside the sandbox); a failure or a refusal becomes Refused."""
    if paths.SANDBOX is None:
        res = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
        if check and res.returncode:
            raise Refused(f"git {' '.join(args)} failed in {repo}: {res.stderr.strip()}")
        return res
    from r2 import sandbox_git
    try:
        return sandbox_git.run(repo, *args, sandbox=paths.SANDBOX, check=check)
    except sandbox_git.SandboxGitError as exc:
        raise Refused(str(exc)) from exc


def _tag_commit(tag, repo=None):
    """The commit the tag points to in `repo` (default FREEZE_REPO), or None when the tag does not exist."""
    repo = Path(repo or paths.FREEZE_REPO)
    if not (repo / ".git").exists():
        return None
    if _git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode:
        return None
    return _git(repo, "rev-parse", f"{tag}^{{commit}}", check=True).stdout.strip()


def _sandbox_freeze_repo():
    """Under R2_SANDBOX: a throwaway git repository (paths.FREEZE_REPO) holding copies of the frozen inputs, so
    that r2.freeze runs unchanged without touching the real repository. Synced and committed until frozen."""
    from r2 import sandbox_git
    from r2.analysis.synthetic import assert_sandbox
    repo = assert_sandbox(paths.FREEZE_REPO)
    _need(*(src() for src in FREEZE_INPUTS.values()))
    try:
        sandbox_git.init_repo(repo, paths.SANDBOX)
    except sandbox_git.SandboxGitError as exc:
        raise Refused(f"sandbox freeze repository: {exc}") from exc
    for rel, src in FREEZE_INPUTS.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src(), repo / rel)
    if _git(repo, "status", "--porcelain", check=True).stdout.strip():
        _git(repo, "add", "-A", check=True)
        _git(repo, *GIT_ID, "commit", "-q", "-m", "sandbox: inputs to freeze (simulated)", check=True)
    return repo


def _guarded_freeze_git(F):
    """r2.freeze._git for the sandbox freeze: every call through r2.sandbox_git (r2/freeze.py is unchanged)."""
    from r2 import sandbox_git

    def _git_in_sandbox(repo, *args, check=True):
        try:
            res = sandbox_git.run(repo, *args, sandbox=paths.SANDBOX)
        except sandbox_git.SandboxGitError as exc:
            raise F.FreezeRefused(str(exc)) from exc
        if check and res.returncode != 0:
            raise F.FreezeRefused(f"git {' '.join(args)} failed: {res.stderr.strip()}")
        return res
    return _git_in_sandbox


def freeze(dry_run=False, yes=False):
    from r2 import freeze as F
    check_write_paths()
    commit = _tag_commit(F.TAG)
    if commit:
        print(f"already frozen: tag {F.TAG} -> {commit}")
        return {"frozen": True, "commit": commit, "already": True}
    cmd = ["python", "-m", "r2.freeze"] + (["--dry-run"] if dry_run else [])
    _show(cmd)
    if not dry_run and not yes:
        raise Refused("the freeze commits and tags the repository and is final: run --dry-run first, then rerun "
                      "with --yes once the authors approve")
    if paths.SANDBOX is None:
        rc = subprocess.run([PY, "-m", "r2.freeze", *(["--dry-run"] if dry_run else [])], cwd=paths.REPO).returncode
        if rc:
            raise Refused(f"python -m r2.freeze exited {rc} (its REFUSED line says why)")
        return {"frozen": not dry_run, "dry_run": dry_run}
    from r2 import sandbox_git
    repo = _sandbox_freeze_repo()
    original = F._git
    F._git = _guarded_freeze_git(F)
    try:
        with sandbox_git.isolated():
            commit = F.freeze(repo=repo, runs_v2=paths.RUNS_V2, dry_run=dry_run)
    except F.FreezeRefused as exc:
        raise Refused(f"r2.freeze: {exc}") from exc
    finally:
        F._git = original
    return {"frozen": not dry_run, "dry_run": dry_run, "commit": commit, "repo": str(repo)}


# ---------------------------------------------------------------- the public timestamp of the freeze
FREEZE_TAG_NAME = "r2-analysis-plan-frozen"
PUBLIC_KINDS = (
    ("pushed tag", re.compile(r"https://(?:github\.com|gitlab\.com)/[\w.-]+/[\w.-]+/(?:releases/tag|tree|-/tags)/"
                              + re.escape(FREEZE_TAG_NAME) + r"/?")),
    ("OSF registration", re.compile(r"https://osf\.io/[a-z0-9]{5,}/?|https://doi\.org/10\.17605/OSF\.IO/[A-Z0-9]{5,}")),
    ("Zenodo DOI", re.compile(r"(?:https://doi\.org/)?10\.5281/zenodo\.\d+|https://zenodo\.org/records?/\d+")),
)
PIPELINE_INCIDENTS = "pipeline.incidents.jsonl"  # under r2_runs/main; check_complete ignores *.incidents.jsonl


def public_kind(url):
    """'pushed tag', 'OSF registration', 'Zenodo DOI', or None for anything else."""
    for kind, rx in PUBLIC_KINDS:
        if rx.fullmatch(url.strip()):
            return kind
    return None


def _freeze_date(text=None):
    if text is None:
        if not paths.FREEZE_RECORD.exists():
            return None
        text = paths.FREEZE_RECORD.read_text(encoding="utf-8")
    m = re.search(r"^- Date: (\d{4}-\d{2}-\d{2})$", text, re.M)
    return m.group(1) if m else None


def public_timestamp():
    """The public timestamp FREEZE_RECORD.md records ({url, kind, since, tag_commit}), or None (not recorded, or
    not a valid public URL)."""
    if not paths.FREEZE_RECORD.exists():
        return None
    text = paths.FREEZE_RECORD.read_text(encoding="utf-8")
    got = {k: (m.group(1).strip() if m else None) for k, m in (
        ("url", re.search(r"^- Public URL: (\S+)$", text, re.M)),
        ("kind", re.search(r"^- Kind: (.+)$", text, re.M)),
        ("since", re.search(r"^- Public since: (\d{4}-\d{2}-\d{2})$", text, re.M)),
        ("tag_commit", re.search(r"^- Tag commit: (\S+)$", text, re.M)))}
    if not got["url"] or not got["since"] or public_kind(got["url"]) is None:
        return None
    return got


def _calendar_date(value, what):
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise Refused(f"{what} must be a calendar date YYYY-MM-DD, got {value!r} (not a calendar date)") from None


def record_public(url, date=None):
    """Record the public timestamp of the freeze in FREEZE_RECORD.md: the pushed tag's URL, an OSF registration
    or a Zenodo DOI, and the date it became public (default today, UTC). Run after the authors approved and made
    the push. Idempotent for the same URL; refuses a different one."""
    from r2.freeze import TAG
    check_write_paths()
    _need(paths.FREEZE_RECORD)
    commit = _tag_commit(TAG)
    if not commit:
        raise Refused(f"no tag {TAG} in {paths.FREEZE_REPO}: freeze first")
    url = url.strip()
    kind = public_kind(url)
    if kind is None:
        raise Refused(f"{url!r} is not a public timestamp: give the pushed tag's URL "
                      f"(https://github.com/<owner>/<repo>/releases/tag/{TAG} or .../tree/{TAG}), an OSF link "
                      "(https://osf.io/<id>) or a Zenodo DOI (10.5281/zenodo.<n>)")
    since = _calendar_date(date or _today(), "--date")
    frozen = _freeze_date()
    if frozen and since < frozen:
        raise Refused(f"--date {since} is before the freeze ({frozen})")
    if since > _today():
        raise Refused(f"--date {since} is in the future")
    current = public_timestamp()
    if current:
        if current["url"] != url:
            raise Refused(f"{paths.FREEZE_RECORD} already records the public timestamp {current['url']}; "
                          "edit the record by hand, with a dated entry, if it changed")
        return {**current, "already": True}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    block = (f"Public timestamp: recorded by python -m r2.pipeline record-public on {stamp}.\n\n"
             f"- Public URL: {url}\n- Kind: {kind}\n- Public since: {since}\n- Tag commit: {commit}")
    text = paths.FREEZE_RECORD.read_text(encoding="utf-8")
    new, n = re.subn(r"Public timestamp: pending a push\..*?(?=\n\n|\Z)", lambda m: block, text, count=1, flags=re.S)
    if not n:
        new = text.rstrip("\n") + "\n\n" + block + "\n"
    paths.FREEZE_RECORD.write_text(new, encoding="utf-8")
    print(f"recorded in {paths.FREEZE_RECORD}: {kind} {url} (public since {since}); commit this file")
    return {"url": url, "kind": kind, "since": since, "tag_commit": commit}


def _pipeline_incident(rec):
    """One line in r2_runs/main/pipeline.incidents.jsonl (evidence, archived with the run logs)."""
    path = paths.RUNS_V2 / "main" / PIPELINE_INCIDENTS
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"at": datetime.now(timezone.utc).isoformat(), **rec}
    if paths.SANDBOX is not None:
        line["synthetic"] = True
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return path


def _pipeline_incidents():
    path = paths.RUNS_V2 / "main" / PIPELINE_INCIDENTS
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- record (public facts after the release)
def record(release_tag=None, zenodo_doi=None):
    """The release facts the documents cite (docs/r2/PUBLIC_RECORD.json), after the authors approved and made the
    release. The release tag must exist (in FREEZE_REPO: the repository, or the sandbox's freeze repository)."""
    check_write_paths()
    if not (release_tag or zenodo_doi):
        raise Refused("nothing to record: give --release-tag and/or --zenodo-doi (the public timestamp of the "
                      "freeze is recorded by record-public)")
    if release_tag and not _tag_commit(release_tag):
        raise Refused(f"no tag {release_tag} in {paths.FREEZE_REPO}: create the release tag first")
    if zenodo_doi and not DOI_RE.fullmatch(zenodo_doi):
        raise Refused(f"--zenodo-doi must be a DOI (10.xxxx/...), got {zenodo_doi!r}")
    path = paths.PUBLIC_RECORD
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    new = {k: v for k, v in (("release_tag", release_tag), ("zenodo_new_doi", zenodo_doi)) if v}
    clash = {k: (data[k], v) for k, v in new.items() if k in data and data[k] != v}
    if clash:
        raise Refused(f"{path} already records different values: {clash}; edit it by hand if a fact changed")
    data.update(new)
    if paths.SANDBOX is not None:
        data["synthetic"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(data.items())), indent=1) + "\n", encoding="utf-8")
    return data


# ---------------------------------------------------------------- post-results
QA_SCRIPTS = ("qa_numbers.py", "qa_xref.py", "qa_keyfacts.py", "qa_hygiene.py")
QA_VERDICTS = {  # script -> (report, verdict line pattern, label); the verdict must be 0 and the exit status 0
    "qa_numbers.py": ("qa_numbers_report.md", r"\*\*UNEXPLAINED: (\d+)\*\*", "UNEXPLAINED"),
    "qa_xref.py": ("qa_xref_report.md", r"^Cross-reference failures: (\d+)$", "Cross-reference failures"),
    "qa_keyfacts.py": ("qa_keyfacts_report.md", r"^Key-fact failures: (\d+)$", "Key-fact failures"),
    "qa_hygiene.py": ("qa_hygiene_report.md", r"^## Total hygiene hits: (\d+)$", "Total hygiene hits"),
}


def _doc_env():
    """The document scripts' environment. Under R2_SANDBOX without GIT_* variables (r2.sandbox_git.clean_env):
    their only git commands are read-only reads of the code checkout for the code listings."""
    env = {**os.environ, "R2_REPO": str(paths.REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    if paths.SANDBOX is not None:
        from r2 import sandbox_git
        env = sandbox_git.clean_env(env)
        env["R2_SANDBOX"] = str(paths.SANDBOX)
    else:
        env.pop("R2_SANDBOX", None)
    return env


def _doc_script(args, check=True):
    """Run a script of npj_R2_WORKING (cwd R2_WORKING, R2_REPO = this repository)."""
    cmd = [PY, *args]
    _show(["python", *args])
    t0 = time.time()
    res = subprocess.run(cmd, cwd=paths.R2_WORKING, env=_doc_env(), capture_output=True, text=True)
    out = {"command": shlex.join(["python", *args]), "rc": res.returncode, "seconds": round(time.time() - t0, 1)}
    if res.returncode:
        out["tail"] = (res.stdout[-1500:] + ("\n" + res.stderr[-1500:] if res.stderr.strip() else "")).strip()
    if check and res.returncode:
        raise Refused(f"{out['command']} exited {res.returncode}:\n{out['tail']}")
    return out


def run_qa():
    """The four QA scripts. Prints each verdict on its own line; any failure (a verdict above 0, a non-zero exit
    status, or no verdict) stops with Refused naming the script, the verdict and the report."""
    results, failed = {}, []
    for script in QA_SCRIPTS:
        report, rx, label = QA_VERDICTS[script]
        path = paths.R2_WORKING / report
        out = _doc_script([script], check=False)
        m = re.search(rx, path.read_text(encoding="utf-8"), re.M) if path.exists() else None
        n = int(m.group(1)) if m else None
        ok = out["rc"] == 0 and n == 0
        verdict = f"{label}: {n}" if n is not None else f"no verdict in {report}"
        print(f"QA {script}: {verdict}: {'PASS' if ok else 'FAIL'}" + ("" if ok else f" (exit status {out['rc']})"),
              flush=True)
        results[script] = {"rc": out["rc"], "failures": n, "report": str(path)}
        if not ok:
            failed.append(f"{script}: {verdict} (exit status {out['rc']}; see {path})")
    if failed:
        raise Refused("QA failed: " + "; ".join(failed))
    return results


def order_check():
    """The freeze date and the public timestamp (FREEZE_RECORD.md) are on or before the first main-run call
    (headline.json rep_call_dates.first), compared by day. Prints each comparison; any failure, or a check that
    cannot be made, stops with Refused. A main run started with --accept-unpublished-freeze (an incident line)
    passes the public comparison as ACCEPTED."""
    rec, head = paths.FREEZE_RECORD, paths.ANALYSIS_REPEATED / "headline.json"
    first = (json.loads(head.read_text()).get("rep_call_dates") or {}).get("first") if head.exists() else None
    frozen = _freeze_date()
    if not first or not frozen:
        why = [w for w, bad in ((f"no freeze date in {rec}", not frozen),
                                (f"no rep_call_dates.first in {head}", not first)) if bad]
        raise Refused("ORDER CHECK cannot be made: " + "; ".join(why))
    day, problems = first[:10], []
    ok = frozen <= day
    print(f"ORDER CHECK freeze: {frozen} <= first main-run call {first}: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        problems.append(f"the freeze date {frozen} is after the first main-run call {first}")
    pub = public_timestamp()
    if pub:
        ok_pub = pub["since"] <= day
        print(f"ORDER CHECK public: {pub['since']} <= first main-run call {first}: {'PASS' if ok_pub else 'FAIL'}",
              flush=True)
        problem = f"the public timestamp {pub['since']} is after the first main-run call {first}"
    else:
        ok_pub = False
        print(f"ORDER CHECK public: no public timestamp in {rec}: FAIL", flush=True)
        problem = f"no public timestamp in {rec} (python -m r2.pipeline record-public <url>)"
    accepted = [i for i in _pipeline_incidents() if i.get("kind") == "unpublished_freeze"]
    if not ok_pub and accepted:
        print(f"ORDER CHECK public: ACCEPTED by --accept-unpublished-freeze (incident of {accepted[0]['at']} in "
              f"{paths.RUNS_V2 / 'main' / PIPELINE_INCIDENTS})", flush=True)
    elif not ok_pub:
        problems.append(problem)
    if problems:
        raise Refused("order check failed: " + "; ".join(problems))
    return {"ok": True, "freeze_date": frozen, "public_since": pub["since"] if pub else None,
            "first_main_call": first, "accepted_unpublished_freeze": bool(accepted and not (pub and ok_pub))}


def data_files_table():
    """tables/S_data_files.csv (Supplementary Table S15) from docs/DATA_FILES.md, once the release is recorded."""
    from r2.analysis.supp_tables import write_csv
    src = paths.REPO / "docs" / "DATA_FILES.md"
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0].startswith("`"):
            role = cells[3].replace("`", "")
            first = re.split(r"(?<=[a-z0-9)])\. (?=[A-Z])", role)[0].rstrip(".") + "."
            rows.append({"path": cells[0].strip("`"), "role": first, "experiment": cells[4].replace("`", ""),
                         "sha256": cells[5].strip("`")})
    out = paths.TABLES_DIR / "S_data_files.csv"
    if paths.SANDBOX is not None:
        rows = [{**r, "synthetic": True} for r in rows]
    write_csv(out, rows)
    return len(rows)


def post_results(render=True):
    """Every post-results step in order; returns {step: result}. Stops at the first failing step (Refused): the
    completeness gate, the order check (freeze and public timestamp before the first main-run call), a document
    build, or a QA verdict."""
    from r2.analysis import figures, repeated, supp_tables, table1
    from r2.run import check_complete
    check_write_paths()
    _need(paths.DATA_V2 / "pairs_v2.json", paths.VERIFY_DIR / "excluded_pairs.json", paths.RUNS_V2 / "main",
          paths.ANALYSIS_REPEATED / "model_availability.csv", paths.R2_WORKING / "edits" / "numbers.py")
    steps = {}

    def step(name, fn):
        t0 = time.time()
        print(f"== {name}", flush=True)
        try:
            res = fn()
        except SystemExit as exc:  # check_complete's gate and the CLIs exit with a message
            raise Refused(f"{name}: {exc}") from exc
        steps[name] = {"result": res, "seconds": round(time.time() - t0, 1)}
        return res

    step("check_complete", lambda: (check_complete.main([str(paths.RUNS_V2 / "main")]),
                                    str(paths.ANALYSIS_REPEATED / "completeness.csv"))[1])
    step("repeated", lambda: {k: v for k, v in repeated.run().items() if k in ("n_models", "n_pairs", "n_cells")})
    step("order_check", order_check)  # before any document is built: a failure stops here
    step("supp_tables", lambda: supp_tables.write_repeated())
    step("table1", lambda: len(table1.write_table1(paths.TABLES_DIR)))
    step("figures", lambda: {k: str(v["png"]) for k, v in figures.make_figures().items()})
    if paths.PUBLIC_RECORD.exists() and json.loads(paths.PUBLIC_RECORD.read_text()).get("release_tag"):
        step("data_files_table", data_files_table)
    step("numbers", lambda: [_doc_script(["edits/numbers.py"]), _doc_script(["edits/numbers_supplement.py"])])
    step("manuscript", lambda: _doc_script(["edits/run_all.py"] + ([] if render else ["--no-render"])))
    step("supplement", lambda: _doc_script(["edits/build_supplement.py"] + (["--render"] if render else [])))
    step("response", lambda: _doc_script(["build_response.py"] + ([] if render else ["--no-pdf"])))
    step("qa", run_qa)
    step("pending_index", lambda: _doc_script(["make_pending_index.py"])["rc"])
    step("markers", lambda: markers()["counts"])
    return steps


# ---------------------------------------------------------------- markers
def _load_fill(work=None):
    """edits/fill.py of the document folder: the marker classification (NUMERIC / PROSE) the builders use."""
    path = Path(work or paths.R2_WORKING) / "edits" / "fill.py"
    _need(path)
    spec = importlib.util.spec_from_file_location("r2_working_fill", path)
    mod = importlib.util.module_from_spec(spec)
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # no __pycache__ in the document folder
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = before
    return mod


def doc_text(path):
    import docx
    from docx.oxml.ns import qn
    doc = docx.Document(str(path))
    parts = []
    for el in doc.element.body.iter(qn("w:p")):
        parts.append("".join(n.text or "" for n in el.iter() if n.tag == qn("w:t")))
    return "\n".join(parts)


def markers(work=None, write=True):
    """Remaining ⟦PENDING⟧ markers of the three documents, each classified by edits/fill.py: NUMERIC (has a
    source the pipeline fills; should be 0 after post-results) or PROSE (needs author-written text). Writes
    MARKERS_REMAINING.md into the document folder unless write=False."""
    work = Path(work or paths.R2_WORKING)
    fill = _load_fill(work)
    out = {"NUMERIC": [], "PROSE": [], "UNCLASSIFIED": []}
    for doc, name in DOCS.items():
        _need(work / name)
        for desc in PENDING_RE.findall(doc_text(work / name)):
            cls = fill.classify(desc)
            out.get(cls, out["UNCLASSIFIED"]).append({"document": doc, "marker": desc})
    counts = {k: len(v) for k, v in out.items()}
    lines = ["# Markers remaining after post-results", "",
             f"NUMERIC (should be 0): {counts['NUMERIC']}; PROSE (author text): {counts['PROSE']}; "
             f"UNCLASSIFIED: {counts['UNCLASSIFIED']}", ""]
    for cls in ("NUMERIC", "UNCLASSIFIED", "PROSE"):
        lines += [f"## {cls}", ""] + [f"- {m['document']}: {m['marker']}" for m in out[cls]] + [""]
    if write:
        (work / "MARKERS_REMAINING.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:3]))
    for m in out["NUMERIC"] + out["UNCLASSIFIED"]:
        print(f"  NOT FILLED {m['document']}: {m['marker'][:150]}")
    return {"counts": counts, **out}


# ---------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(prog="python -m r2.pipeline", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import-edits")
    p.add_argument("xlsx")
    p.add_argument("--author", default="A.G.")
    p.add_argument("--date", default=None, help="default: today (UTC)")
    p = sub.add_parser("consensus")
    for name in ("ag_r1", "mo_r1", "ag_r2", "mo_r2"):
        p.add_argument(name)
    p.add_argument("--consensus", default=None, help="consensus CSV (case_id,item,consensus); default "
                                                     "verification/consensus.csv")
    p = sub.add_parser("probe")
    p.add_argument("--yes", action="store_true")
    for kind in ("pilot", "main"):
        p = sub.add_parser(kind)
        p.add_argument("--yes", action="store_true")
        p.add_argument("--models", default="all")
        if kind == "main":
            p.add_argument("--retry-persistent", action="store_true")
            p.add_argument("--accept-spec-change", action="store_true")
            p.add_argument("--accept-unpublished-freeze", action="store_true",
                           help="run although FREEZE_RECORD.md records no public timestamp (writes an incident)")
    p = sub.add_parser("pilot-summary")
    p.add_argument("dir", nargs="?", default=None)
    p = sub.add_parser("freeze")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--yes", action="store_true")
    p = sub.add_parser("record-public")
    p.add_argument("url", help="the pushed tag's URL, an OSF link or a Zenodo DOI")
    p.add_argument("--date", default=None, help="the day it became public (default: today, UTC)")
    p = sub.add_parser("record")
    p.add_argument("--release-tag", default=None)
    p.add_argument("--zenodo-doi", default=None)
    p = sub.add_parser("scan-legacy-keys")
    p.add_argument("root", nargs="?", default=None, help=f"default {paths.LEGACY_ROOT}")
    p = sub.add_parser("post-results")
    p.add_argument("--no-render", action="store_true")
    sub.add_parser("markers")
    p = sub.add_parser("sandbox-init")
    p.add_argument("dir")
    p.add_argument("--models", type=int, default=None)
    return ap


def dispatch(a, client_factory=None, sleep=None):
    if a.cmd == "import-edits":
        return import_edits(a.xlsx, a.author, a.date)
    if a.cmd == "consensus":
        return consensus(a.ag_r1, a.mo_r1, a.ag_r2, a.mo_r2, a.consensus)
    if a.cmd == "probe":
        return probe(a.yes, client_factory, sleep)
    if a.cmd in ("pilot", "main"):
        flags = {"models": a.models}
        if a.cmd == "main":
            flags.update(retry_persistent=a.retry_persistent, accept_spec_change=a.accept_spec_change,
                         accept_unpublished_freeze=a.accept_unpublished_freeze)
        return run_harness(a.cmd, a.yes, client_factory, sleep, **flags)
    if a.cmd == "pilot-summary":
        return {"models": pilot_summary(a.dir)}
    if a.cmd == "freeze":
        return freeze(a.dry_run, a.yes)
    if a.cmd == "record-public":
        return record_public(a.url, a.date)
    if a.cmd == "record":
        return record(a.release_tag, a.zenodo_doi)
    if a.cmd == "scan-legacy-keys":
        res = scan_legacy_keys(a.root)
        return {"files": len(res["files"]), "providers": res["providers"]}
    if a.cmd == "post-results":
        return post_results(render=not a.no_render)
    if a.cmd == "markers":
        return markers()
    if a.cmd == "sandbox-init":
        return sandbox_init(a.dir, a.models)
    raise AssertionError(a.cmd)


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        res = dispatch(a)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except NeedsHumans as exc:
        print(f"WAITING: {exc}", file=sys.stderr)
        return 3
    if isinstance(res, int):
        return res
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
