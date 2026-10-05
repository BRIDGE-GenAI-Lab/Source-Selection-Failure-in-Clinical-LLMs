"""Freeze the analysis plan: fill hashes and date, drop the DRAFT banner, write FREEZE_RECORD.md,
commit, and create the annotated tag LOCALLY. Never pushes (pushing is the user's decision).

    python -m r2.freeze [--dry-run]
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from r2 import paths
from r2.run.harness import safe_name

TAG = "r2-analysis-plan-frozen"
BANNER = "DRAFT — not frozen"
PLAN = "docs/r2/ANALYSIS_PLAN.md"
RECORD = "docs/r2/FREEZE_RECORD.md"
HASHED = ("data/v2/pairs_v2.json", "r2/stimuli/render.py", "verification/excluded_pairs.json",
          "r2/run/models_v2.json", "r2/run/harness.py")
N_PAIRS = 500
FOOTER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


class FreezeRefused(RuntimeError):
    pass


def _git(repo, *args, check=True):
    res = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise FreezeRefused(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_preconditions(repo, runs_v2):
    repo, runs_v2 = Path(repo), Path(runs_v2)
    excluded = repo / "verification" / "excluded_pairs.json"
    if not excluded.is_file():
        raise FreezeRefused(f"{excluded} does not exist: the physician verification is not finished")
    pairs_path = repo / "data" / "v2" / "pairs_v2.json"
    if not pairs_path.is_file():
        raise FreezeRefused(f"{pairs_path} does not exist")
    pairs = json.loads(pairs_path.read_text())
    if pairs.get("unresolved"):
        raise FreezeRefused(f"{pairs_path} has {len(pairs['unresolved'])} unresolved pairs; need 0")
    if len(pairs.get("cases", [])) != N_PAIRS:
        raise FreezeRefused(f"{pairs_path} has {len(pairs.get('cases', []))} cases; need exactly {N_PAIRS}")
    pilot = runs_v2 / "pilot"
    if not pilot.is_dir():
        raise FreezeRefused(f"pilot directory {pilot} does not exist: run the pilot first")
    specs = json.loads((repo / "r2" / "run" / "models_v2.json").read_text())["models"]
    missing = []
    for spec in specs:
        f = pilot / f"{safe_name(spec['model_id'])}.jsonl"
        if not f.is_file() or not any(line.strip() for line in f.read_text(encoding="utf-8").splitlines()):
            missing.append(spec["model_id"])
    if missing:
        raise FreezeRefused(f"pilot has no records for model(s): {missing}")
    dirty = _git(repo, "status", "--porcelain").stdout.strip()
    if dirty:
        raise FreezeRefused(f"working tree is not clean (commit or remove first):\n{dirty}")
    if _git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{TAG}", check=False).returncode == 0:
        raise FreezeRefused(f"tag {TAG} already exists; a freeze is final")
    plan = (repo / PLAN).read_text(encoding="utf-8")
    if not plan.startswith(BANNER):
        raise FreezeRefused(f"{PLAN} has no DRAFT banner on its first line (already frozen?)")


def fill_plan(text, hashes, date):
    lines = text.split("\n")
    lines.pop(0)  # banner
    while lines and not lines[0].strip():
        lines.pop(0)
    text = "\n".join(lines)
    for rel, digest in hashes.items():
        text, n = re.subn(rf"^(- {re.escape(rel)}: )<sha>$", rf"\g<1>{digest}", text, flags=re.M)
        if n != 1:
            raise FreezeRefused(f"{PLAN}: expected one '<sha>' slot for {rel}, found {n}")
    if "<YYYY-MM-DD>" not in text:
        raise FreezeRefused(f"{PLAN}: no <YYYY-MM-DD> slot")
    text = text.replace("<YYYY-MM-DD>", date)
    if "<sha>" in text or "<YYYY-MM-DD>" in text:
        raise FreezeRefused(f"{PLAN}: unfilled slots remain")
    return text


def freeze_record(base, hashes, date):
    rows = "\n".join(f"- {rel}: {digest}" for rel, digest in hashes.items())
    return f"""# Freeze record

- Tag: {TAG} (annotated, created locally)
- Base commit (HEAD immediately before the freeze commit): {base}
- The tag points to the freeze commit that adds this file and the filled plan; resolve it with
  `git rev-parse {TAG}^{{commit}}`.
- Date: {date}
- Analysis plan: {PLAN}

SHA-256 of the frozen inputs:

{rows}

Public timestamp: pending a push. The tag has not been pushed. After the user approves a push, record the
remote tag date and the tag SHA here, with a dated entry.

Changes after the freeze (each needs a dated entry here before any further main-run call):

- none
"""


def _rollback(repo, base, before, cause):
    """Undo the freeze commit after a failed tag: soft-reset it, restore the plan and record files from
    their pre-freeze content, and report. Raises FreezeRefused in every case."""
    try:
        _git(repo, "reset", "--soft", "HEAD~1")
        _git(repo, "reset", "-q", "HEAD", "--", PLAN, RECORD)
        for rel, content in before.items():
            target = Path(repo) / rel
            if content is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(content)
    except (FreezeRefused, OSError) as exc2:
        raise FreezeRefused(f"tagging failed ({cause}) AND rollback failed ({exc2}); the freeze commit "
                            f"may still be present on HEAD (base {base}): inspect `git status` and "
                            f"`git log` before retrying") from exc2
    raise FreezeRefused(f"tagging failed ({cause}); freeze commit rolled back to {base}, plan and record "
                        f"restored; no tag was created") from cause


def freeze(repo=None, runs_v2=None, dry_run=False, today=None):
    repo = Path(repo or paths.REPO)
    runs_v2 = Path(runs_v2 or paths.RUNS_V2)
    check_preconditions(repo, runs_v2)
    date = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hashes = {rel: _sha(repo / rel) for rel in HASHED}
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    plan_text = fill_plan((repo / PLAN).read_text(encoding="utf-8"), hashes, date)
    if dry_run:
        print(f"dry run: would freeze at base {base} on {date}")
        for rel, digest in hashes.items():
            print(f"  {rel}: {digest}")
        print(f"  would fill and de-banner {PLAN}, write {RECORD}, commit both, "
              f"and create local annotated tag {TAG} (no push)")
        return None
    before = {rel: ((repo / rel).read_bytes() if (repo / rel).exists() else None) for rel in (PLAN, RECORD)}
    (repo / PLAN).write_text(plan_text, encoding="utf-8")
    (repo / RECORD).write_text(freeze_record(base, hashes, date), encoding="utf-8")
    _git(repo, "add", PLAN, RECORD)
    _git(repo, "commit", "-q", "-m", f"r2: freeze analysis plan for the repeated experiment\n\n{FOOTER}")
    try:
        _git(repo, "tag", "-a", TAG, "-m", "Analysis plan frozen before repeated-experiment data collection")
    except FreezeRefused as exc:
        _rollback(repo, base, before, exc)
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    print(f"frozen: tag {TAG} -> {commit} (local only; not pushed)")
    return commit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print what would be done and change nothing")
    args = parser.parse_args(argv)
    try:
        freeze(dry_run=args.dry_run)
    except FreezeRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
