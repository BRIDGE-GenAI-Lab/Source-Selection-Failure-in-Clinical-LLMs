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
BANNER_LINE = (f"{BANNER}. The freeze tool (python -m r2.freeze) removes this line, fills the hash and date slots, "
               "and tags the commit.")
PLAN = "docs/r2/ANALYSIS_PLAN.md"
RECORD = "docs/r2/FREEZE_RECORD.md"
HASHED = ("data/v2/pairs_v2.json", "r2/stimuli/render.py", "verification/excluded_pairs.json",
          "r2/run/models_v2.json", "r2/run/harness.py")
N_PAIRS = 500
FOOTER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


class FreezeRefused(RuntimeError):
    pass


# ---------------------------------------------------------------- reading FREEZE_RECORD.md
# The public timestamp of the freeze: the pushed tag's URL, an OSF registration or a Zenodo DOI.
PUBLIC_KINDS = (
    ("pushed tag", re.compile(r"https://(?:github\.com|gitlab\.com)/[\w.-]+/[\w.-]+/(?:releases/tag|tree|-/tags)/"
                              + re.escape(TAG) + r"/?")),
    ("OSF registration", re.compile(r"https://osf\.io/[a-z0-9]{5,}/?|https://doi\.org/10\.17605/OSF\.IO/[A-Z0-9]{5,}")),
    ("Zenodo DOI", re.compile(r"(?:https://doi\.org/)?10\.5281/zenodo\.\d+|https://zenodo\.org/records?/\d+")),
)
RECORD_HASH = re.compile(r"^- (\S+): ([0-9a-f]{64})$", re.M)
CHANGES_HEAD = "Changes after the freeze"
CHANGE_FORMAT = "- YYYY-MM-DD: <path> sha256 <new SHA-256>: <reason>"
CHANGE_ENTRY = re.compile(r"^- (\d{4}-\d{2}-\d{2}): (\S+) sha256 ([0-9a-f]{64}): (\S.*?)\s*$", re.M)


def public_kind(url):
    """'pushed tag', 'OSF registration', 'Zenodo DOI', or None for anything else."""
    for kind, rx in PUBLIC_KINDS:
        if rx.fullmatch(url.strip()):
            return kind
    return None


def record_date(text):
    """The freeze date ("- Date: YYYY-MM-DD") of a FREEZE_RECORD.md text, or None."""
    m = re.search(r"^- Date: (\d{4}-\d{2}-\d{2})$", text, re.M)
    return m.group(1) if m else None


def frozen_hashes(text):
    """{path: SHA-256} of the HASHED files in a FREEZE_RECORD.md text, in HASHED order. Refuses (FreezeRefused) a
    missing or a repeated path."""
    found = {}
    for rel, digest in RECORD_HASH.findall(text):
        if rel in HASHED:
            if rel in found:
                raise FreezeRefused(f"{RECORD}: {rel} is listed more than once among the frozen hashes")
            found[rel] = digest
    missing = [rel for rel in HASHED if rel not in found]
    if missing:
        raise FreezeRefused(f"{RECORD}: no frozen SHA-256 for {missing}")
    return {rel: found[rel] for rel in HASHED}


def change_entries(text):
    """The dated entries under "Changes after the freeze" ({date, path, sha256, reason}), in file order."""
    i = text.find(CHANGES_HEAD)
    if i < 0:
        return []
    return [{"date": d, "path": p, "sha256": h, "reason": r} for d, p, h, r in CHANGE_ENTRY.findall(text[i:])]


def public_timestamp(text):
    """The public timestamp a FREEZE_RECORD.md text records ({url, kind, since, tag_commit}), or None (not
    recorded, or not a valid public URL)."""
    got = {k: (m.group(1).strip() if m else None) for k, m in (
        ("url", re.search(r"^- Public URL: (\S+)$", text, re.M)),
        ("kind", re.search(r"^- Kind: (.+)$", text, re.M)),
        ("since", re.search(r"^- Public since: (\d{4}-\d{2}-\d{2})$", text, re.M)),
        ("tag_commit", re.search(r"^- Tag commit: (\S+)$", text, re.M)))}
    if not got["url"] or not got["since"] or public_kind(got["url"]) is None:
        return None
    return got


def draft_plan(text):
    """The DRAFT analysis plan for `text`, which is either the draft itself or the plan as frozen (fill_plan undone:
    the banner line back, the hash and date slots emptied). The sandbox freeze and the tests freeze a copy of the
    draft, so they keep working after the real plan is frozen."""
    if text.startswith(BANNER):
        return text
    for rel in HASHED:
        text, n = re.subn(rf"^(- {re.escape(rel)}: )[0-9a-f]{{64}}$", r"\g<1><sha>", text, flags=re.M)
        if n != 1:
            raise FreezeRefused(f"{PLAN}: expected one frozen hash for {rel}, found {n}")
    text, n = re.subn(r"^Date frozen: \d{4}-\d{2}-\d{2}\.", "Date frozen: <YYYY-MM-DD>.", text, count=1, flags=re.M)
    if n != 1:
        raise FreezeRefused(f"{PLAN}: no 'Date frozen: YYYY-MM-DD.' line")
    return f"{BANNER_LINE}\n\n{text}"


def environment():
    """The analysis environment at the freeze: the Python version and every installed distribution with its
    version (as pip freeze would list them; read from importlib.metadata, nothing is installed or fetched)."""
    from importlib import metadata
    pkgs = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]},
                  key=str.lower)
    return {"python": sys.version.split()[0], "implementation": sys.implementation.name, "packages": pkgs}


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


def freeze_record(base, hashes, date, env=None):
    rows = "\n".join(f"- {rel}: {digest}" for rel, digest in hashes.items())
    env_text = ""
    if env:
        env_text = (f"\nAnalysis environment at the freeze ({env['implementation']} {env['python']}; installed "
                    "distributions, as pip freeze lists them):\n\n" + "\n".join(f"    {p}" for p in env["packages"]) + "\n")
    return f"""# Freeze record

- Tag: {TAG} (annotated, created locally)
- Base commit (HEAD immediately before the freeze commit): {base}
- The tag points to the freeze commit that adds this file and the filled plan; resolve it with
  `git rev-parse {TAG}^{{commit}}`.
- Date: {date}
- Analysis plan: {PLAN}

SHA-256 of the frozen inputs:

{rows}
{env_text}
Public timestamp: pending a push. The tag has not been pushed. After the user approves a push, record the
remote tag date and the tag SHA here, with a dated entry.

{CHANGES_HEAD}. The main run (python -m r2.run.harness, also through python -m r2.pipeline main) refuses
while any file above differs from its SHA-256, unless a dated entry below, committed before the run, gives the
file's new SHA-256 and the reason. Each accepted change is logged as a frozen_input_changed incident in
r2_runs/main/freeze.incidents.jsonl at every main-run start. One line per change:
`{CHANGE_FORMAT}`

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
    (repo / RECORD).write_text(freeze_record(base, hashes, date, environment()), encoding="utf-8")
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
