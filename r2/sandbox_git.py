"""Git under R2_SANDBOX: no command may reach a real repository.

Every git command the sandbox runs (the throwaway freeze repository, r2.freeze inside it, the harness tag check)
goes through run():
- its environment loses GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE, GIT_OBJECT_DIRECTORY,
  GIT_ALTERNATE_OBJECT_DIRECTORIES, GIT_COMMON_DIR, GIT_CEILING_DIRECTORIES and every other GIT_* variable (git
  exports them inside hooks; inherited, they redirect a command into another repository), hooks are disabled
  and optional locks are off;
- before each command, `git rev-parse --show-toplevel`, the git directory and the common directory must all lie
  inside the sandbox (a .git file pointing elsewhere, or discovery climbing out of the sandbox, is refused).

The document builders' read-only reads of the code checkout (git show / diff --quiet / rev-parse / log on
R2_REPO, for the code listings) run with clean_env() too; they are the only git commands of a sandbox run
outside the sandbox, and they write nothing (GIT_OPTIONAL_LOCKS=0).
"""
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

from r2.analysis.synthetic import _within

LOCATION_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES")
NO_HOOKS = ("-c", "core.hooksPath=/dev/null")
IDENTITY = (("user.name", "R2 sandbox"), ("user.email", "sandbox@rehearsal.invalid"),
            ("commit.gpgsign", "false"), ("tag.gpgsign", "false"))


class SandboxGitError(RuntimeError):
    """A git command would act on a repository outside the sandbox, or failed."""


def clean_env(env=None):
    """`env` (default os.environ) without any GIT_* variable, plus GIT_OPTIONAL_LOCKS=0."""
    out = {k: v for k, v in dict(os.environ if env is None else env).items()
           if k not in LOCATION_VARS and not k.startswith("GIT_")}
    out["GIT_OPTIONAL_LOCKS"] = "0"
    return out


@contextmanager
def isolated():
    """os.environ without GIT_* variables for the duration (for code that runs git with the inherited
    environment, such as r2.freeze); restored afterwards."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("GIT_")}
    for k in saved:
        del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(saved)


def _inside(path, sandbox):
    return _within(Path(path).resolve(), Path(sandbox).resolve())


def location(repo, env=None):
    """(toplevel, git dir, common dir) of the repository git finds from `repo`."""
    env = clean_env() if env is None else env
    try:
        res = subprocess.run(["git", *NO_HOOKS, "rev-parse", "--show-toplevel", "--absolute-git-dir",
                              "--git-common-dir"], cwd=repo, env=env, capture_output=True, text=True)
    except OSError as exc:
        raise SandboxGitError(f"cannot run git in {repo}: {exc}") from exc
    lines = res.stdout.splitlines()
    if res.returncode or len(lines) < 3:
        raise SandboxGitError(f"{repo}: no git repository found ({res.stderr.strip() or res.stdout.strip()})")
    common = Path(lines[2])
    return Path(lines[0]), Path(lines[1]), (common if common.is_absolute() else Path(repo) / common)


def assert_inside(repo, sandbox, env=None):
    """The repository git acts on from `repo` lies inside `sandbox`: toplevel, git dir and common dir."""
    top, gitdir, common = location(repo, env)
    outside = [f"{name} {p.resolve()}" for name, p in (("toplevel", top), ("git dir", gitdir),
                                                       ("common dir", common)) if not _inside(p, sandbox)]
    if outside:
        raise SandboxGitError(f"refusing git in {repo}: {'; '.join(outside)} outside the sandbox {sandbox}")
    return top


def run(repo, *args, sandbox, check=False):
    """`git <args>` in `repo`, a repository inside `sandbox`, with clean_env() and no hooks. Checks the
    repository location before the command (for `init`: the target path before, the new repository after)."""
    repo = Path(repo)
    if not _inside(repo, sandbox):
        raise SandboxGitError(f"refusing git in {repo}: the path is outside the sandbox {sandbox}")
    env = clean_env()
    init = bool(args) and args[0] == "init"
    if not init:
        assert_inside(repo, sandbox, env)
    res = subprocess.run(["git", *NO_HOOKS, *args], cwd=repo, env=env, capture_output=True, text=True)
    if init and res.returncode == 0:
        top = assert_inside(repo, sandbox, env)
        if top.resolve() != repo.resolve():
            raise SandboxGitError(f"git init in {repo} produced a repository rooted at {top}")
    if check and res.returncode:
        raise SandboxGitError(f"git {' '.join(args)} failed in {repo}: {res.stderr.strip() or res.stdout.strip()}")
    return res


def init_repo(repo, sandbox):
    """A git repository of its own at `repo` (inside `sandbox`), with the sandbox identity and no signing.
    A .git file (a link to a repository elsewhere) is refused. Every step's exit status is checked."""
    repo = Path(repo)
    dot = repo / ".git"
    if dot.exists() and not dot.is_dir():
        raise SandboxGitError(f"{dot} is a gitfile (a link to another repository), not a repository of its own")
    if not dot.exists():
        repo.mkdir(parents=True, exist_ok=True)
        run(repo, "init", "-q", sandbox=sandbox, check=True)
        for k, v in IDENTITY:
            run(repo, "config", k, v, sandbox=sandbox, check=True)
    assert_inside(repo, sandbox)
    return repo
