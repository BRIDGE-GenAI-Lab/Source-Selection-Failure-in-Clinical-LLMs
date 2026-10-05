"""Unified runner for the repeated experiment: one prompt (r2.stimuli.render), both orders,
one code path for every provider (OpenAI-compatible endpoints), full request/response logs."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from r2.paths import DATA_V2, REPO, RUNS_V2, VERIFY_DIR
from r2.stimuli.render import render_messages, visible_content

ORDERS = ("TS", "ST")
PROVIDERS = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY"},
    "together": {"base_url": "https://api.together.xyz/v1", "key_env": "TOGETHER_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com", "key_env": "DEEPSEEK_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY"},
}
RETRY_PROMPT = "Return only the JSON object described above, with no other text."
TRANSIENT = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}
THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
TOKEN_PARAMS = ("max_tokens", "max_completion_tokens")
INCIDENTS = "incidents.jsonl"  # the former shared incident file; still never treated as a result file
INCIDENTS_SUFFIX = ".incidents.jsonl"  # per model: <safe_name>.incidents.jsonl, written under the model lock
N_PAIRS = 500
RETRY_TRIES = 10
RETRY_AFTER_CAP = 900  # seconds; a longer Retry-After is slept as this, then retried
FREEZE_TAG = "r2-analysis-plan-frozen"


class ModelMismatch(RuntimeError):
    pass


class ModelUnavailable(RuntimeError):
    pass


class StaleRecords(RuntimeError):
    pass


class CorruptResults(RuntimeError):
    pass


class RunLocked(RuntimeError):
    pass


CELL_ERRORS = {"BadRequestError", "UnprocessableEntityError"}
MODEL_ERRORS = {"AuthenticationError", "PermissionDeniedError", "NotFoundError"}
# Circuit breaker: stop a model after this many consecutive cell-error records, or when the same
# error text (first ERROR_TEXT_CHARS characters) repeats this many times among error records with no ok
# record in between. Only fresh cells count; a re-attempted cell (one that already has an error-only
# record) is invisible to the breaker. A re-attempt that fails with the error text of an earlier
# error-only record for the same cell is final: it is written with "persistent_error": true.
BREAKER_LIMIT = 3
ERROR_TEXT_CHARS = 200


def _error_kind(exc):
    name = type(exc).__name__
    if name in CELL_ERRORS:
        return "cell"
    if name in MODEL_ERRORS:
        return "model"
    return None


def load_specs(path=REPO / "r2" / "run" / "models_v2.json"):
    specs = json.loads(Path(path).read_text())["models"]
    keys = {"model_id": lambda s: s["model_id"], "safe_name(model_id)": lambda s: safe_name(s["model_id"]),
            "original_name": lambda s: s["original_name"]}
    for label, key in keys.items():
        dup = sorted(k for k, n in Counter(key(s) for s in specs).items() if n > 1)
        if dup:
            raise ValueError(f"{path}: duplicate {label}: {dup}")
    return specs


def load_v2_cases(path=DATA_V2 / "pairs_v2.json"):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("unresolved"):
        raise ValueError(f"{path}: corrected stimulus set still has {len(data['unresolved'])} unresolved pairs")
    cases = {c["case_id"]: c for c in data["cases"]}
    if len(cases) != N_PAIRS or len(data["cases"]) != N_PAIRS:
        raise ValueError(f"{path}: expected {N_PAIRS} pairs, found {len(data['cases'])} "
                         f"({len(cases)} distinct case_ids)")
    same = sorted(cid for cid, c in cases.items()
                  if visible_content(c["tools"]["T"]) == visible_content(c["tools"]["S"]))
    if same:
        raise ValueError(f"{path}: authentic and modified excerpts render identically for {same}")
    return cases, hashlib.sha256(raw).hexdigest()


def make_client(provider):
    from openai import OpenAI
    cfg = PROVIDERS[provider]
    return OpenAI(api_key=os.environ[cfg["key_env"]], base_url=cfg["base_url"], timeout=600, max_retries=0)


def safe_name(model_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def results_path(spec, out_dir):
    return Path(out_dir) / f"{safe_name(spec['model_id'])}.jsonl"


def incidents_path(spec, out_dir):
    return Path(out_dir) / f"{safe_name(spec['model_id'])}{INCIDENTS_SUFFIX}"


def is_incident_file(name):
    return name == INCIDENTS or name.endswith(INCIDENTS_SUFFIX)


def iter_result_files(out_dir):
    """Result files in out_dir: *.jsonl except AppleDouble (._*) and incident files."""
    return [p for p in sorted(Path(out_dir).glob("*.jsonl"))
            if not p.name.startswith("._") and not is_incident_file(p.name)]


def is_error_only(r):
    """An API error before any response was received. Not a completed cell unless it is marked
    persistent_error (see is_done); otherwise re-attempted on resume."""
    return "error" in r and not r.get("attempts")


def is_done(r):
    """A record that completes its cell: it carries a response, or it is a final persistent error."""
    return not is_error_only(r) or r.get("persistent_error") is True


def _error_text(r):
    return r["error"][:ERROR_TEXT_CHARS]


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def final_records(path):
    """The last record per (case_id, order); analysis and export use only these."""
    last = {}
    for r in read_records(path):
        last[(r["case_id"], r["order"])] = r
    return list(last.values())


class _Breaker:
    def __init__(self, spec):
        self.spec, self.consecutive, self.text, self.same = spec, 0, None, 0

    def observe(self, rec):
        """Called for fresh cells only. Any record without an error resets the consecutive count; an ok
        record also resets the same-text count."""
        if rec.get("status") == "ok":
            self.text, self.same = None, 0
        if "error" not in rec:
            self.consecutive = 0
            return
        self.consecutive += 1
        text = _error_text(rec)
        self.same = self.same + 1 if text == self.text else 1
        self.text = text
        if self.consecutive >= BREAKER_LIMIT or self.same >= BREAKER_LIMIT:
            raise ModelUnavailable(
                f"{self.spec['model_id']}: circuit breaker ({self.consecutive} consecutive cell errors, "
                f"same error text {self.same}x in a row): {text}")


def parse_decision(text):
    if not text:
        return None
    if "</think>" in text.lower():
        text = text[text.lower().rfind("</think>") + len("</think>"):]
    elif "<think>" in text.lower():
        return None
    text = THINK.sub("", text)
    end = text.rfind("}")
    start = text.find("{")
    while start != -1 and start < end:
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        sel = str(obj.get("selected_tool", "")).strip().upper() if isinstance(obj, dict) else ""
        if sel in ("A", "B"):
            obj["selected_tool"] = sel
            obj["confidence_raw"] = obj.get("confidence")
            try:
                conf = float(obj["confidence_raw"])
            except (TypeError, ValueError):
                conf = None
            obj["confidence"] = conf if conf is not None and 0.0 <= conf <= 1.0 else None  # no rescaling
            diffs = obj.get("differences_found")
            if isinstance(diffs, str):
                obj["differences_found"] = [diffs] if diffs.strip() else []
            elif not isinstance(diffs, list):
                obj["differences_found"] = []
            return obj
        start = text.find("{", start + 1)
    return None


def _norm(model):
    return (model or "").split("/")[-1].lower()


def check_served(spec, served):
    """Exact match after dropping any org/ prefix and lower-casing; `served_prefix` in the spec is an
    explicit opt-in for a different expected served identifier."""
    expected = _norm(spec.get("served_prefix") or spec["model_id"])
    if _norm(served) != expected:
        raise ModelMismatch(f"requested {spec['model_id']!r}, provider served {served!r}")


def _retry_after(exc):
    """Seconds requested by a Retry-After header (delta-seconds or HTTP-date) on the exception's HTTP
    response; None when absent, unparseable, non-finite or negative (the caller then uses its backoff)."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _create(client, kwargs, sleep, tries=RETRY_TRIES):
    """One API call with retries on transient errors: exponential backoff, or the server's Retry-After
    when longer, capped at RETRY_AFTER_CAP. Every sleep is announced on stdout first."""
    for i in range(tries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if type(exc).__name__ not in TRANSIENT or i == tries - 1:
                raise
            backoff = min(300, 2 * 2 ** i)
            requested = _retry_after(exc)
            delay = backoff if requested is None else min(RETRY_AFTER_CAP, max(requested, backoff))
            note = ("none" if requested is None else
                    f"{requested:g} s" + (f", capped at {RETRY_AFTER_CAP} s" if requested > RETRY_AFTER_CAP else ""))
            print(f"RETRY {kwargs.get('model')} try {i + 1}/{tries}: {type(exc).__name__}: {str(exc)[:200]}; "
                  f"sleeping {delay:g} s (backoff {backoff} s, Retry-After {note})", flush=True)
            sleep(delay)


def execute_cell(client, spec, case, order, context, sleep=time.sleep, out_dir=None):
    msgs = render_messages(case, order)
    messages = [{"role": "system", "content": msgs["system"]}, {"role": "user", "content": msgs["user"]}]
    attempts, decision, error = [], None, None
    for _ in range(2):
        kwargs = {"model": spec["model_id"], "messages": messages,
                  spec.get("token_param", "max_tokens"): spec["max_tokens"]}
        if spec.get("temperature") is not None:
            kwargs["temperature"] = spec["temperature"]
        t0 = time.time()
        try:
            resp = _create(client, kwargs, sleep).model_dump()
        except Exception as exc:
            kind = _error_kind(exc)
            if kind == "model":
                raise ModelUnavailable(f"{spec['model_id']}: {type(exc).__name__}: {exc}"[:500]) from exc
            if kind == "cell":
                error = f"{type(exc).__name__}: {exc}"[:500]
                break
            raise
        latency_ms = (time.time() - t0) * 1000
        try:
            check_served(spec, resp.get("model"))
        except ModelMismatch as exc:
            if out_dir is not None:  # the paid response is the drift evidence: keep it before stopping
                _log_incident(out_dir, spec, {"kind": "model_mismatch", "model_id": spec["model_id"],
                                        "original_name": spec["original_name"], "case_id": case["case_id"],
                                        "order": order, "error": str(exc), "request": kwargs, "response": resp,
                                        "latency_ms": latency_ms, **context})
            raise
        message = ((resp.get("choices") or [{}])[0].get("message")) or {}
        content = message.get("content") or ""
        attempts.append({"request": kwargs, "response": resp, "latency_ms": latency_ms})
        decision = parse_decision(content)
        if decision:
            break
        if content:
            messages = messages + [{"role": "assistant", "content": content},
                                   {"role": "user", "content": RETRY_PROMPT}]
    selected = decision["selected_tool"] if decision else None
    last = attempts[-1]["response"] if attempts else {}
    rec = {
        **context,
        "model_id": spec["model_id"], "provider": spec["provider"], "original_name": spec["original_name"],
        "model_served": last.get("model"), "system_fingerprint": last.get("system_fingerprint"),
        "case_id": case["case_id"], "order": order, "labels": msgs["labels"],
        "payload_sha256": msgs["payload_sha256"], "system": msgs["system"], "user": msgs["user"],
        "attempts": attempts, "decision": decision,
        "status": "ok" if decision else "non_response",
        "selected_tool": selected,
        "correct": None if selected is None else int(msgs["labels"][selected] == "T"),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        rec["error"] = error
    return rec


def _log_incident(out_dir, spec, incident):
    """Append one incident to the model's own incident file. Callers hold the model lock (run_model)."""
    with open(incidents_path(spec, out_dir), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({**incident, "logged_at": datetime.now(timezone.utc).isoformat()},
                            ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _sha(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _spec_mismatches(r, spec, msgs):
    """(field, recorded, current) for every way a stored record differs from the current prompt and
    spec. Prompts are compared in full and reported by SHA-256."""
    out = []
    for field in ("system", "user"):
        if r.get(field) != msgs[field]:
            out.append((field, _sha(r.get(field)), _sha(msgs[field])))
    for field in ("model_id", "original_name"):
        if r.get(field) != spec[field]:
            out.append((field, r.get(field), spec[field]))
    if is_error_only(r):
        return out
    req = ((r.get("attempts") or [{}])[0] or {}).get("request") or {}
    temp = spec.get("temperature")
    if ("temperature" in req) != (temp is not None) or (temp is not None and req["temperature"] != temp):
        out.append(("temperature", req.get("temperature", "<absent>"), "<absent>" if temp is None else temp))
    token_param = spec.get("token_param", "max_tokens")
    sent = {k: req[k] for k in TOKEN_PARAMS if k in req}
    if sent != {token_param: spec["max_tokens"]}:
        out.append(("token_param", sent, {token_param: spec["max_tokens"]}))
    return out


def _done(path, cases, context, spec, out_dir, accept_spec_change=False):
    """(done, error_texts). done: cells with a completing record (is_done). error_texts: per cell, the
    first ERROR_TEXT_CHARS characters of every error-only record, for the persistent-error rule."""
    done, error_texts = set(), {}
    if not path.exists():
        return done, error_texts
    lines = path.read_bytes().split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    offset = 0
    for i, line in enumerate(lines):
        start, offset = offset, offset + len(line) + 1
        if not line.strip():
            continue
        try:
            r = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if i == len(lines) - 1:
                print(f"WARNING: dropped torn final line ({len(line)} bytes) in {path}", flush=True)
                os.truncate(path, start)  # keeps every earlier byte; nothing is rewritten
                return done, error_texts
            raise CorruptResults(f"{path}: unparseable line {i + 1}")
        key = (r["case_id"], r["order"])
        msgs = render_messages(cases[key[0]], key[1]) if key[0] in cases else None
        ok = (r.get("stimulus_sha256") == context["stimulus_sha256"] and msgs is not None
              and r.get("payload_sha256") == msgs["payload_sha256"])
        if not ok:
            raise StaleRecords(f"{path}: stale record for {key}")
        mismatches = _spec_mismatches(r, spec, msgs)
        if mismatches and not accept_spec_change:
            raise StaleRecords(f"{path}: record for {key} differs from the current prompt or spec in "
                               f"{[m[0] for m in mismatches]}; rerun with --accept-spec-change to keep it")
        for field, recorded, current in mismatches:
            print(f"ACCEPTED spec change {path.name} {key}: {field}: recorded {recorded!r}, "
                  f"current {current!r}", flush=True)
            _log_incident(out_dir, spec, {"kind": "accepted_spec_change", "model_id": spec["model_id"],
                                    "file": path.name, "case_id": key[0], "order": key[1], "field": field,
                                    "recorded": recorded, "current": current})
        if is_done(r):
            done.add(key)
        else:
            error_texts.setdefault(key, set()).add(_error_text(r))
    return done, error_texts


@contextmanager
def _model_lock(out_dir, spec):
    """Exclusive, non-blocking lock on <out_dir>/<safe_name>.lock for the whole of run_model."""
    lock_path = Path(out_dir) / f"{safe_name(spec['model_id'])}.lock"
    fh = open(lock_path, "a")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunLocked(f"{lock_path} is held by another process: "
                            f"{spec['model_id']} is already running") from exc
        yield
    finally:
        fh.close()  # closing the descriptor releases the lock


def run_model(spec, cases, out_dir, client, context, limit=None, sleep=time.sleep, accept_spec_change=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = results_path(spec, out_dir)
    with _model_lock(out_dir, spec):
        done, error_texts = _done(path, cases, context, spec, out_dir, accept_spec_change)
        ids = sorted(cases)[:limit] if limit else sorted(cases)
        if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n")
        breaker = _Breaker(spec)
        with open(path, "a", encoding="utf-8") as fh:
            for cid in ids:
                for order in ORDERS:
                    if (cid, order) in done:
                        continue
                    rec = execute_cell(client, spec, cases[cid], order, context, sleep=sleep,
                                       out_dir=out_dir)
                    earlier = error_texts.get((cid, order))
                    if earlier and is_error_only(rec) and _error_text(rec) in earlier:
                        rec["persistent_error"] = True
                        print(f"FINAL {spec['original_name']} {cid} {order}: same error as an earlier "
                              f"attempt, recorded as persistent_error: {_error_text(rec)}", flush=True)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    if not earlier:  # re-attempted cells count toward neither breaker rule
                        breaker.observe(rec)
    return path


def _clean_commit():
    status = subprocess.run(["git", "status", "--porcelain", "r2", "data/v2"], cwd=REPO,
                            capture_output=True, text=True)
    if status.returncode != 0:
        raise SystemExit(f"git status failed (rc {status.returncode}); cannot record harness_commit")
    if status.stdout.strip():
        raise SystemExit(f"commit r2/ and data/v2/ before running:\n{status.stdout.strip()}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    sha = head.stdout.strip()
    if head.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit(f"git rev-parse HEAD failed (rc {head.returncode}, output {sha!r}); "
                         "cannot record harness_commit")
    return sha


def _select(specs, selector):
    if selector == "all":
        return specs
    if selector.startswith("provider:"):
        provider = selector.split(":", 1)[1]
        if provider not in {s["provider"] for s in specs}:
            raise SystemExit(f"--models: unknown provider {provider!r}; "
                             f"known: {sorted({s['provider'] for s in specs})}")
        return [s for s in specs if s["provider"] == provider]
    names = {n.strip() for n in selector.split(",") if n.strip()}
    unknown = sorted(names - {s["original_name"] for s in specs})
    if unknown:
        raise SystemExit(f"--models: unknown model name(s) {unknown}; "
                         "use original_name values from models_v2.json")
    return [s for s in specs if s["original_name"] in names]


def _tag_exists(tag):
    res = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"], cwd=REPO,
                         capture_output=True, text=True)
    return res.returncode == 0


def _require_frozen(stim_sha):
    """The main run needs the frozen analysis plan and exactly the physician-verified stimulus file."""
    if not _tag_exists(FREEZE_TAG):
        raise SystemExit(f"refusing the main run: git tag {FREEZE_TAG} does not exist "
                         "(use --pilot for a pilot run)")
    path = VERIFY_DIR / "excluded_pairs.json"
    verified = json.loads(path.read_text()).get("pairs_v2_sha256") if path.exists() else None
    if verified != stim_sha:
        raise SystemExit(f"refusing the main run: data/v2/pairs_v2.json has sha256 {stim_sha}, but "
                         f"{path} records pairs_v2_sha256 = {verified} (the physician-verified stimuli)")


def _same_dir(a, b):
    """True when a and b name the same directory: os.path.samefile when both exist, else the resolved
    paths compared case-insensitively (the run volume is case-insensitive exFAT)."""
    a, b = Path(a), Path(b)
    if a.exists() and b.exists():
        return os.path.samefile(a, b)
    return str(a.resolve()).casefold() == str(b.resolve()).casefold()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="default: RUNS_V2/main, or RUNS_V2/pilot with --pilot")
    parser.add_argument("--pilot", action="store_true",
                        help="pilot run: skip the freeze-tag and verified-stimulus checks")
    parser.add_argument("--accept-spec-change", action="store_true",
                        help="keep stored records whose prompt, model mapping or decoding differs from the "
                             "current spec; each accepted mismatch is logged to "
                             "<out>/<safe_name>.incidents.jsonl")
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = str(RUNS_V2 / ("pilot" if args.pilot else "main"))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.pilot and _same_dir(args.out, RUNS_V2 / "main"):
        raise SystemExit("refusing a --pilot run into the main out dir")
    cases, stim_sha = load_v2_cases()
    if not args.pilot:
        _require_frozen(stim_sha)
    context = {"harness_commit": _clean_commit(), "stimulus_sha256": stim_sha}
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    for spec in _select(load_specs(), args.models):
        print(f"[{datetime.now():%H:%M:%S}] {spec['original_name']}", flush=True)
        try:
            run_model(spec, cases, args.out, make_client(spec["provider"]), context, limit=args.limit,
                      accept_spec_change=args.accept_spec_change)
        except (ModelMismatch, ModelUnavailable, RunLocked) as exc:
            print(f"STOPPED {spec['original_name']}: {exc}", flush=True)


if __name__ == "__main__":
    main()
