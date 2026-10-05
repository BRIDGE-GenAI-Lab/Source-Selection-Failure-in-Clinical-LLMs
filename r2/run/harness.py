"""Unified runner for the repeated experiment: one prompt (r2.stimuli.render), both orders,
one code path for every provider (OpenAI-compatible endpoints), full request/response logs."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
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
INCIDENTS = "incidents.jsonl"
N_PAIRS = 500
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
# error text (first ERROR_TEXT_CHARS characters) repeats this many times in a row among error records.
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


def iter_result_files(out_dir):
    return [p for p in sorted(Path(out_dir).glob("*.jsonl")) if not p.name.startswith("._")]


def is_error_only(r):
    """An API error before any response was received: not a completed cell, re-attempted on resume."""
    return "error" in r and not r.get("attempts")


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
        if "error" not in rec:
            self.consecutive = 0
            return
        self.consecutive += 1
        text = rec["error"][:ERROR_TEXT_CHARS]
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
            try:
                obj["confidence"] = min(max(float(obj.get("confidence")), 0.0), 1.0)
            except (TypeError, ValueError):
                obj["confidence"] = None
            if not isinstance(obj.get("differences_found"), list):
                obj["differences_found"] = []
            return obj
        start = text.find("{", start + 1)
    return None


def _norm(model):
    return (model or "").split("/")[-1].lower()


def check_served(spec, served):
    expected = _norm(spec.get("served_prefix") or spec["model_id"])
    if not _norm(served).startswith(expected):
        raise ModelMismatch(f"requested {spec['model_id']!r}, provider served {served!r}")


def _create(client, kwargs, sleep, tries=6):
    for i in range(tries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if type(exc).__name__ not in TRANSIENT or i == tries - 1:
                raise
            sleep(min(60, 2 * 2 ** i))


def execute_cell(client, spec, case, order, context, sleep=time.sleep):
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
        check_served(spec, resp.get("model"))
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


def _log_incident(out_dir, incident):
    with open(Path(out_dir) / INCIDENTS, "a", encoding="utf-8") as fh:
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
    done = set()
    if not path.exists():
        return done
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
                return done
            raise CorruptResults(f"{path}: unparseable line {i + 1}")
        key = (r["case_id"], r["order"])
        ok = (r.get("stimulus_sha256") == context["stimulus_sha256"] and key[0] in cases
              and r.get("payload_sha256") == render_messages(cases[key[0]], key[1])["payload_sha256"])
        if not ok:
            raise StaleRecords(f"{path}: stale record for {key}")
        mismatches = _spec_mismatches(r, spec, render_messages(cases[key[0]], key[1]))
        if mismatches and not accept_spec_change:
            raise StaleRecords(f"{path}: record for {key} differs from the current prompt or spec in "
                               f"{[m[0] for m in mismatches]}; rerun with --accept-spec-change to keep it")
        for field, recorded, current in mismatches:
            print(f"ACCEPTED spec change {path.name} {key}: {field}: recorded {recorded!r}, "
                  f"current {current!r}", flush=True)
            _log_incident(out_dir, {"kind": "accepted_spec_change", "model_id": spec["model_id"],
                                    "file": path.name, "case_id": key[0], "order": key[1], "field": field,
                                    "recorded": recorded, "current": current})
        if not is_error_only(r):
            done.add(key)
    return done


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
        done = _done(path, cases, context, spec, out_dir, accept_spec_change)
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
                    rec = execute_cell(client, spec, cases[cid], order, context, sleep=sleep)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    breaker.observe(rec)
    return path


def _clean_commit():
    dirty = subprocess.run(["git", "status", "--porcelain", "r2", "data/v2"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"commit r2/ and data/v2/ before running:\n{dirty}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _select(specs, selector):
    if selector == "all":
        return specs
    if selector.startswith("provider:"):
        return [s for s in specs if s["provider"] == selector.split(":", 1)[1]]
    names = set(selector.split(","))
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="default: RUNS_V2/main, or RUNS_V2/pilot with --pilot")
    parser.add_argument("--pilot", action="store_true",
                        help="pilot run: skip the freeze-tag and verified-stimulus checks")
    parser.add_argument("--accept-spec-change", action="store_true",
                        help="keep stored records whose prompt, model mapping or decoding differs from the "
                             "current spec; each accepted mismatch is logged to <out>/incidents.jsonl")
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = str(RUNS_V2 / ("pilot" if args.pilot else "main"))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.pilot and Path(args.out).resolve() == (RUNS_V2 / "main").resolve():
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
