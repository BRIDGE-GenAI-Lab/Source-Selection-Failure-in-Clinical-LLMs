"""Unified runner for the repeated experiment: one prompt (r2.stimuli.render), both orders,
one code path for every provider (OpenAI-compatible endpoints), full request/response logs."""
import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from r2.paths import DATA_V2, FREEZE_REPO, MODELS_FILE, REPO, RUNS_V2, SANDBOX, VERIFY_DIR
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
# Accepted selected_tool values (after strip, any case): "A"/"B", or "Excerpt|Option|Tool|Source A|B".
# Anything else is not a decision (non_response). The value as sent is kept in selected_raw.
SELECTED = re.compile(r"(?:(?:excerpt|option|tool|source)\s+)?([ab])", re.I)
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
# Error texts are compared by error_key: the text with volatile tokens replaced (normalize_error), cut to
# ERROR_TEXT_CHARS. The raw text is stored unchanged in "error"; the key is stored as "error_key".
_HEX = "0-9a-fA-F"
ERROR_NORMALIZERS = (
    (re.compile(rf"(?<![{_HEX}])[{_HEX}]{{8}}-[{_HEX}]{{4}}-[{_HEX}]{{4}}-[{_HEX}]{{4}}-[{_HEX}]{{12}}(?![{_HEX}])"),
     "<uuid>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"), "<ts>"),
    (re.compile(r"\breq_[A-Za-z0-9]+"), "<req>"),
    (re.compile(r"""(\brequest[ _-]?id["']?\s*[:=]?\s*["']?)(?!<)(?=[A-Za-z_-]*\d)[A-Za-z0-9_-]+""", re.I),
     r"\1<id>"),  # a value with at least one digit, so "request ID in your message" is left alone
    (re.compile(rf"(?<![0-9A-Za-z])[{_HEX}]{{8,}}(?![0-9A-Za-z])"), "<hex>"),
    (re.compile(r"\d{6,}"), "<n>"),
)


def normalize_error(text):
    """The error text with UUIDs, ISO timestamps, req_ ids, request_id / request id values, hex runs of 8 or
    more characters and digit runs of 6 or more replaced by fixed tokens."""
    for pattern, token in ERROR_NORMALIZERS:
        text = pattern.sub(token, text)
    return text


def error_key(text):
    return normalize_error(text)[:ERROR_TEXT_CHARS]


def _error_kind(exc):
    name = type(exc).__name__
    if name in CELL_ERRORS:
        return "cell"
    if name in MODEL_ERRORS:
        return "model"
    if name in TRANSIENT:
        return "transient"  # reaches execute_cell only after _create has used all its tries
    return None


def load_specs(path=None):
    """The model specs of r2/run/models_v2.json (MODELS_FILE: a copy inside R2_SANDBOX when sandboxed)."""
    path = MODELS_FILE if path is None else path
    specs = json.loads(Path(path).read_text())["models"]
    keys = {"model_id": lambda s: s["model_id"], "safe_name(model_id)": lambda s: safe_name(s["model_id"]),
            "original_name": lambda s: s["original_name"]}
    for label, key in keys.items():
        dup = sorted(k for k, n in Counter(key(s) for s in specs).items() if n > 1)
        if dup:
            raise ValueError(f"{path}: duplicate {label}: {dup}")
    return specs


def load_v2_cases(path=None):
    """(cases by case_id, sha256 of the file bytes) of `path` (default DATA_V2/pairs_v2.json, read at call time).
    Raises ValueError, never assert, on any defect."""
    path = DATA_V2 / "pairs_v2.json" if path is None else path
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"{path}: expected an object with a 'cases' list")
    bad = [i for i, c in enumerate(data["cases"])
           if not (isinstance(c, dict) and isinstance(c.get("case_id"), str) and isinstance(c.get("tools"), dict)
                   and isinstance(c["tools"].get("T"), dict) and isinstance(c["tools"].get("S"), dict))]
    if bad:
        raise ValueError(f"{path}: cases at positions {bad[:10]} lack a case_id or tools T and S")
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


def load_keys(env_file=None):
    """Put the provider keys of REPO/.env into the environment. The file wins over an exported variable (a stale
    exported key, such as a revoked legacy key, would otherwise be used without a word); a differing exported value
    is reported. Prints where each provider key comes from, never any part of a key. Returns {variable: source}."""
    from dotenv import dotenv_values
    path = Path(REPO) / ".env" if env_file is None else Path(env_file)
    values = {k: v for k, v in (dotenv_values(path) if path.exists() else {}).items() if v}
    sources = {}
    for name in [cfg["key_env"] for cfg in PROVIDERS.values()]:
        if name in values:
            if os.environ.get(name) and os.environ[name] != values[name]:
                print(f"WARNING: {name} is also exported in the environment with a different value; the value in "
                      f"{path} is used", flush=True)
            os.environ[name] = values[name]
            sources[name] = str(path)
        else:
            sources[name] = "environment" if os.environ.get(name) else "missing"
        print(f"key {name}: {sources[name]}", flush=True)
    return sources


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


def is_unresolved(r):
    """A cell's last record that is an error-only record not yet re-attempted to finality: the cell needs a
    resume (one re-attempt) before it is complete."""
    return is_error_only(r) and r.get("persistent_error") is not True


def unresolved_cells(path):
    return sorted((r["case_id"], r["order"]) for r in final_records(path) if is_unresolved(r))


def _error_key(r):
    """The comparison key of a record's error, recomputed from the raw text (the stored error_key is for audit)."""
    return error_key(r["error"])


def read_records(path):
    """Every record of a results file. Lines are split on b"\\n" only: str.splitlines() would also split
    on U+2028, U+2029 and NEL, which json.dumps(ensure_ascii=False) writes raw inside model text."""
    out = []
    for i, line in enumerate(Path(path).read_bytes().split(b"\n"), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorruptResults(f"{path}: line {i} is unparseable ({type(exc).__name__}); a torn final line is "
                                 "repaired when the harness resumes the model: resume the model first") from exc
    return out


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
        key = _error_key(rec)
        self.same = self.same + 1 if key == self.text else 1
        self.text = key
        if self.consecutive >= BREAKER_LIMIT or self.same >= BREAKER_LIMIT:
            raise ModelUnavailable(
                f"{self.spec['model_id']}: circuit breaker ({self.consecutive} consecutive cell errors, "
                f"same error text {self.same}x in a row): {rec['error'][:ERROR_TEXT_CHARS]}")


def normalize_selected(value):
    """'A' or 'B' for an accepted selected_tool value (see SELECTED), else None."""
    if not isinstance(value, str):
        return None
    m = SELECTED.fullmatch(value.strip())
    return m.group(1).upper() if m else None


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
        sel = normalize_selected(obj.get("selected_tool")) if isinstance(obj, dict) else None
        if sel is not None:
            obj["selected_raw"] = obj["selected_tool"]
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


def first_messages(rendered):
    """The messages of a cell's first attempt, from render_messages output."""
    return [{"role": "system", "content": rendered["system"]}, {"role": "user", "content": rendered["user"]}]


def request_kwargs(spec, messages):
    """The chat.completions.create arguments for a spec: the only builder (harness and probe)."""
    kwargs = {"model": spec["model_id"], "messages": messages,
              spec.get("token_param", "max_tokens"): spec["max_tokens"]}
    if spec.get("temperature") is not None:
        kwargs["temperature"] = spec["temperature"]
    return kwargs


def execute_cell(client, spec, case, order, context, sleep=time.sleep, out_dir=None):
    msgs = render_messages(case, order)
    attempts = []
    try:
        decision, error, failed_request = _attempt_cell(client, spec, case, order, context, sleep, out_dir, msgs,
                                                        attempts)
    except Exception as exc:  # a stop after a paid attempt: hand the attempts to the stop incident
        try:
            exc.attempts = attempts
        except AttributeError:  # an exception type without instance attributes: the attempts are lost
            pass
        raise
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
        rec["error"] = error  # raw, unchanged
        rec["error_key"] = error_key(error)  # what the persistent-error rule and the breaker compare
        rec["request"] = failed_request  # the call that failed (error records carry no response)
    return rec


def _attempt_cell(client, spec, case, order, context, sleep, out_dir, msgs, attempts):
    """Up to two calls for one cell, appending each response to `attempts`. Returns
    (decision, error, failed_request)."""
    messages = first_messages(msgs)
    decision, error, failed_request = None, None, None
    for _ in range(2):
        kwargs = request_kwargs(spec, messages)
        t0 = time.time()
        try:
            resp = _create(client, kwargs, sleep).model_dump()
        except Exception as exc:
            kind = _error_kind(exc)
            if kind == "model":
                raise ModelUnavailable(f"{spec['model_id']}: {type(exc).__name__}: {exc}"[:500]) from exc
            if kind == "transient":  # e.g. a quota 429: stop the model (resumable), not the provider job
                raise ModelUnavailable(f"{spec['model_id']}: {type(exc).__name__} persisted after {RETRY_TRIES} "
                                       f"tries: {exc}"[:500]) from exc
            if kind == "cell":
                error, failed_request = f"{type(exc).__name__}: {exc}"[:500], kwargs
                break
            raise
        latency_ms = (time.time() - t0) * 1000
        try:
            check_served(spec, resp.get("model"))
        except ModelMismatch as exc:
            if out_dir is not None:  # the paid responses are the drift evidence: keep them before stopping
                # (no cell record is written, so earlier attempts of this cell are logged here too)
                _log_incident(out_dir, spec, {"kind": "model_mismatch", "model_id": spec["model_id"],
                                        "original_name": spec["original_name"], "case_id": case["case_id"],
                                        "order": order, "error": str(exc), "request": kwargs, "response": resp,
                                        "latency_ms": latency_ms, "attempts": attempts, **context})
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
    return decision, error, failed_request


def _log_incident(out_dir, spec, incident):
    """Append one incident to the model's own incident file. Callers hold the model lock (run_model)."""
    with open(incidents_path(spec, out_dir), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({**incident, "logged_at": datetime.now(timezone.utc).isoformat()},
                            ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _sha(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


NEVER_ACCEPTED = ("system", "user", "model_id", "original_name")  # --accept-spec-change keeps decoding changes only


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
    if is_unresolved(r):
        return out  # re-attempted on this resume anyway; a persistent_error record is checked like any other
    attempts = r.get("attempts") or []
    req = ((attempts[0] or {}).get("request") if attempts else r.get("request")) or {}
    temp = spec.get("temperature")
    if ("temperature" in req) != (temp is not None) or (temp is not None and req["temperature"] != temp):
        out.append(("temperature", req.get("temperature", "<absent>"), "<absent>" if temp is None else temp))
    token_param = spec.get("token_param", "max_tokens")
    sent = {k: req[k] for k in TOKEN_PARAMS if k in req}
    if sent != {token_param: spec["max_tokens"]}:
        out.append(("token_param", sent, {token_param: spec["max_tokens"]}))
    return out


def _done(path, cases, context, spec, out_dir, accept_spec_change=False, retry_persistent=False):
    """Validate every stored record; the LAST record of each cell decides its state. Returns
    (done, error_texts, reopen):
    - done: cells whose last record completes them (is_done), except persistent cells under retry_persistent;
    - error_texts: per cell, the error_key of every error-only record (persistent rule);
    - reopen: with retry_persistent, {cell: (prior raw error text, 1-based line)} for each cell whose last
      record is a persistent_error record."""
    last, error_texts = {}, {}
    lines = path.read_bytes().split(b"\n") if path.exists() else []
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
                _log_incident(out_dir, spec, {"kind": "torn_tail", "model_id": spec["model_id"], "file": path.name,
                                              "line": i + 1, "offset": start, "length": len(line),
                                              "dropped_b64": base64.b64encode(line).decode("ascii")})
                os.truncate(path, start)  # keeps every earlier byte; nothing is rewritten
                break
            raise CorruptResults(f"{path}: unparseable line {i + 1}")
        key = (r["case_id"], r["order"])
        msgs = render_messages(cases[key[0]], key[1]) if key[0] in cases else None
        ok = (r.get("stimulus_sha256") == context["stimulus_sha256"] and msgs is not None
              and r.get("payload_sha256") == msgs["payload_sha256"])
        if not ok:
            raise StaleRecords(f"{path}: stale record for {key}")
        mismatches = _spec_mismatches(r, spec, msgs)
        fixed = [m[0] for m in mismatches if m[0] in NEVER_ACCEPTED]
        if fixed:  # a response to another prompt, or of another model: never kept, whatever the flag
            raise StaleRecords(f"{path}: record for {key} differs from the current prompt or model mapping in "
                               f"{fixed}; a response to another input is never kept (--accept-spec-change covers "
                               "decoding changes only): restore the spec, or move the file aside under the "
                               "substitution rule (docs/r2/RUNBOOK.md)")
        if mismatches and not accept_spec_change:
            raise StaleRecords(f"{path}: record for {key} differs from the current prompt or spec in "
                               f"{[m[0] for m in mismatches]}; rerun with --accept-spec-change to keep it")
        for field, recorded, current in mismatches:
            print(f"ACCEPTED spec change {path.name} {key}: {field}: recorded {recorded!r}, "
                  f"current {current!r}", flush=True)
            _log_incident(out_dir, spec, {"kind": "accepted_spec_change", "model_id": spec["model_id"],
                                    "file": path.name, "case_id": key[0], "order": key[1], "field": field,
                                    "recorded": recorded, "current": current})
        last[key] = (r, i + 1)
        if is_error_only(r):
            error_texts.setdefault(key, set()).add(_error_key(r))
    done, reopen = set(), {}
    for key, (r, line_no) in last.items():
        if retry_persistent and r.get("persistent_error") is True:
            reopen[key] = (r["error"], line_no)
        elif is_done(r):
            done.add(key)
    return done, error_texts, reopen


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


def run_model(spec, cases, out_dir, client, context, limit=None, sleep=time.sleep, accept_spec_change=False,
              retry_persistent=False):
    """Run every not-yet-done cell of the first `limit` case_ids (all when limit is None; 0 runs none).
    Fresh (never-attempted) cells run before re-attempts. With retry_persistent, cells whose last record is
    persistent_error are re-attempted too, each after a reopened_persistent incident."""
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0 or None, got {limit}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = results_path(spec, out_dir)
    with _model_lock(out_dir, spec):
        cell = None  # the cell being attempted when a stop is raised (None: before any cell)
        try:
            done, error_texts, reopen = _done(path, cases, context, spec, out_dir, accept_spec_change,
                                              retry_persistent)
            ids = sorted(cases) if limit is None else sorted(cases)[:limit]
            todo = [(cid, order) for cid in ids for order in ORDERS if (cid, order) not in done]
            # Fresh cells first: a provider-wide error then trips the breaker on fresh cells before any
            # re-attempt can reproduce its text and be finalized.
            todo = [c for c in todo if c not in error_texts] + [c for c in todo if c in error_texts]
            if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write("\n")
            breaker = _Breaker(spec)
            with open(path, "a", encoding="utf-8") as fh:
                for cell in todo:
                    cid, order = cell
                    if cell in reopen:
                        prior, line_no = reopen[cell]
                        print(f"REOPEN {spec['original_name']} {cid} {order}: persistent_error at line {line_no}: "
                              f"{prior[:ERROR_TEXT_CHARS]}", flush=True)
                        _log_incident(out_dir, spec, {"kind": "reopened_persistent", "model_id": spec["model_id"],
                                                      "original_name": spec["original_name"], "case_id": cid,
                                                      "order": order, "prior_error": prior,
                                                      "prior_error_key": error_key(prior),
                                                      "prior_line": line_no, **context})
                    rec = execute_cell(client, spec, cases[cid], order, context, sleep=sleep,
                                       out_dir=out_dir)
                    earlier = error_texts.get(cell)
                    if earlier and is_error_only(rec) and _error_key(rec) in earlier:
                        rec["persistent_error"] = True
                        print(f"FINAL {spec['original_name']} {cid} {order}: same error as an earlier "
                              f"attempt, recorded as persistent_error: {rec['error'][:ERROR_TEXT_CHARS]}", flush=True)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    if rec.get("persistent_error"):
                        _log_incident(out_dir, spec, {"kind": "persistent_error", "model_id": spec["model_id"],
                                                      "original_name": spec["original_name"], "case_id": cid,
                                                      "order": order, "error": rec["error"],
                                                      "error_key": rec["error_key"], **context})
                    if not earlier:  # re-attempted cells count toward neither breaker rule
                        breaker.observe(rec)
        except ModelMismatch:
            raise  # execute_cell has already logged a model_mismatch incident with the served response
        except Exception as exc:
            _log_incident(out_dir, spec, {"kind": "stopped", "model_id": spec["model_id"],
                                          "original_name": spec["original_name"],
                                          "case_id": cell[0] if cell else None,
                                          "order": cell[1] if cell else None,
                                          "error_type": type(exc).__name__, "error": str(exc)[:2000],
                                          "attempts": getattr(exc, "attempts", []), **context})
            raise
    return path


CODE_PATHS = ("r2", "data/v2", "tests")  # must be committed: harness_commit identifies all of them


def _sandbox_commit():
    """Under R2_SANDBOX: "sandbox:code-<16 hex>", a SHA-256 over the paths and bytes of r2/**/*.py and
    tests/**/*.py. It runs no git: a sandbox runs no git command outside itself (r2.sandbox_git). The sandbox
    stimuli are not in the repository, so data/v2 is not included; nothing here is ever analysed as data."""
    h = hashlib.sha256()
    for top in ("r2", "tests"):
        for p in sorted((REPO / top).rglob("*.py")):
            if p.name.startswith("._") or "__pycache__" in p.parts:
                continue
            h.update(p.relative_to(REPO).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return f"sandbox:code-{h.hexdigest()[:16]}"


def _clean_commit(main=False):
    """HEAD of the repository, which must have no uncommitted change under CODE_PATHS (and, for a main run, under
    MAIN_EXTRA_PATHS: verification/ and docs/r2/FREEZE_RECORD.md). Under R2_SANDBOX: _sandbox_commit()."""
    if SANDBOX is not None:
        return _sandbox_commit()
    where = CODE_PATHS + (MAIN_EXTRA_PATHS if main else ())
    status = subprocess.run(["git", "status", "--porcelain", *where], cwd=REPO,
                            capture_output=True, text=True)
    if status.returncode != 0:
        raise SystemExit(f"git status failed (rc {status.returncode}); cannot record harness_commit")
    if status.stdout.strip():
        names = ", ".join(f"{p}/" if "." not in Path(p).name else p for p in where)
        raise SystemExit(f"commit {names} before running:\n{status.stdout.strip()}")
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


def _freeze_git(*args, repo=None):
    """git in FREEZE_REPO (the repository itself, or the sandbox's freeze repository under R2_SANDBOX, checked
    through r2.sandbox_git: no inherited GIT_* variables, and the repository must lie in the sandbox). Read-only
    uses only. A missing sandbox repository is a failed command (exit status 128)."""
    repo = FREEZE_REPO if repo is None else repo
    if SANDBOX is not None:
        from r2 import sandbox_git
        if not Path(repo).exists():
            return subprocess.CompletedProcess(args, 128, "", f"{repo} does not exist")
        try:
            return sandbox_git.run(repo, *args, sandbox=SANDBOX)
        except sandbox_git.SandboxGitError as exc:
            raise SystemExit(f"refusing the main run: {exc}") from exc
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def _tag_exists(tag, repo=None):
    """The tag exists in FREEZE_REPO: the repository itself, or the sandbox's freeze repository under R2_SANDBOX."""
    return _freeze_git("rev-parse", "-q", "--verify", f"refs/tags/{tag}", repo=repo).returncode == 0


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


FREEZE_RECORD_REL = "docs/r2/FREEZE_RECORD.md"
FREEZE_INCIDENTS = "freeze.incidents.jsonl"  # run-level main-run incidents: unpublished_freeze, frozen_input_changed
MAIN_EXTRA_PATHS = ("verification", FREEZE_RECORD_REL)  # also committed before a main run (exclusions, flags, record)


def frozen_sources():
    """Each frozen file (r2.freeze.HASHED) -> the file the main run uses: the repository's, or under R2_SANDBOX the
    sandbox copies of the state (stimuli, exclusions, model specs) and the repository's code."""
    return {"data/v2/pairs_v2.json": Path(DATA_V2) / "pairs_v2.json",
            "r2/stimuli/render.py": Path(REPO) / "r2" / "stimuli" / "render.py",
            "verification/excluded_pairs.json": Path(VERIFY_DIR) / "excluded_pairs.json",
            "r2/run/models_v2.json": Path(MODELS_FILE),
            "r2/run/harness.py": Path(REPO) / "r2" / "run" / "harness.py"}


def _log_run_incident(out_dir, incident):
    """Append one run-level incident to <out_dir>/freeze.incidents.jsonl (never a result file)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    line = {**incident, "logged_at": datetime.now(timezone.utc).isoformat()}
    if SANDBOX is not None:
        line["synthetic"] = True
    with open(out_dir / FREEZE_INCIDENTS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_incidents(out_dir):
    """The run-level incidents of <out_dir>/freeze.incidents.jsonl (an empty list when there are none)."""
    path = Path(out_dir) / FREEZE_INCIDENTS
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def verify_freeze(stim_sha, accept_unpublished=False):
    """Every main run (a direct python -m r2.run.harness call included) refuses with SystemExit("refusing the main
    run: ...") unless:
    1. the tag r2-analysis-plan-frozen exists in FREEZE_REPO;
    2. verification/excluded_pairs.json records pairs_v2_sha256 = the stimulus file's SHA-256;
    3. FREEZE_RECORD.md at the tag holds the five frozen hashes, and the current FREEZE_RECORD.md keeps those lines;
    4. each frozen file's current SHA-256 equals its frozen hash, or a dated entry under "Changes after the freeze"
       of the current FREEZE_RECORD.md, dated from the freeze day to today, gives that new hash (an accepted change);
    5. FREEZE_RECORD.md records the public timestamp of the freeze, or accept_unpublished.
    Writes nothing. Returns {"public": the public timestamp or None, "changed": [frozen_input_changed incidents],
    "freeze_record": path}; log_freeze() prints and logs them once the run starts."""
    from r2 import freeze as F
    if not _tag_exists(FREEZE_TAG):
        raise SystemExit(f"refusing the main run: git tag {FREEZE_TAG} does not exist "
                         "(use --pilot for a pilot run)")
    path = Path(VERIFY_DIR) / "excluded_pairs.json"
    verified = json.loads(path.read_text()).get("pairs_v2_sha256") if path.exists() else None
    if verified != stim_sha:
        raise SystemExit(f"refusing the main run: data/v2/pairs_v2.json has sha256 {stim_sha}, but "
                         f"{path} records pairs_v2_sha256 = {verified} (the physician-verified stimuli)")
    at_tag = _freeze_git("show", f"{FREEZE_TAG}:{FREEZE_RECORD_REL}")
    if at_tag.returncode:
        raise SystemExit(f"refusing the main run: no {FREEZE_RECORD_REL} at the tag {FREEZE_TAG} "
                         f"({at_tag.stderr.strip()})")
    record_path = Path(FREEZE_REPO) / FREEZE_RECORD_REL
    current = record_path.read_text(encoding="utf-8") if record_path.exists() else ""
    try:
        frozen = F.frozen_hashes(at_tag.stdout)
        if F.frozen_hashes(current) != frozen:
            raise SystemExit(f"refusing the main run: {record_path}: its hash lines differ from FREEZE_RECORD.md at the "
                             f"tag {FREEZE_TAG}; the frozen hashes are never edited (record a change as a dated entry "
                             f"under '{F.CHANGES_HEAD}')")
    except F.FreezeRefused as exc:
        raise SystemExit(f"refusing the main run: {exc}") from exc
    freeze_day, today = F.record_date(at_tag.stdout), _today()
    entries = F.change_entries(current)
    changed, problems = [], []
    for rel, src in frozen_sources().items():
        digest = hashlib.sha256(src.read_bytes()).hexdigest() if src.exists() else None
        if digest == frozen[rel]:
            continue
        entry = next((e for e in reversed(entries) if e["path"] == rel and e["sha256"] == digest), None)
        if entry is None:
            problems.append(f"{rel} has sha256 {digest}, but FREEZE_RECORD.md at the tag records {frozen[rel]} and "
                            f"no dated entry under '{F.CHANGES_HEAD}' gives the new hash")
        elif freeze_day and entry["date"] < freeze_day:
            problems.append(f"{rel}: the change entry is dated {entry['date']}, before the freeze ({freeze_day})")
        elif entry["date"] > today:
            problems.append(f"{rel}: the change entry is dated {entry['date']}, after today ({today})")
        else:
            changed.append({"kind": "frozen_input_changed", "path": rel, "frozen_sha256": frozen[rel],
                            "current_sha256": digest, "entry_date": entry["date"], "reason": entry["reason"]})
    if problems:
        raise SystemExit("refusing the main run: " + "; ".join(problems))
    public = F.public_timestamp(current)
    if public is None and not accept_unpublished:
        raise SystemExit(f"refusing the main run: {record_path} records no public timestamp of the freeze. Pushing the "
                         "tag needs the authors' approval; once the freeze is public (the pushed tag, an OSF "
                         "registration or a Zenodo DOI), run python -m r2.pipeline record-public <url>, commit "
                         f"{FREEZE_RECORD_REL} and rerun. The only override is --accept-unpublished-freeze, which "
                         "writes an incident line")
    return {"public": public, "changed": changed, "freeze_record": str(record_path)}


def log_freeze(checked, out_dir):
    """Print and log (to <out_dir>/freeze.incidents.jsonl) what verify_freeze accepted: one frozen_input_changed
    incident per changed frozen file, and an unpublished_freeze incident when the freeze is not public."""
    for inc in checked["changed"]:
        print(f"FROZEN INPUT CHANGED {inc['path']}: {inc['frozen_sha256'][:12]} -> {inc['current_sha256'][:12]} "
              f"(entry of {inc['entry_date']}: {inc['reason']})", flush=True)
        _log_run_incident(out_dir, inc)
    if checked["public"] is None:
        _log_run_incident(out_dir, {"kind": "unpublished_freeze", "freeze_record": checked["freeze_record"],
                                    "note": "main run started before the freeze was public"})
        print("ACCEPTED: main run without a public timestamp of the freeze (--accept-unpublished-freeze); incident "
              f"written to {Path(out_dir) / FREEZE_INCIDENTS}", flush=True)


def _same_dir(a, b):
    """True when a and b name the same directory: os.path.samefile when both exist, else the resolved
    paths compared case-insensitively (the run volume is case-insensitive exFAT)."""
    a, b = Path(a), Path(b)
    if a.exists() and b.exists():
        return os.path.samefile(a, b)
    return str(a.resolve()).casefold() == str(b.resolve()).casefold()


def _non_negative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--limit", type=_non_negative_int, default=None,
                        help="run only the first N pairs (sorted case_id); 0 runs none; omit for all")
    parser.add_argument("--out", default=None, help="default: RUNS_V2/main, or RUNS_V2/pilot with --pilot")
    parser.add_argument("--pilot", action="store_true",
                        help="pilot run: skip the freeze checks (tag, verified stimuli, frozen hashes, public "
                             "timestamp)")
    parser.add_argument("--accept-spec-change", action="store_true",
                        help="keep stored records whose decoding (temperature, token parameter or limit) differs "
                             "from the current spec; each accepted mismatch is logged to "
                             "<out>/<safe_name>.incidents.jsonl. A changed prompt or model mapping is never kept")
    parser.add_argument("--accept-unpublished-freeze", action="store_true",
                        help="main run although FREEZE_RECORD.md records no public timestamp of the freeze; writes an "
                             "unpublished_freeze incident to <out>/freeze.incidents.jsonl")
    parser.add_argument("--retry-persistent", action="store_true",
                        help="re-attempt cells whose last record is persistent_error (each logged as a "
                             "reopened_persistent incident); use after a provider-wide condition is fixed")
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = str(RUNS_V2 / ("pilot" if args.pilot else "main"))
    return args


def main(argv=None, client_factory=None, sleep=None):
    """client_factory(provider) -> an OpenAI-compatible client; default make_client (the real API, keys from
    REPO/.env or the environment). Under R2_SANDBOX a factory must be injected (r2.rehearse): the sandbox never
    calls a real API, and every record it writes carries "synthetic": true."""
    args = parse_args(argv)
    if SANDBOX is not None and client_factory is None:
        raise SystemExit(f"refusing: R2_SANDBOX={SANDBOX} is set and no client was injected; a sandbox run never "
                         "calls a real API (run it through r2.rehearse, or unset R2_SANDBOX for the real run)")
    if args.pilot and _same_dir(args.out, RUNS_V2 / "main"):
        raise SystemExit("refusing a --pilot run into the main out dir")
    cases, stim_sha = load_v2_cases()
    checked = None if args.pilot else verify_freeze(stim_sha, accept_unpublished=args.accept_unpublished_freeze)
    context = {"harness_commit": _clean_commit(main=not args.pilot), "stimulus_sha256": stim_sha}
    if checked is not None:  # every check passed: now the incidents (accepted changes, an unpublished freeze)
        log_freeze(checked, args.out)
    if SANDBOX is not None:
        context["synthetic"] = True
    if client_factory is None:
        load_keys()
        client_factory = make_client
    specs, stopped, need_resume = _select(load_specs(), args.models), [], 0
    for spec in specs:
        name = spec["original_name"]
        print(f"[{datetime.now():%H:%M:%S}] {name}", flush=True)
        locked = False
        try:
            run_model(spec, cases, args.out, client_factory(spec["provider"]), context, limit=args.limit,
                      accept_spec_change=args.accept_spec_change, retry_persistent=args.retry_persistent,
                      **({} if sleep is None else {"sleep": sleep}))
        except (ModelMismatch, ModelUnavailable, RunLocked, StaleRecords, CorruptResults) as exc:
            stopped.append(name)
            locked = isinstance(exc, RunLocked)
            print(f"STOPPED {name}: {type(exc).__name__}: {exc}", flush=True)
        except Exception as exc:  # never let one model end the provider job
            stopped.append(name)
            print(f"STOPPED {name}: unexpected {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        path = results_path(spec, args.out)
        if locked or not path.exists():
            continue  # a locked model belongs to another process; its file is not read here
        try:
            pending = unresolved_cells(path)
        except (CorruptResults, ValueError):
            continue  # an unreadable file: the model is already reported as STOPPED
        if pending:
            need_resume += len(pending)
            print(f"PENDING {name}: {len(pending)} cells end in an error and need a resume to be re-attempted: "
                  f"{pending[:20]}{' ...' if len(pending) > 20 else ''}", flush=True)
    print(f"FINISHED {len(specs)} model(s); stopped {len(stopped)}: {stopped}; {need_resume} cells need a resume",
          flush=True)
    return {"stopped": stopped, "need_resume": need_resume}


def cli(argv=None, **kwargs):
    """Exit status 1 when any model stopped or any cell needs a resume, else 0. kwargs go to main()."""
    summary = main(argv, **kwargs)
    return 1 if summary["stopped"] or summary["need_resume"] else 0


if __name__ == "__main__":
    raise SystemExit(cli())
