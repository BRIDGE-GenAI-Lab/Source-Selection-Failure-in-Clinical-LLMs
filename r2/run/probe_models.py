"""One call per model, built and sent exactly as the harness sends a cell: is it served, and under
which identifier? The request is the harness's first attempt for one case (request_kwargs over the
render_messages output), sent through the harness's _create retry path."""
import csv
import time

from dotenv import load_dotenv

from r2.io import load_original_cases
from r2.paths import REPO, RUNS_V2
from r2.run.harness import _create, first_messages, load_specs, make_client, request_kwargs
from r2.stimuli.render import render_messages

# The probe needs only the request shape, which render_messages gives for any case; the original cases
# are always loadable, while pairs_v2.json is refused by load_v2_cases until every pair is resolved.
PROBE_CASE_ID, PROBE_ORDER = "CASE_001", "TS"


def probe_case():
    return load_original_cases()[0][PROBE_CASE_ID]


def probe(spec, client=None, sleep=time.sleep, case=None):
    try:
        client = client or make_client(spec["provider"])
        messages = first_messages(render_messages(case or probe_case(), PROBE_ORDER))
        resp = _create(client, request_kwargs(spec, messages), sleep).model_dump()
        return {"status": "available", "served": resp.get("model"),
                "fingerprint": resp.get("system_fingerprint"), "error": ""}
    except Exception as exc:
        return {"status": "error", "served": "", "fingerprint": "", "error": f"{type(exc).__name__}: {exc}"[:300]}


def main():
    load_dotenv(REPO / ".env")
    RUNS_V2.mkdir(parents=True, exist_ok=True)
    case = probe_case()
    rows = [{"original_name": s["original_name"], "provider": s["provider"], "model_id": s["model_id"],
             **probe(s, case=case)} for s in load_specs()]
    for path in (RUNS_V2 / "model_availability.csv", REPO / "analysis_repeated" / "model_availability.csv"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for r in rows:
        print(f"{r['status']:9s} {r['original_name']:52s} served={r['served']} {r['error'][:80]}")


if __name__ == "__main__":
    main()
