"""One small call per model: is it served, and under which identifier?"""
import csv

from dotenv import load_dotenv

from r2.paths import REPO, RUNS_V2
from r2.run.harness import _create, load_specs, make_client


def probe(spec):
    try:
        client = make_client(spec["provider"])
        kwargs = {"model": spec["model_id"],
                  "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                  spec.get("token_param", "max_tokens"): 512}
        resp = _create(client, kwargs, sleep=lambda s: None, tries=2).model_dump()
        return {"status": "available", "served": resp.get("model"),
                "fingerprint": resp.get("system_fingerprint"), "error": ""}
    except Exception as exc:
        return {"status": "error", "served": "", "fingerprint": "", "error": f"{type(exc).__name__}: {exc}"[:300]}


def main():
    load_dotenv(REPO / ".env")
    RUNS_V2.mkdir(parents=True, exist_ok=True)
    rows = [{"original_name": s["original_name"], "provider": s["provider"], "model_id": s["model_id"], **probe(s)}
            for s in load_specs()]
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
