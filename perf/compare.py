#!/usr/bin/env python3
"""Judge a performance run against its budget and a reference run, and post the result as a
commit status the Vrixel SDLC reads (perf-smoke / report on a pull request, performance / report
after a staging deploy). Standard library only.

A run is a k6 summary (handleSummary JSON) or a flat JSON of metrics, e.g.
{"p95_ms": 120, "p99_ms": 240, "error_rate": 0.0, "rps": 18.5}. The reference is the image built
from main (pull requests) or the last passing staging run, kept on the perf-baselines branch.

usage: compare.py --run pr.json --kind api [--reference main.json | --baseline staging.json]
                  [--budget '{"p95_ms": 500}'] [--regression 20] [--context "perf-smoke / report"]
                  [--sha SHA] [--save-baseline staging.json]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

# Industry-standard starting budgets (the Vrixel SDLC's defaults; products tighten them).
BUDGETS = {
    "api": {"p95_ms": 500, "p99_ms": 1000, "error_rate": 0.01},
    "event-worker": {"lag_p95_s": 5, "error_rate": 0.01},
    "web": {"lcp_ms": 2500, "tbt_ms": 200, "cls": 0.1, "js_kb": 200},
    "mobile": {"cold_start_ms": 2000, "bundle_kb": 3000},
    "agent": {"turns_max": 8, "run_p95_s": 60},
}
# A change smaller than this is noise, whatever its percentage.
FLOORS = {
    "p95_ms": 10,
    "p99_ms": 20,
    "lag_p95_s": 0.5,
    "lcp_ms": 200,
    "tbt_ms": 50,
    "cls": 0.02,
    "js_kb": 5,
    "bundle_kb": 20,
    "cold_start_ms": 100,
    "turns_max": 1,
    "run_p95_s": 2,
    "tokens_per_run": 200,
}
UNITS = {
    "p95_ms": " ms",
    "p99_ms": " ms",
    "lag_p95_s": " s",
    "lcp_ms": " ms",
    "tbt_ms": " ms",
    "js_kb": " KB",
    "bundle_kb": " KB",
    "cold_start_ms": " ms",
    "run_p95_s": " s",
}


def metrics(data: dict | None) -> dict | None:
    """Flat metrics from a k6 summary (the measured scenario if tagged) or a flat JSON."""
    if not data:
        return None
    if "metrics" not in data:
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    m = data["metrics"]

    def pick(name: str) -> dict:
        # The measured part only, when the script tags it (warm-up excluded).
        return (m.get(f"{name}{{scenario:load}}") or m.get(name) or {}).get("values", {})

    dur, failed, reqs = pick("http_req_duration"), pick("http_req_failed"), pick("http_reqs")
    out = {
        "p95_ms": dur.get("p(95)"),
        "p99_ms": dur.get("p(99)"),
        "error_rate": failed.get("rate", 0.0),
        "rps": reqs.get("rate"),
    }
    return {k: float(v) for k, v in out.items() if v is not None}


def read(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def judge(run: dict, budget: dict, ref: dict | None, pct: float, against: str) -> list[str]:
    problems = []
    for metric, limit in budget.items():
        v = run.get(metric)
        if v is not None and v > float(limit):
            problems.append(f"{metric} {fmt(metric, v)} is over the budget of {fmt(metric, limit)}")
    for metric, a in (ref or {}).items():
        b = run.get(metric)
        if metric in ("rps",) or b is None or a is None:
            continue
        if metric == "error_rate":
            if b > a + 0.005:
                problems.append(f"error rate {b:.2%} is up from {a:.2%} on {against}")
            continue
        if a > 0 and b > a * (1 + pct / 100) and b - a > FLOORS.get(metric, 0):
            worse = 100 * (b - a) / a
            problems.append(
                f"{metric} {fmt(metric, b)} is {worse:.0f}% worse than {against} ({fmt(metric, a)})"
            )
    return problems


def fmt(metric: str, v: float) -> str:
    if metric in ("error_rate",):
        return f"{v:.2%}"
    return f"{v:,.0f}{UNITS.get(metric, '')}" if v >= 10 else f"{v:.2f}{UNITS.get(metric, '')}"


def describe(run: dict, ref: dict | None, budget: dict, against: str) -> str:
    """A one-line description (commit statuses take 140 characters)."""
    main = next(
        (m for m in ("p95_ms", "lag_p95_s", "lcp_ms", "js_kb", "bundle_kb", "turns_max") if m in run), None
    )
    if not main:
        return "no metrics"
    parts = [f"{main.split('_')[0]} {fmt(main, run[main])}"]
    if main in budget:
        parts.append(f"budget {fmt(main, budget[main])}")
    if ref and ref.get(main):
        parts.append(f"{against} {fmt(main, ref[main])}, {100 * (run[main] - ref[main]) / ref[main]:+.0f}%")
    text = f"{parts[0]} ({', '.join(parts[1:])})" if len(parts) > 1 else parts[0]
    if "error_rate" in run:
        text += f", errors {run['error_rate']:.1%}"
    return text[:140]


def github(method: str, path: str, body: dict | None = None) -> dict | None:
    token, api = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_API_URL", "https://api.github.com")
    if not token or not api.startswith("https://"):
        return None
    req = urllib.request.Request(  # noqa: S310 (https only, checked above)
        api + path,
        method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310 (https only)
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


BRANCH = "perf-baselines"


def load_baseline(repo: str, name: str) -> dict | None:
    """The last passing run, from the perf-baselines branch (none yet: no comparison)."""
    found = github("GET", f"/repos/{repo}/contents/{name}?ref={BRANCH}")
    return json.loads(base64.b64decode(found["content"])) if found and found.get("content") else None


def save_baseline(repo: str, name: str, run: dict, sha: str) -> None:
    """Keep this run as the next comparison, on the perf-baselines branch (created on first use)."""
    if github("GET", f"/repos/{repo}/git/ref/heads/{BRANCH}") is None:
        default = github("GET", f"/repos/{repo}")["default_branch"]
        head = github("GET", f"/repos/{repo}/git/ref/heads/{default}")["object"]["sha"]
        github("POST", f"/repos/{repo}/git/refs", {"ref": f"refs/heads/{BRANCH}", "sha": head})
    current = github("GET", f"/repos/{repo}/contents/{name}?ref={BRANCH}")
    body = {
        "message": f"Performance baseline from {sha[:7]}",
        "branch": BRANCH,
        "content": base64.b64encode(json.dumps({**run, "sha": sha}, indent=2).encode()).decode(),
    }
    if current and current.get("sha"):
        body["sha"] = current["sha"]
    github("PUT", f"/repos/{repo}/contents/{name}", body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--kind", default="api", choices=sorted(BUDGETS))
    ap.add_argument("--reference", help="the image built from main, run the same way")
    ap.add_argument("--baseline", help="name of the last passing run on the perf-baselines branch")
    ap.add_argument("--save-baseline", help="keep this run on the perf-baselines branch when it passes")
    ap.add_argument(
        "--budget", default="", help="JSON budget for this kind (the product's); defaults otherwise"
    )
    ap.add_argument("--regression", type=float, default=20.0, help="how much worse than the reference, in %%")
    ap.add_argument("--context", default="", help="commit status to post, e.g. 'perf-smoke / report'")
    ap.add_argument("--sha", default=os.environ.get("REPORT_SHA") or os.environ.get("GITHUB_SHA", ""))
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    a = ap.parse_args()

    run = metrics(read(a.run))
    if not run:
        print(f"{a.run}: no results", file=sys.stderr)
        problems, run, ref, against = ["the load test produced no results"], {}, None, "main"
    else:
        budget = {**BUDGETS[a.kind], **(json.loads(a.budget) if a.budget else {})}
        if a.baseline:
            ref, against = metrics(load_baseline(a.repo, a.baseline)), "the last release"
        else:
            ref, against = metrics(read(a.reference)), "main"
        problems = judge(run, budget, ref, a.regression, against)
    budget = {**BUDGETS[a.kind], **(json.loads(a.budget) if a.budget else {})}
    lines = [
        f"### {a.context or 'Performance'}: {'passed' if not problems else 'failed'}",
        "",
        "| Metric | This run | " + ("Reference | " if ref else "") + "Budget |",
        "| --- | --- | " + ("--- | " if ref else "") + "--- |",
    ]
    for metric in sorted(set(run) | set(budget)):
        lines.append(
            f"| {metric} | {fmt(metric, run[metric]) if metric in run else ''} | "
            + (f"{fmt(metric, ref[metric]) if metric in (ref or {}) else ''} | " if ref else "")
            + f"{fmt(metric, budget[metric]) if metric in budget else ''} |"
        )
    lines += [""] + [f"- {p}" for p in problems]
    if not ref:
        lines.append(f"\nNo reference run yet ({against}): judged against the budget only.")
    report = "\n".join(lines)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")
    if a.context and a.sha and a.repo:
        desc = "; ".join(problems)[:140] if problems else describe(run, ref, budget, against)
        github(
            "POST",
            f"/repos/{a.repo}/statuses/{a.sha}",
            {
                "state": "failure" if problems else "success",
                "context": a.context,
                "description": desc,
                "target_url": f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{a.repo}"
                f"/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}",
            },
        )
    if not problems and a.save_baseline and a.repo:
        save_baseline(a.repo, a.save_baseline, run, a.sha)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
