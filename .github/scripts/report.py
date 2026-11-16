#!/usr/bin/env python3
"""Post a test suite's results as a commit status the Valxel SDLC reads.

Managed by the Valxel SDLC (config/repos.yaml managed_files): `valxel repos check` reports a copy
that differs, and `valxel repos apply` opens a pull request to bring it back. Change it in
agentic-sdlc (src/valxel_sdlc/standard_files/report.py), not here.

The pipeline's quality policy matches suites by check name (unit*, contract*, integration*, e2e*,
smoke*, code-quality*) and reads counts and coverage from the summary text, e.g.
"128 passed, 0 failed, 2 skipped. Coverage: 83.5%". This script builds that text from JUnit XML
(any test runner) and a coverage report (JaCoCo XML, Istanbul coverage-summary.json or Cobertura
XML), and posts it as a commit status. Standard library only; runs on any GitHub-hosted runner.

usage: report.py --context "unit / report" --junit 'build/test-results/test/*.xml'
                 [--jacoco jacocoTestReport.xml | --istanbul coverage-summary.json | --cobertura coverage.xml]
                 [--sha SHA] [--repo owner/name] [--url URL] [--min-pass-rate 0.95]

       report.py --context "code-quality / new code" --new-code new-code.json [--min-new-coverage 80]

--min-pass-rate: the status is a success when at least this share of the tests passed (a live run
of an agent's evals against the real model can miss a case or two by chance). The summary always
gives the counts, so the Valxel SDLC sees the rate either way.

--new-code: diff-cover's JSON report (`diff-cover <coverage> --compare-branch origin/<base>
--format json:new-code.json`), i.e. coverage of the lines this pull request changes. The status
fails below --min-new-coverage, and names the lines that no test runs.
"""

import argparse
import glob
import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET


def junit(patterns: list[str]) -> dict:
    total = failed = skipped = 0
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            root = ET.parse(path).getroot()
            for suite in ([root] if root.tag == "testsuite" else root.iter("testsuite")):
                total += int(suite.get("tests", 0))
                failed += int(suite.get("failures", 0)) + int(suite.get("errors", 0))
                skipped += int(suite.get("skipped", 0))
    return {"passed": total - failed - skipped, "failed": failed, "skipped": skipped}


def jacoco(path: str) -> float | None:
    if not path or not os.path.exists(path):
        return None
    root = ET.parse(path).getroot()
    for c in root.findall("counter"):  # report-level counters
        if c.get("type") == "LINE":
            covered, missed = int(c.get("covered")), int(c.get("missed"))
            return 100.0 * covered / max(1, covered + missed)
    return None


def cobertura(path: str) -> float | None:
    """coverage.py / pytest-cov XML (and other Cobertura producers)."""
    if not path or not os.path.exists(path):
        return None
    return 100.0 * float(ET.parse(path).getroot().get("line-rate", 0))


def istanbul(path: str) -> float | None:
    if not path or not os.path.exists(path):
        return None
    return float(json.load(open(path))["total"]["lines"]["pct"])


def spans(lines: list[int]) -> str:
    """[3, 4, 5, 9] -> "3-5, 9"."""
    out, start, prev = [], None, None
    for n in sorted(set(lines)):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev != start else f"{start}")
    return ", ".join(out)


def new_code(path: str, minimum: float) -> tuple[str, str, str]:
    """(state, short text for the status, the full list for the job summary)."""
    if not path or not os.path.exists(path):
        return "failure", "No coverage of new code: diff-cover's report is missing.", ""
    report = json.load(open(path))
    total = int(report.get("total_num_lines") or 0)
    missed = int(report.get("total_num_violations") or 0)
    pct = 100.0 if total == 0 else 100.0 * (total - missed) / total
    files = {f: s.get("violation_lines") or [] for f, s in (report.get("src_stats") or {}).items()}
    files = {f: v for f, v in files.items() if v}
    text = f"Coverage on new code: {pct:.1f}% (at least {minimum:g}%)."
    if total == 0:
        text = f"Coverage on new code: 100% (no coverable lines changed; at least {minimum:g}%)."
    elif missed:
        text += f" {missed} of {total} changed lines untested: " + "; ".join(
            f"{os.path.basename(f)} {spans(v)}" for f, v in sorted(files.items()))
    detail = "\n".join(f"- `{f}`: lines {spans(v)}" for f, v in sorted(files.items()))
    return ("success" if pct >= minimum else "failure"), text, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    ap.add_argument("--junit", action="append", default=[])
    ap.add_argument("--jacoco")
    ap.add_argument("--istanbul")
    ap.add_argument("--cobertura")
    ap.add_argument("--min-pass-rate", type=float, default=1.0)
    ap.add_argument("--new-code")
    ap.add_argument("--min-new-coverage", type=float, default=80.0)
    ap.add_argument("--sha", default=os.environ.get("REPORT_SHA") or os.environ.get("GITHUB_SHA"))
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    ap.add_argument("--url", default=f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                                     f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}")
    args = ap.parse_args()

    detail = ""
    if args.new_code is not None:
        state, text, detail = new_code(args.new_code, args.min_new_coverage)
    else:
        counts = junit(args.junit)
        cov = jacoco(args.jacoco) if args.jacoco else cobertura(args.cobertura) if args.cobertura else istanbul(args.istanbul)
        text = f"{counts['passed']} passed, {counts['failed']} failed, {counts['skipped']} skipped."
        if cov is not None:
            text += f" Coverage: {cov:.1f}%"
        ran = counts["passed"] + counts["failed"]
        rate = counts["passed"] / ran if ran else 0.0
        if args.min_pass_rate < 1.0 and ran:
            text += f" Pass rate: {rate * 100:.1f}% (at least {args.min_pass_rate * 100:.0f}%)."
        state = "success" if ran and (counts["failed"] == 0 or rate >= args.min_pass_rate) else "failure"
    print(f"{args.context}: {state}: {text}")

    with open(os.environ.get("GITHUB_STEP_SUMMARY", os.devnull), "a") as summary:
        summary.write(f"### {args.context}\n\n{text}\n" + (f"\nLines no test runs:\n\n{detail}\n" if detail else ""))
    token = os.environ.get("GITHUB_TOKEN")
    if not (token and args.repo and args.sha):
        print("No GITHUB_TOKEN, repo or sha: not posting a status", file=sys.stderr)
        return 0 if state == "success" else 1
    body = json.dumps({"state": state, "context": args.context, "description": text[:140],
                       "target_url": args.url}).encode()
    req = urllib.request.Request(f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com')}/repos/{args.repo}/statuses/{args.sha}",
                                 data=body, method="POST", headers={
                                     "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                     "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30).read()
    return 0 if state == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
