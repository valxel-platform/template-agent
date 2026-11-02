#!/usr/bin/env python3
"""Post a test suite's results as a commit status the Vrixel SDLC reads.

The pipeline's quality policy matches suites by check name (unit*, contract*, integration*, e2e*,
smoke*) and reads counts and coverage from the summary text, e.g.
"128 passed, 0 failed, 2 skipped. Coverage: 83.5%". This script builds that text from JUnit XML
(any test runner) and a coverage report (JaCoCo XML or Istanbul coverage-summary.json), and posts
it as a commit status. Standard library only; runs on any GitHub-hosted runner.

usage: report.py --context "unit / report" --junit 'build/test-results/test/*.xml'
                 [--jacoco jacocoTestReport.xml | --istanbul coverage-summary.json | --cobertura coverage.xml]
                 [--sha SHA] [--repo owner/name] [--url URL]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    ap.add_argument("--junit", action="append", default=[])
    ap.add_argument("--jacoco")
    ap.add_argument("--istanbul")
    ap.add_argument("--cobertura")
    ap.add_argument("--sha", default=os.environ.get("REPORT_SHA") or os.environ.get("GITHUB_SHA"))
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    ap.add_argument("--url", default=f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                                     f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}")
    args = ap.parse_args()

    counts = junit(args.junit)
    cov = jacoco(args.jacoco) if args.jacoco else cobertura(args.cobertura) if args.cobertura else istanbul(args.istanbul)
    text = f"{counts['passed']} passed, {counts['failed']} failed, {counts['skipped']} skipped."
    if cov is not None:
        text += f" Coverage: {cov:.1f}%"
    state = "success" if counts["failed"] == 0 and sum(counts.values()) > 0 else "failure"
    print(f"{args.context}: {state}: {text}")

    with open(os.environ.get("GITHUB_STEP_SUMMARY", os.devnull), "a") as summary:
        summary.write(f"### {args.context}\n\n{text}\n")
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
