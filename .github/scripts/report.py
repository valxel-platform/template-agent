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
                 [--sha SHA] [--repo owner/name] [--url URL] [--min-pass-rate 0.95] [--must-pass 'leak:*']

       report.py --context "code-quality / new code" --new-code new-code.json [--min-new-coverage 80]

--min-pass-rate: the status is a success when at least this share of the tests passed (a live run
of an agent's evals against the real model can miss a case or two by chance). The summary always
gives the counts, so the Valxel SDLC sees the rate either way.

--must-pass: test cases (by name, shell-style patterns, repeatable) that must pass whatever the pass
rate: a knowledge repo's retrieval evals allow a few questions to miss (--min-pass-rate), but never
a passage shown to someone who must not see it ('leak:*') or a deleted record ('deleted:*').

--new-code: diff-cover's JSON report (`diff-cover <coverage> --compare-branch origin/<base>
--format json:new-code.json`), i.e. coverage of the lines this pull request changes. The status
fails below --min-new-coverage, and names the lines that no test runs.
"""

import argparse
import fnmatch
import glob
import http.client
import json
import os
import sys
import urllib.parse
from xml.parsers import expat


def elements(path: str) -> list[tuple[int, str, dict]]:
    """Every element of an XML report as (depth, tag, attributes). A report that declares an
    entity is refused, and external DTDs (JaCoCo names one) are never fetched."""
    out, depth = [], [0]

    def start(tag, attrs):
        out.append((depth[0], tag, attrs))
        depth[0] += 1

    def end(_tag):
        depth[0] -= 1

    def refuse(*_args):
        raise ValueError(f"{path}: XML entities aren't allowed in a test report")

    p = expat.ParserCreate()
    p.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    p.StartElementHandler, p.EndElementHandler = start, end
    p.EntityDeclHandler = refuse
    p.UnparsedEntityDeclHandler = refuse
    with open(path, "rb") as f:
        p.ParseFile(f)
    return out


def junit(patterns: list[str]) -> dict:
    total = failed = skipped = 0
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            els = elements(path)  # a report always has a root element (expat raises otherwise)
            suites = els[:1] if els[0][1] == "testsuite" else [e for e in els if e[1] == "testsuite"]
            for _, _, suite in suites:
                total += int(suite.get("tests", 0))
                failed += int(suite.get("failures", 0)) + int(suite.get("errors", 0))
                skipped += int(suite.get("skipped", 0))
    return {"passed": total - failed - skipped, "failed": failed, "skipped": skipped}


def failed_cases(patterns: list[str]) -> list[str]:
    """The names of the test cases that failed or errored."""
    out = []
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            case, case_depth = None, -1
            for depth, tag, attrs in elements(path):
                if case is not None and depth <= case_depth:
                    case = None
                if tag == "testcase":
                    case, case_depth = attrs.get("name", ""), depth
                elif case is not None and tag in ("failure", "error"):
                    out.append(case)
                    case = None
    return out


def jacoco(path: str) -> float | None:
    if not path or not os.path.exists(path):
        return None
    for depth, tag, c in elements(path):
        if depth == 1 and tag == "counter" and c.get("type") == "LINE":  # report-level counters
            covered, missed = int(c.get("covered")), int(c.get("missed"))
            return 100.0 * covered / max(1, covered + missed)
    return None


def cobertura(path: str) -> float | None:
    """coverage.py / pytest-cov XML (and other Cobertura producers)."""
    if not path or not os.path.exists(path):
        return None
    els = elements(path)
    return 100.0 * float(els[0][2].get("line-rate", 0)) if els else None


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
    ap.add_argument("--must-pass", action="append", default=[])
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
        if args.must_pass and counts["failed"]:
            broken = sorted({c for c in failed_cases(args.junit) if any(fnmatch.fnmatch(c, p) for p in args.must_pass)})
            if broken:
                state = "failure"
                text += f" Must pass, failed: {'; '.join(broken)[:200]}."
    print(f"{args.context}: {state}: {text}")

    with open(os.environ.get("GITHUB_STEP_SUMMARY", os.devnull), "a") as summary:
        summary.write(f"### {args.context}\n\n{text}\n" + (f"\nLines no test runs:\n\n{detail}\n" if detail else ""))
    token = os.environ.get("GITHUB_TOKEN")
    if not (token and args.repo and args.sha):
        print("No GITHUB_TOKEN, repo or sha: not posting a status", file=sys.stderr)
        return 0 if state == "success" else 1
    api = urllib.parse.urlsplit(os.environ.get("GITHUB_API_URL") or "https://api.github.com")
    if api.scheme != "https" or not api.hostname:
        raise SystemExit(f"GITHUB_API_URL must be an https URL, not {api.geturl()!r}")
    body = json.dumps({"state": state, "context": args.context, "description": text[:140],
                       "target_url": args.url})
    conn = http.client.HTTPSConnection(api.hostname, api.port or 443, timeout=30)
    conn.request("POST", f"{api.path.rstrip('/')}/repos/{args.repo}/statuses/{args.sha}", body=body, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "valxel-report",
        "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()
    if resp.status >= 300:
        raise SystemExit(f"Posting the status failed: HTTP {resp.status}")
    return 0 if state == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
