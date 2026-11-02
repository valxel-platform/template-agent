"""Checks a deployed agent: it is up, it reaches its MCP servers, and it refuses callers without a
token. Writes JUnit for the post-deploy report. Starts no runs, so it is safe in production."""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree as ET


def status(url: str, method: str = "GET") -> int:
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"not an http(s) URL: {url}")
    req = urllib.request.Request(  # noqa: S310  (scheme checked above)
        url,
        method=method,
        data=b"{}" if method == "POST" else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310  (URL from the deploy event)
            return int(r.status)
    except urllib.error.HTTPError as e:
        return e.code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--junit", default="reports/smoke.xml")
    args = ap.parse_args(argv)
    base = args.base_url.rstrip("/")
    checks: dict[str, Callable[[], bool]] = {
        "healthz": lambda: status(f"{base}/healthz") == 200,
        "readyz (MCP servers reachable)": lambda: status(f"{base}/readyz") == 200,
        "runs need a token": lambda: status(f"{base}/runs", "POST") == 401,
    }
    suite = ET.Element("testsuite", name="smoke")
    failed = 0
    for name, check in checks.items():
        t0 = time.monotonic()
        try:
            ok = check()
        except (OSError, ValueError):
            ok = False
        tc = ET.SubElement(
            suite, "testcase", classname="smoke", name=name, time=f"{time.monotonic() - t0:.3f}"
        )
        if not ok:
            failed += 1
            ET.SubElement(tc, "failure", message=f"{name} failed at {base}")
        print(f"{'pass' if ok else 'FAIL'}  {name}")
    suite.set("tests", str(len(checks)))
    suite.set("failures", str(failed))
    Path(args.junit).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(args.junit, encoding="utf-8", xml_declaration=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
