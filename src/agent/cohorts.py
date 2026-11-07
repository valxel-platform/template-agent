"""Users with a feature on against users with it off, while the SDLC tool releases it gradually.

Every request that evaluates a flag (`flags.enabled`) counts once for that flag, on the side the
caller got, and as failed when it ends in a 5xx or an exception. Once a minute each flag's counts
are logged as one line, which the tool sums across pods from OpenChoreo's logs and compares:
users who fail more with the feature on stop its release.

    vrixel.flag_cohorts {"flag": "emr.emr-feat-3.kiosk", "on": {"requests": 40, "errors": 0},
                         "off": {"requests": 760, "errors": 2}}

Only counts leave the service: no user, request or value beyond the flag's key.
"""

from __future__ import annotations

import atexit
import json
import logging
import re
import threading
from contextvars import ContextVar
from typing import Any

MARKER = "vrixel.flag_cohorts"
log = logging.getLogger("agent.cohorts")
_KEY = re.compile(r"[A-Za-z0-9._-]{1,200}")
_current: ContextVar[dict[str, bool] | None] = ContextVar("flag_cohorts", default=None)
_counts: dict[str, list[int]] = {}  # flag -> [on requests, on errors, off requests, off errors]
_lock = threading.Lock()
_reporter: threading.Thread | None = None


def seen(key: str, on: bool) -> None:
    """A flag was evaluated for the current request. Outside a request it isn't counted."""
    current = _current.get()
    if current is not None and _KEY.fullmatch(key or ""):
        current.setdefault(key, bool(on))


def begin() -> Any:
    return _current.set({})


def end(token: Any, failed: bool) -> None:
    current = _current.get() or {}
    _current.reset(token)
    if not current:
        return
    with _lock:
        for key, on in current.items():
            c = _counts.setdefault(key, [0, 0, 0, 0])
            side = 0 if on else 2
            c[side] += 1
            if failed:
                c[side + 1] += 1


def drain() -> list[str]:
    """The lines since the last call, one per flag, and the counts start again."""
    with _lock:
        counts = dict(_counts)
        _counts.clear()
    return [
        f"{MARKER} "
        + json.dumps(
            {"flag": k, "on": {"requests": c[0], "errors": c[1]}, "off": {"requests": c[2], "errors": c[3]}},
            separators=(",", ":"),
        )
        for k, c in sorted(counts.items())
    ]


def report() -> None:
    for line in drain():
        log.info(line)


def start_reporting(period_s: float = 60.0) -> None:
    """Log the counts once a minute, on a daemon thread, and once more at exit."""
    global _reporter
    if _reporter is not None:
        return
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(period_s):
            try:
                report()
            except Exception as exc:  # noqa: BLE001 - a report must never stop the reporter thread
                log.warning("flag cohorts not reported: %s", exc)

    _reporter = threading.Thread(target=loop, name="flag-cohorts", daemon=True)
    _reporter.start()
    atexit.register(report)


class CohortMiddleware:
    """Counts each HTTP request for the flags it evaluated: failed on a 5xx or an exception."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        status = {"code": 200}

        async def watch(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                status["code"] = int(message.get("status", 200))
            await send(message)

        token = begin()
        failed = False
        try:
            await self.app(scope, receive, watch)
        except Exception:
            failed = True
            raise
        finally:
            end(token, failed or status["code"] >= 500)
