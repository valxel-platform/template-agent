"""Users with a feature on against users with it off, while the SDLC tool releases it gradually.

Every request that evaluates a flag (`flags.enabled`) counts once for that flag, on the side the
caller got, as failed when it ends in a 5xx or an exception, and as slow when it takes longer
than the run budget (`FLAG_COHORTS_SLOW_MS`, 60 s: an agent's run, not an API's answer). Once a
minute each flag's counts are logged as one line, which the tool sums across pods from
OpenChoreo's logs and compares: users who fail more, or wait longer, with the feature on stop its
release.

    vrixel.flag_cohorts {"flag": "emr.emr-feat-3.kiosk", "on": {"requests": 40, "errors": 0, "slow": 1},
                         "off": {"requests": 760, "errors": 2, "slow": 9}}

The product events a request records (`events.record`) count on the same side, so the tool can
also compare how often users with a feature on do what its goal counts:

    vrixel.flag_cohorts {"flag": "...", "on": {"requests": 40, "errors": 0, "slow": 1,
                         "events": {"referral_triaged": 9}}, "off": {...}}

Only counts leave the service: no user, request or value beyond the flag's key.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import threading
import time
from contextvars import ContextVar
from typing import Any

MARKER = "vrixel.flag_cohorts"
log = logging.getLogger("agent.cohorts")
_KEY = re.compile(r"[A-Za-z0-9._-]{1,200}")
_current: ContextVar[dict[str, bool] | None] = ContextVar("flag_cohorts", default=None)
_events: ContextVar[dict[str, int] | None] = ContextVar("flag_cohort_events", default=None)
_EVENT = re.compile(r"[a-z][a-z0-9_.]{0,99}")
# flag -> event -> [on, off]
_event_counts: dict[str, dict[str, list[int]]] = {}
# flag -> [on requests, on errors, on slow, off requests, off errors, off slow]
_counts: dict[str, list[int]] = {}
SLOW_MS = float(os.environ.get("FLAG_COHORTS_SLOW_MS", "60000"))
_lock = threading.Lock()
_reporter: threading.Thread | None = None


def seen(key: str, on: bool) -> None:
    """A flag was evaluated for the current request. Outside a request it isn't counted."""
    current = _current.get()
    if current is not None and _KEY.fullmatch(key or ""):
        current.setdefault(key, bool(on))


def event(name: str) -> None:
    """A product event happened in the current request (`events.record`); outside one it isn't counted."""
    current = _events.get()
    if current is not None and _EVENT.fullmatch(name or ""):
        current[name] = current.get(name, 0) + 1


def begin() -> Any:
    return (_current.set({}), _events.set({}))


def end(token: Any, failed: bool, elapsed_ms: float = 0.0) -> None:
    current = _current.get() or {}
    happened = _events.get() or {}
    flags_token, events_token = token if isinstance(token, tuple) else (token, None)
    _current.reset(flags_token)
    if events_token is not None:
        _events.reset(events_token)
    if not current:
        return
    with _lock:
        for key, on in current.items():
            c = _counts.setdefault(key, [0, 0, 0, 0, 0, 0])
            side = 0 if on else 3
            c[side] += 1
            if failed:
                c[side + 1] += 1
            if elapsed_ms > SLOW_MS:
                c[side + 2] += 1
            for name, n in happened.items():
                _event_counts.setdefault(key, {}).setdefault(name, [0, 0])[0 if on else 1] += n


def drain() -> list[str]:
    """The lines since the last call, one per flag, and the counts start again."""
    with _lock:
        counts = dict(_counts)
        events = dict(_event_counts)
        _counts.clear()
        _event_counts.clear()

    def side(c: list[int], i: int, ev: dict[str, list[int]]) -> dict[str, Any]:
        out: dict[str, Any] = {"requests": c[i * 3], "errors": c[i * 3 + 1], "slow": c[i * 3 + 2]}
        if ev:
            out["events"] = {name: n[i] for name, n in sorted(ev.items())}
        return out

    return [
        f"{MARKER} "
        + json.dumps(
            {"flag": k, "on": side(c, 0, events.get(k, {})), "off": side(c, 1, events.get(k, {}))},
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
        started = time.monotonic()
        failed = False
        try:
            await self.app(scope, receive, watch)
        except Exception:
            failed = True
            raise
        finally:
            end(token, failed or status["code"] >= 500, (time.monotonic() - started) * 1000)
